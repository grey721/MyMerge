"""Batched Transformers + PEFT inference with candidate-scoped LoRA fusion."""

from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

from ..core.config import AdapterConfig, ModelConfig
from ..core.domain import GenerationRequest, GenerationResult
from ..core.policies import FusionPolicy
from .algebra import fusion_algebra_from_model_config
from .base import ModelBackend, OptionalDependencyError
from .materialization import PeftCandidateFusion, PeftDirectAdapterMaterializer
from .peft_pool import PeftResidentAdapterPool


class PeftCausalLMBackend(ModelBackend):
    """Local CausalLM backend with topology-stage batch generation.

    Candidate-scoped on-demand fusion lazily compiles one temporary adapter per
    distinct Agent mixture, reuses it for the complete genome evaluation, and
    releases it when the candidate session exits.

    Calls sharing one mixture and decoding configuration are batched. Different
    mixtures remain serialized because PEFT ``set_adapter`` mutates model state;
    candidate-level parallelism should use one backend instance per GPU worker.
    The source pool is intentionally resident once per worker for throughput,
    while fused candidate adapters are candidate-scoped and ephemeral.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        adapters: Sequence[AdapterConfig],
    ) -> None:
        self.model_config = model_config
        self._lock = threading.RLock()
        self._fusion_algebra = fusion_algebra_from_model_config(model_config)
        try:
            self._torch, self._tokenizer, self._pool = self._load(adapters)
            self._candidate_fusion = PeftCandidateFusion(
                self._pool,
                algebra=self._fusion_algebra,
                materializer=PeftDirectAdapterMaterializer(
                    pool=self._pool,
                    algebra=self._fusion_algebra
                ),
            )
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    @property
    def _model(self):
        return self._pool.model

    @property
    def adapter_names(self) -> Tuple[str, ...]:
        return self._pool.adapter_names

    @property
    def adapter_specs(self) -> Tuple[AdapterConfig, ...]:
        return self._pool.specs

    def estimate_fusion_rank(self, mixture: FusionPolicy) -> float:
        return self._candidate_fusion.estimate_rank(mixture)

    def runtime_snapshot(self) -> Mapping[str, Any]:
        """Expose adapter lifecycle and CUDA memory without leaking internals."""

        loaded_names = self._pool.loaded_adapter_names
        source_names = self.adapter_names
        source_set = set(source_names)
        candidate_adapter_names = tuple(
            getattr(self._candidate_fusion, "resident_adapter_names", ())
        )
        snapshot = {
            "backend": type(self).__name__,
            "fusion_algebra": self._fusion_algebra.name,
            "device": str(_model_input_device(self._model)),
            "source_adapter_names": list(source_names),
            "loaded_adapter_names": list(loaded_names),
            "temporary_adapter_names": [
                name for name in loaded_names if name not in source_set
            ],
            "candidate_adapter_names": list(candidate_adapter_names),
            "adapter_pool": dict(self._pool.runtime_snapshot()),
            "active_adapter": _json_safe_adapter_state(
                getattr(self._model, "active_adapter", None)
            ),
            "fusion": dict(self._candidate_fusion.runtime_snapshot()),
        }
        if self._torch.cuda.is_available():
            device = self._torch.cuda.current_device()
            snapshot["cuda"] = {
                "device_index": int(device),
                "device_name": self._torch.cuda.get_device_name(device),
                "memory_allocated_bytes": int(
                    self._torch.cuda.memory_allocated(device)
                ),
                "memory_reserved_bytes": int(
                    self._torch.cuda.memory_reserved(device)
                ),
                "max_memory_allocated_bytes": int(
                    self._torch.cuda.max_memory_allocated(device)
                ),
            }
        return snapshot

    def release_unreferenced_adapters(
        self, reference_counts: Mapping[str, int]
    ) -> Mapping[str, Any]:
        """Release Stage-II-inaccessible PEFT source LoRAs at a barrier."""

        with self._lock:
            if self._candidate_fusion.resident_adapter_names:
                raise RuntimeError(
                    "Cannot prune the PEFT source pool during a candidate session"
                )
            report = self._pool.release_unreferenced(reference_counts)
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            return dict(report.as_dict())

    def _load(self, adapters: Sequence[AdapterConfig]):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "The PEFT backend requires the 'peft' optional dependencies. "
                "Install the project with: pip install -e '.[peft]'"
            ) from exc

        dtype = _resolve_torch_dtype(torch, self.model_config.dtype)
        model_kwargs = dict(self.model_config.options.get("model_kwargs", {}))
        model_kwargs.setdefault("torch_dtype", dtype)
        model_kwargs.setdefault("device_map", self.model_config.device_map)
        model_kwargs.setdefault(
            "trust_remote_code", self.model_config.trust_remote_code
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.name_or_path, **model_kwargs
        )
        tokenizer_kwargs = dict(
            self.model_config.options.get("tokenizer_kwargs", {})
        )
        tokenizer_kwargs.setdefault(
            "trust_remote_code", self.model_config.trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name_or_path, **tokenizer_kwargs
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Decoder-only generation must left-pad batched prompts so the newly
        # generated tokens begin at the same sequence position.
        tokenizer.padding_side = "left"

        pool = PeftResidentAdapterPool.from_pretrained(
            adapters,
            base_model,
            PeftModel,
        )
        return torch, tokenizer, pool

    def generate(
        self, request: GenerationRequest, mixture: FusionPolicy
    ) -> GenerationResult:
        return self.generate_many((request,), (mixture,))[0]

    def generate_many(
        self,
        requests: Sequence[GenerationRequest],
        mixtures: Sequence[FusionPolicy],
    ) -> Tuple[GenerationResult, ...]:
        """Generate requests in mixture/config-compatible batches.

        PEFT mutates the active adapter on the model, so requests using the
        same mixture and decoding parameters are grouped into one model call.
        Different mixtures remain correct and preserve their original order.
        """

        if len(requests) != len(mixtures):
            raise ValueError("requests and mixtures must have the same length")
        if not requests:
            return ()
        for mixture in mixtures:
            self._pool.validate_policy(mixture)

        grouped = {}  # type: Dict[Tuple[str, int, float, Any], List[int]]
        for index, (request, mixture) in enumerate(zip(requests, mixtures)):
            key = (
                mixture.fingerprint(),
                request.max_new_tokens,
                request.temperature,
                request.seed,
            )
            grouped.setdefault(key, []).append(index)

        results = [None] * len(requests)  # type: List[GenerationResult]
        for indices in grouped.values():
            batch_requests = tuple(requests[index] for index in indices)
            batch_mixture = mixtures[indices[0]]
            batch_results = self._generate_batch(batch_requests, batch_mixture)
            for index, result in zip(indices, batch_results):
                results[index] = result
        if any(result is None for result in results):
            raise RuntimeError("PEFT batch generation returned missing results")
        return tuple(results)

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        prompt = self._render_prompt(request)
        return len(
            self._tokenizer.encode(prompt, add_special_tokens=False)
        )

    def _render_prompt(self, request: GenerationRequest) -> str:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        chat_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if not self.model_config.generation.enable_thinking:
            chat_kwargs["enable_thinking"] = False
        try:
            return self._tokenizer.apply_chat_template(
                messages, **chat_kwargs
            )
        except TypeError:
            chat_kwargs.pop("enable_thinking", None)
            return self._tokenizer.apply_chat_template(
                messages, **chat_kwargs
            )

    def _generate_batch(
        self,
        requests: Sequence[GenerationRequest],
        mixture: FusionPolicy,
    ) -> Tuple[GenerationResult, ...]:
        self._pool.validate_policy(mixture)
        with self._lock:
            prompts = [self._render_prompt(request) for request in requests]

            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
            device = _model_input_device(self._model)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_width = int(inputs["input_ids"].shape[-1])
            input_lengths = inputs["attention_mask"].sum(dim=-1).tolist()

            generation_kwargs = dict(
                self.model_config.options.get("generation_kwargs", {})
            )
            generation_kwargs.setdefault(
                "max_new_tokens", requests[0].max_new_tokens
            )
            generation_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)
            temperature = requests[0].temperature
            if temperature > 0.0:
                generation_kwargs.setdefault("do_sample", True)
                generation_kwargs.setdefault("temperature", temperature)
            else:
                generation_kwargs.setdefault("do_sample", False)
                # A local generation_config.json may carry sampling defaults.
                # Neutralize them for greedy requests without changing output.
                generation_kwargs.setdefault("temperature", 1.0)
                generation_kwargs.setdefault("top_p", 1.0)
                generation_kwargs.setdefault("top_k", 50)

            if requests[0].seed is not None:
                self._torch.manual_seed(requests[0].seed)
                if self._torch.cuda.is_available():
                    self._torch.cuda.manual_seed_all(requests[0].seed)

            _synchronize_cuda(self._torch)
            batch_start = time.perf_counter()
            activation_start = time.perf_counter()
            with self._mixture_context(mixture) as adapter_name:
                _synchronize_cuda(self._torch)
                activation_latency = time.perf_counter() - activation_start
                generation_start = time.perf_counter()
                with self._torch.inference_mode():
                    outputs = self._model.generate(**inputs, **generation_kwargs)
                _synchronize_cuda(self._torch)
                model_generation_latency = (
                    time.perf_counter() - generation_start
                )
            _synchronize_cuda(self._torch)
            latency = time.perf_counter() - batch_start
            cleanup_latency = max(
                0.0,
                latency - activation_latency - model_generation_latency,
            )

            generated = outputs[:, input_width:]
            amortized_latency = latency / max(1, len(requests))
            results = []
            for index, _ in enumerate(requests):
                generated_row = generated[index]
                text = self._tokenizer.decode(
                    generated_row, skip_special_tokens=True
                ).strip()
                results.append(
                    GenerationResult(
                        text=text,
                        latency_seconds=amortized_latency,
                        prompt_tokens=int(input_lengths[index]),
                        completion_tokens=_count_completion_tokens(
                            generated_row, self._tokenizer.pad_token_id
                        ),
                        metadata={
                            "backend": "peft",
                            "adapter": adapter_name,
                            "mixture": mixture.as_dict(),
                            "batch_size": len(requests),
                            "batch_latency_seconds": latency,
                            "activation_latency_seconds": activation_latency,
                            "model_generation_latency_seconds": (
                                model_generation_latency
                            ),
                            "fusion_cleanup_latency_seconds": cleanup_latency,
                            "latency_accounting": "amortized_batch",
                        },
                    )
                )
            return tuple(results)

    def _mixture_context(self, mixture: FusionPolicy):
        return self._candidate_fusion.activate(self._model, mixture)

    @contextmanager
    def candidate_session(self) -> Iterator[None]:
        """Keep candidate-scoped compiled adapters alive across all cases."""

        with self._lock:
            with self._candidate_fusion.candidate_session(self._model):
                yield

    def close(self) -> None:
        failures = []
        with self._lock:
            if hasattr(self, "_pool"):
                model = self._pool.model
                if hasattr(self, "_candidate_fusion"):
                    try:
                        self._candidate_fusion.close(model)
                    except Exception as exc:
                        failures.append(("candidate_fusion", exc))
                    finally:
                        del self._candidate_fusion
                try:
                    self._pool.close()
                except Exception as exc:
                    failures.append(("adapter_pool", exc))
                del self._pool
            gc.collect()
            if hasattr(self, "_torch") and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        if failures:
            raise RuntimeError(
                "PEFT backend cleanup failed: {}".format(
                    "; ".join(
                        "{}: {}".format(name, exc) for name, exc in failures
                    )
                )
            )


def _count_completion_tokens(token_ids, pad_token_id) -> int:
    """Count non-padding generated tokens in a padded decoder-only batch."""

    values = token_ids.tolist()
    if pad_token_id is not None:
        while values and values[-1] == pad_token_id:
            values.pop()
    return len(values)


def _resolve_torch_dtype(torch_module, dtype_name: str):
    normalized = dtype_name.lower()
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    normalized = aliases.get(normalized, normalized)
    if not hasattr(torch_module, normalized):
        raise ValueError("Unsupported torch dtype {!r}".format(dtype_name))
    return getattr(torch_module, normalized)


def _model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def _json_safe_adapter_state(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return str(value)


def _synchronize_cuda(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
