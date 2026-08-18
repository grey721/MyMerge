"""Optional vLLM backend with candidate-scoped in-memory LoRA fusion."""

from __future__ import annotations

import gc
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ...core.config import AdapterConfig, ModelConfig
from ...core.domain import GenerationRequest, GenerationResult
from ...core.policies import FusionPolicy, uniform_policy_mixture
from ..algebra import (
    FusionAlgebra,
    create_fusion_algebra,
    fusion_algebra_from_model_config,
)
from ..base import ModelBackend, OptionalDependencyError
from ..materialization import BASE_ADAPTER_NAME
from ..resources import FusedAdapterResource, FusionAdapterFactory
from .materialization import (
    MaterializedTensorLoRA,
    VllmResidentAdapterPool,
    VllmTensorAdapterMaterializer,
    supported_rank_ceiling,
)
from .transport import VllmTensorLoraTransport


class _VllmFusedAdapter(FusedAdapterResource):
    """One registered candidate LoRA plus its owned in-memory tensors."""

    def __init__(
        self,
        policy: FusionPolicy,
        request: Any,
        materialized: MaterializedTensorLoRA,
        factory: "VllmFusionAdapterFactory",
    ) -> None:
        self.request = request
        self.materialized = materialized
        self._source_keys = tuple(materialized.source_keys)
        self._policy_fingerprint = policy.fingerprint()
        self._factory = factory
        self._closed = False

    @property
    def adapter_key(self) -> str:
        return self.materialized.name

    @property
    def source_keys(self) -> Tuple[str, ...]:
        return self._source_keys

    @property
    def policy_fingerprint(self) -> str:
        return self._policy_fingerprint

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def resident_bytes(self) -> int:
        if self._closed:
            return 0
        return self.materialized.resident_bytes

    @property
    def device(self) -> Optional[str]:
        return None if self._closed else self.materialized.device

    def close(self) -> None:
        if self._closed:
            return
        self._factory.release(self)

    def _mark_closed(self) -> None:
        self._closed = True
        self._factory = None


class VllmFusionAdapterFactory(FusionAdapterFactory):
    """Materialize, register and release vLLM candidate Adapter resources."""

    def __init__(
        self,
        transport: VllmTensorLoraTransport,
        pool: VllmResidentAdapterPool,
        materializer: VllmTensorAdapterMaterializer,
    ) -> None:
        self.transport = transport
        self.pool = pool
        self.materializer = materializer
        self._next_request_id = pool.size + 1
        self.registration_seconds = 0.0
        self.unregistration_seconds = 0.0
        self.registration_count = 0
        self.unregistration_count = 0

    def create(self, policy: FusionPolicy) -> FusedAdapterResource:
        request_id = self._next_request_id
        self._next_request_id += 1
        name = "mix_candidate_{}".format(policy.fingerprint())
        materialized = self.materializer.materialize(policy, name)
        request = self.transport.make_request(
            name=name,
            request_id=request_id,
            peft_config=materialized.peft_config,
            tensors=materialized.tensors,
        )
        started = time.perf_counter()
        try:
            registered = self.transport.register(request)
        except Exception:
            self.materializer.release(materialized)
            raise
        finally:
            self.registration_seconds += time.perf_counter() - started
            self.registration_count += 1
        if registered is False:
            self.materializer.release(materialized)
            raise RuntimeError(
                "vLLM rejected new candidate LoRA ID {}".format(request_id)
            )
        return _VllmFusedAdapter(policy, request, materialized, self)

    def release(self, resource: _VllmFusedAdapter) -> None:
        started = time.perf_counter()
        try:
            removed = self.transport.unregister(
                int(resource.request.lora_int_id)
            )
        finally:
            self.unregistration_seconds += time.perf_counter() - started
            self.unregistration_count += 1
        if removed is False:
            # The engine no longer owns the registration, so local tensors can
            # be reclaimed. Still report the inconsistent lifecycle once.
            self.materializer.release(resource.materialized)
            resource._mark_closed()
            raise RuntimeError("Adapter was not registered")
        self.materializer.release(resource.materialized)
        resource._mark_closed()


class VllmCandidateFusion:
    """Map backend-neutral policies to vLLM LoRA IDs for one candidate."""

    def __init__(
        self,
        transport: VllmTensorLoraTransport,
        pool: VllmResidentAdapterPool,
        algebra: Optional[FusionAlgebra] = None,
        materializer: Optional[VllmTensorAdapterMaterializer] = None,
    ) -> None:
        self.transport = transport
        self.pool = pool
        self.algebra = algebra or create_fusion_algebra()
        self.materializer = materializer or VllmTensorAdapterMaterializer(
            pool, algebra=self.algebra
        )
        self.factory = VllmFusionAdapterFactory(
            transport,
            pool,
            self.materializer,
        )
        self._session_handles = None  # type: Optional[OrderedDict[str, FusedAdapterResource]]
        self._unreleased_handles = OrderedDict()  # type: OrderedDict[str, FusedAdapterResource]

    @property
    def resident_temporary_names(self) -> Tuple[str, ...]:
        handles = list((self._session_handles or {}).values())
        handles.extend(self._unreleased_handles.values())
        return tuple(dict.fromkeys(
            handle.adapter_key
            for handle in handles
            if not handle.closed
        ))

    def estimate_rank(self, policy: FusionPolicy) -> float:
        return float(
            self.algebra.estimate_rank(policy, self.pool.descriptors)
        )

    @contextmanager
    def candidate_session(self) -> Iterator[None]:
        if self._session_handles is not None:
            raise RuntimeError("Nested vLLM candidate sessions are unsupported")
        if self._unreleased_handles:
            self._release_handles(self._unreleased_handles)
        handles = OrderedDict()  # type: OrderedDict[str, FusedAdapterResource]
        self._session_handles = handles
        try:
            yield
        finally:
            try:
                self._unreleased_handles.update(handles)
                handles.clear()
                self._release_handles(self._unreleased_handles)
            finally:
                self._session_handles = None

    def request_for(self, policy: FusionPolicy):
        uniform = uniform_policy_mixture(policy)
        if uniform is not None:
            if not uniform.active:
                return None, BASE_ADAPTER_NAME
            if len(uniform.active) == 1 and abs(uniform.active[0][1] - 1.0) <= 1e-10:
                adapter_name = uniform.active[0][0]
                request = self.pool.request_for_source(adapter_name)
                return request, adapter_name

        if self._session_handles is None:
            raise RuntimeError(
                "Mixed-adapter vLLM inference requires candidate_session()"
            )
        fingerprint = policy.fingerprint()
        handle = self._session_handles.get(fingerprint)
        if handle is None:
            handle = self.factory.create(policy)
            self._session_handles[fingerprint] = handle
        return handle.request, handle.adapter_key

    def close(self) -> None:
        if self._session_handles is not None:
            self._unreleased_handles.update(self._session_handles)
            self._session_handles.clear()
            self._session_handles = None
        if self._unreleased_handles:
            self._release_handles(self._unreleased_handles)

    def runtime_snapshot(self) -> Mapping[str, Any]:
        materializer = dict(self.materializer.runtime_snapshot())
        materializer.update(
            {
                "registration_count": self.factory.registration_count,
                "registration_seconds": self.factory.registration_seconds,
                "unregistration_count": self.factory.unregistration_count,
                "unregistration_seconds": self.factory.unregistration_seconds,
            }
        )
        return {
            "implementation": type(self).__name__,
            "algebra": self.algebra.name,
            "resident_temporary_count": len(self.resident_temporary_names),
            "unreleased_adapter_names": list(self.resident_temporary_names),
            "materializer": materializer,
        }

    def _release_handles(self, handles: OrderedDict) -> None:
        failures = []
        for fingerprint, handle in tuple(handles.items()):
            try:
                handle.close()
            except Exception as exc:
                failures.append((handle.adapter_key, exc))
            else:
                handles.pop(fingerprint, None)
        if failures:
            raise RuntimeError(
                "Failed to release vLLM candidate LoRA(s): {}".format(
                    "; ".join("{}: {}".format(name, exc) for name, exc in failures)
                )
            )


class VllmCausalLMBackend(ModelBackend):
    """vLLM 0.10 offline engine with in-memory, candidate-scoped LoRAs.

    One instance belongs to one persistent AgentMerge GPU worker.  vLLM
    performs continuous batching over all prompts submitted by one topology
    stage and accepts heterogeneous per-prompt LoRA IDs up to ``max_loras``.
    Candidate-level distribution remains the outer runtime's responsibility.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        adapters: Sequence[AdapterConfig],
        inference_batch_size: int,
        candidate_lora_capacity: int = 4,
    ) -> None:
        self.model_config = model_config
        self._candidate_lora_capacity = max(1, int(candidate_lora_capacity))
        self._inference_batch_size = int(inference_batch_size)
        if self._inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        self._lock = threading.RLock()
        self._fusion_algebra = fusion_algebra_from_model_config(model_config)
        self._pool = VllmResidentAdapterPool(adapters)
        try:
            self._llm, self._tokenizer, self._transport = self._load_engine()
            self._pool.move_to(self._source_adapter_device)
            self._pool.bind_transport(self._transport)
            self._fusion = VllmCandidateFusion(
                self._transport,
                self._pool,
                algebra=self._fusion_algebra,
            )
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise
        self._generation_calls = 0
        self._generated_prompts = 0
        self._generation_seconds = 0.0
        self._heterogeneous_lora_batch_count = 0
        self._max_distinct_loras_per_batch = 0

    @property
    def adapter_names(self) -> Tuple[str, ...]:
        return self._pool.adapter_names

    @property
    def adapter_specs(self) -> Tuple[AdapterConfig, ...]:
        return self._pool.specs

    def estimate_fusion_rank(self, mixture: FusionPolicy) -> float:
        return self._fusion.estimate_rank(mixture)

    def _load_engine(self):
        options = _vllm_options(self.model_config)
        transport = str(options.pop("lora_transport", "memory")).lower()
        if transport != "memory":
            raise ValueError(
                "The current vLLM backend supports lora_transport=memory only"
            )
        tensor_parallel_size = int(options.pop("tensor_parallel_size", 1))
        if tensor_parallel_size != 1:
            raise ValueError(
                "AgentMerge vLLM memory backend requires tensor_parallel_size=1; "
                "use multiple persistent one-GPU workers via --device"
            )
        self._source_adapter_device = str(
            options.pop("source_adapter_device", "cuda")
        ).lower()
        if self._source_adapter_device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("vllm.source_adapter_device must be cpu or cuda")
        engine_mode = str(options.pop("engine_mode", "v1"))
        transport = VllmTensorLoraTransport.prepare(
            engine_mode, load_device=self._source_adapter_device
        )
        self._distributed_port = _configure_vllm_process_port()
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - handled above too
            raise OptionalDependencyError(
                "Install the vLLM backend with: pip install -e '.[vllm]'"
            ) from exc

        max_rank = int(
            options.pop(
                "max_lora_rank", supported_rank_ceiling(self._pool.total_rank)
            )
        )
        if max_rank < self._pool.total_rank:
            raise ValueError(
                "vllm.max_lora_rank={} is smaller than maximum fused rank {}".format(
                    max_rank, self._pool.total_rank
                )
            )
        max_loras = int(options.pop("max_loras", 1))
        if max_loras < 1:
            raise ValueError("vllm.max_loras must be positive")
        self._max_loras = max_loras
        max_cpu_loras = int(
            options.pop(
                "max_cpu_loras",
                self._pool.size + max(4, self._candidate_lora_capacity),
            )
        )
        required_cpu_loras = self._pool.size + self._candidate_lora_capacity
        if max_cpu_loras < required_cpu_loras:
            raise ValueError(
                "vllm.max_cpu_loras must be at least {} to hold the source "
                "pool and candidate Agent LoRAs".format(required_cpu_loras)
            )
        self._max_cpu_loras = max_cpu_loras
        if "max_num_seqs" in options:
            raise ValueError(
                "vllm.max_num_seqs is derived from inference_batch_size"
            )
        self._max_num_seqs = self._inference_batch_size
        self._kv_cache_memory_bytes = options.get("kv_cache_memory_bytes")
        self._prefix_caching_enabled = bool(
            options.pop("enable_prefix_caching", True)
        )
        engine_kwargs = {
            "model": self.model_config.name_or_path,
            "tokenizer": self.model_config.name_or_path,
            "dtype": _vllm_dtype(self.model_config.dtype),
            "trust_remote_code": self.model_config.trust_remote_code,
            "tensor_parallel_size": 1,
            "seed": self.model_config.generation.seed or 0,
            "enable_lora": True,
            "max_lora_rank": max_rank,
            "max_loras": max_loras,
            "max_cpu_loras": max_cpu_loras,
            # AgentMerge submits topology-stage batches, not serving-scale
            # hundreds of concurrent requests. A bounded capture set reduces
            # CUDA-graph warm-up time and reserved memory substantially.
            "max_num_seqs": self._max_num_seqs,
            "enable_prefix_caching": self._prefix_caching_enabled,
            "disable_log_stats": True,
        }
        engine_kwargs.update(options)
        try:
            llm = LLM(**engine_kwargs)
        except Exception:
            _cleanup_vllm_environment()
            raise
        transport.bind_engine(llm.llm_engine)
        tokenizer = llm.get_tokenizer()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return llm, tokenizer, transport

    def generate(
        self, request: GenerationRequest, mixture: FusionPolicy
    ) -> GenerationResult:
        return self.generate_many((request,), (mixture,))[0]

    def generate_many(
        self,
        requests: Sequence[GenerationRequest],
        mixtures: Sequence[FusionPolicy],
    ) -> Tuple[GenerationResult, ...]:
        if len(requests) != len(mixtures):
            raise ValueError("requests and mixtures must have the same length")
        if not requests:
            return ()
        for mixture in mixtures:
            self._pool.validate_policy(mixture)

        results = [None] * len(requests)  # type: List[Optional[GenerationResult]]
        with self._lock:
            resolved = tuple(
                self._fusion.request_for(policy) for policy in mixtures
            )
            for indices in self._pack_lora_batches(resolved):
                group_requests = tuple(requests[index] for index in indices)
                group_loras = tuple(resolved[index] for index in indices)
                outputs = self._generate_group(group_requests, group_loras)
                for index, output in zip(indices, outputs):
                    results[index] = output
        if any(result is None for result in results):
            raise RuntimeError("vLLM batch generation returned missing results")
        return tuple(results)  # type: ignore[return-value]

    def _generate_group(
        self,
        requests: Sequence[GenerationRequest],
        resolved_loras: Sequence[Tuple[Any, str]],
    ) -> Tuple[GenerationResult, ...]:
        prompts = [self._render_prompt(request) for request in requests]
        sampling_params = [self._sampling_params(request) for request in requests]
        lora_requests = [item[0] for item in resolved_loras]
        adapter_names = [item[1] for item in resolved_loras]
        started = time.perf_counter()
        outputs = self._llm.generate(
            prompts,
            sampling_params,
            use_tqdm=False,
            # vLLM 0.10 accepts one LoRARequest per prompt. This keeps unlike
            # policies in the same scheduler call whenever max_loras permits.
            lora_request=lora_requests,
        )
        elapsed = time.perf_counter() - started
        self._generation_calls += 1
        self._generated_prompts += len(prompts)
        self._generation_seconds += elapsed
        distinct_lora_ids = {
            int(request.lora_int_id)
            for request in lora_requests
            if request is not None
        }
        if len(distinct_lora_ids) > 1:
            self._heterogeneous_lora_batch_count += 1
        self._max_distinct_loras_per_batch = max(
            self._max_distinct_loras_per_batch,
            len(distinct_lora_ids),
        )
        amortized = elapsed / max(1, len(prompts))
        results = []
        for output, adapter_name in zip(outputs, adapter_names):
            if not output.outputs:
                raise RuntimeError("vLLM returned no completion")
            completion = output.outputs[0]
            prompt_tokens = len(output.prompt_token_ids or ())
            completion_tokens = len(completion.token_ids or ())
            results.append(
                GenerationResult(
                    text=str(completion.text).strip(),
                    latency_seconds=amortized,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    metadata={
                        "backend": "vllm",
                        "adapter": adapter_name,
                        "batch_size": len(prompts),
                        "batch_latency_seconds": elapsed,
                        "model_generation_latency_seconds": elapsed,
                        "latency_accounting": "amortized_batch",
                    },
                )
            )
        return tuple(results)

    def _pack_lora_batches(
        self, resolved_loras: Sequence[Tuple[Any, str]]
    ) -> Tuple[Tuple[int, ...], ...]:
        """Greedily pack policies up to vLLM's simultaneous-LoRA capacity."""

        batches = []
        current = []
        request_ids = set()
        for index, (request, _) in enumerate(resolved_loras):
            request_id = (
                None if request is None else int(request.lora_int_id)
            )
            needed = request_id is not None and request_id not in request_ids
            if current and needed and len(request_ids) >= self._max_loras:
                batches.append(tuple(current))
                current = []
                request_ids = set()
            current.append(index)
            if request_id is not None:
                request_ids.add(request_id)
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def _render_prompt(self, request: GenerationRequest) -> str:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if not self.model_config.generation.enable_thinking:
            kwargs["enable_thinking"] = False
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        prompt = self._render_prompt(request)
        return len(
            self._tokenizer.encode(prompt, add_special_tokens=False)
        )

    def _sampling_params(self, request: GenerationRequest):
        from vllm import SamplingParams

        kwargs = dict(self.model_config.options.get("generation_kwargs", {}))
        kwargs.pop("do_sample", None)
        kwargs.pop("pad_token_id", None)
        kwargs.setdefault("max_tokens", request.max_new_tokens)
        kwargs.setdefault("temperature", request.temperature)
        if request.temperature <= 0.0:
            kwargs.setdefault("top_p", 1.0)
        if request.seed is not None:
            kwargs.setdefault("seed", request.seed)
        return SamplingParams(**kwargs)

    @contextmanager
    def candidate_session(self) -> Iterator[None]:
        with self._lock:
            with self._fusion.candidate_session():
                yield

    def runtime_snapshot(self) -> Mapping[str, Any]:
        temporary_names = list(self._fusion.resident_temporary_names)
        return {
            "backend": type(self).__name__,
            "engine": "vllm",
            "engine_mode": self._transport.display_engine_mode,
            "lora_transport": self._transport.name,
            "fusion_algebra": self._fusion_algebra.name,
            "max_simultaneous_loras": self._max_loras,
            "max_cpu_loras": self._max_cpu_loras,
            "max_num_seqs": self._max_num_seqs,
            "max_num_seqs_source": "benchmark.inference_batch_size",
            "kv_cache_memory_bytes": self._kv_cache_memory_bytes,
            "prefix_caching_enabled": self._prefix_caching_enabled,
            "source_adapter_names": list(self.adapter_names),
            # Common lifecycle fields also exposed by the PEFT backend. The
            # source tensors are resident even when vLLM has not activated a
            # source LoRA in its GPU LRU yet.
            "loaded_adapter_names": list(self._pool.resident_adapter_names),
            "temporary_adapter_names": temporary_names,
            "candidate_adapter_names": temporary_names,
            "adapter_pool": dict(self._pool.runtime_snapshot()),
            "transport": dict(self._transport.runtime_snapshot()),
            "distributed_port": self._distributed_port,
            "generation_calls": self._generation_calls,
            "generated_prompts": self._generated_prompts,
            "generation_seconds": self._generation_seconds,
            "heterogeneous_lora_batch_count": (
                self._heterogeneous_lora_batch_count
            ),
            "max_distinct_loras_per_batch": self._max_distinct_loras_per_batch,
            "fusion": dict(self._fusion.runtime_snapshot()),
        }

    def release_unreferenced_adapters(
        self, reference_counts: Mapping[str, int]
    ) -> Mapping[str, Any]:
        """Unregister and close Stage-II-inaccessible tensor LoRAs."""

        with self._lock:
            if self._fusion.resident_temporary_names:
                raise RuntimeError(
                    "Cannot prune the vLLM source pool during a candidate session"
                )
            report = self._pool.release_unreferenced(reference_counts)
            return dict(report.as_dict())

    def close(self) -> None:
        failures = []
        with self._lock:
            if hasattr(self, "_fusion"):
                try:
                    self._fusion.close()
                except Exception as exc:
                    failures.append(("candidate_fusion", exc))
                finally:
                    del self._fusion
            # Source LoRAs must be unregistered while the engine is alive.
            if hasattr(self, "_pool"):
                try:
                    self._pool.close()
                except Exception as exc:
                    failures.append(("adapter_pool", exc))
            if hasattr(self, "_llm"):
                engine = getattr(self._llm, "llm_engine", None)
                executor = getattr(engine, "model_executor", None)
                if executor is not None:
                    try:
                        executor.shutdown()
                    except Exception as exc:
                        failures.append(("vllm_executor", exc))
                    finally:
                        try:
                            engine.model_executor = None
                        except Exception:
                            pass
                del self._llm
            if hasattr(self, "_transport"):
                self._transport.close()
                del self._transport
            if hasattr(self, "_pool"):
                del self._pool
            gc.collect()
            _cleanup_vllm_environment()
        if failures:
            raise RuntimeError(
                "vLLM backend cleanup failed: {}".format(
                    "; ".join(
                        "{}: {}".format(name, exc) for name, exc in failures
                    )
                )
            )


def _vllm_options(model_config: ModelConfig) -> Dict[str, Any]:
    raw = model_config.options.get("vllm", {})
    if not isinstance(raw, Mapping):
        raise ValueError("model.options.vllm must be a mapping")
    return dict(raw)


def _vllm_dtype(value: str) -> str:
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    return aliases.get(value.lower(), value.lower())


def _configure_vllm_process_port() -> int:
    """Give every outer GPU worker its own vLLM rendezvous range.

    vLLM's default open-port probe releases its socket before torch.distributed
    binds it.  Concurrent AgentMerge workers can consequently choose the same
    endpoint.  A PID-derived starting port removes that self-race; vLLM still
    increments the value when an unrelated service already owns it.
    """

    lower = 12000
    span = 48000
    port = lower + (os.getpid() % span)
    os.environ["VLLM_PORT"] = str(port)
    return port


def _cleanup_vllm_environment() -> None:
    """Release vLLM's process group and cached device allocations."""

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:
        # Import errors are possible when construction failed before vLLM was
        # fully imported. Runtime errors here must not hide the primary error.
        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass


__all__ = [
    "VllmCandidateFusion",
    "VllmCausalLMBackend",
    "VllmFusionAdapterFactory",
]
