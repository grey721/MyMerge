"""Version-scoped vLLM 0.10 tensor-LoRA compatibility layer.

vLLM's public request accepts a local adapter path.  PettingLLMs/VERL uses the
same narrow extension point implemented here: attach PEFT metadata and tensors
to ``LoRARequest`` and replace only the worker's adapter loader.  The patch is
installed before engine construction and is intentionally isolated from the
rest of AgentMerge.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Type

from ..base import OptionalDependencyError


_TENSOR_REQUEST_CLASS: Optional[Type[Any]] = None
_PATCHED_CLASS: Optional[Type[Any]] = None
_PATCHED_TOKENIZER_CLASS: Optional[Type[Any]] = None


def install_tensor_lora_support() -> Type[Any]:
    """Install the in-memory loader and return ``TensorLoRARequest``."""

    global _TENSOR_REQUEST_CLASS, _PATCHED_CLASS, _PATCHED_TOKENIZER_CLASS
    try:
        import msgspec
        import vllm
        from packaging.version import Version
        from vllm.lora.models import LoRAModel
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.lora.request import LoRARequest
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        from vllm.transformers_utils.tokenizer_group import TokenizerGroup
    except ImportError as exc:
        raise OptionalDependencyError(
            "The vLLM backend requires vLLM 0.10.x. Install the project in a "
            "dedicated environment with: pip install -e '.[vllm]'"
        ) from exc

    current = Version(str(vllm.__version__).split("+")[0])
    if current < Version("0.10.0") or current >= Version("0.11.0"):
        raise OptionalDependencyError(
            "AgentMerge tensor-LoRA currently supports vLLM >=0.10,<0.11; "
            "found {}".format(vllm.__version__)
        )

    if _TENSOR_REQUEST_CLASS is None:

        class TensorLoRARequest(LoRARequest):
            peft_config: dict = msgspec.field(default_factory=dict)
            lora_tensors: dict = msgspec.field(default_factory=dict)
            load_device: str = "cuda"

        TensorLoRARequest.__name__ = "TensorLoRARequest"
        TensorLoRARequest.__qualname__ = "TensorLoRARequest"
        TensorLoRARequest.__module__ = __name__
        _TENSOR_REQUEST_CLASS = TensorLoRARequest

    if _PATCHED_TOKENIZER_CLASS is not TokenizerGroup:
        original_get_lora_tokenizer = TokenizerGroup.get_lora_tokenizer
        tensor_request_class = _TENSOR_REQUEST_CLASS

        def _get_lora_tokenizer(self, lora_request=None):
            # AgentMerge audits factor-only LoRAs and does not support adapter
            # vocabularies. Reuse the resident base tokenizer instead of
            # attempting one filesystem/network tokenizer load per candidate.
            if isinstance(lora_request, tensor_request_class):
                return self.tokenizer
            return original_get_lora_tokenizer(self, lora_request)

        TokenizerGroup.get_lora_tokenizer = _get_lora_tokenizer
        _PATCHED_TOKENIZER_CLASS = TokenizerGroup

    if _PATCHED_CLASS is LRUCacheWorkerLoRAManager:
        return _TENSOR_REQUEST_CLASS

    original_loader = LRUCacheWorkerLoRAManager._load_adapter
    tensor_request_class = _TENSOR_REQUEST_CLASS

    def _load_adapter(self, request):
        if not isinstance(request, tensor_request_class):
            return original_loader(self, request)

        helper = PEFTHelper.from_dict(dict(request.peft_config))
        helper.validate_legal(self.lora_config)
        tensors = protect_resident_source_tensors(
            request.lora_tensors,
            helper.vllm_lora_scaling_factor,
        )
        model = self._adapter_manager.model
        weights_mapper = getattr(model, "hf_to_vllm_mapper", None)
        lora = LoRAModel.from_lora_tensors(
            lora_model_id=request.lora_int_id,
            tensors=tensors,
            peft_helper=helper,
            device=request.load_device,
            dtype=self.lora_config.lora_dtype,
            embeddings=None,
            target_embedding_padding=(
                self.vocab_size + self.lora_config.lora_extra_vocab_size
            ),
            embedding_modules=self.embedding_modules,
            embedding_padding_modules=self.embedding_padding_modules,
            weights_mapper=weights_mapper,
        )
        if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
            raise ValueError(
                "LoRA added vocab size {} exceeds configured {}".format(
                    lora.extra_vocab_size,
                    self.lora_config.lora_extra_vocab_size,
                )
            )
        return lora

    LRUCacheWorkerLoRAManager._load_adapter = _load_adapter
    _PATCHED_CLASS = LRUCacheWorkerLoRAManager
    return _TENSOR_REQUEST_CLASS


def protect_resident_source_tensors(
    tensors: Mapping[str, Any], scaling: float
) -> Mapping[str, Any]:
    """Protect resident B factors from vLLM's in-place scale folding.

    ``LoRALayerWeights.optimize`` multiplies every B factor in place whenever
    the PEFT scaling is not one.  With CUDA-resident input tensors the vLLM
    conversion can otherwise retain a view of the source pool and permanently
    alter it.  Candidate LoRAs already bake their scale into B and therefore
    take the zero-copy path.
    """

    if abs(float(scaling) - 1.0) <= 1e-12:
        return tensors
    return {
        name: (tensor.clone() if name.endswith(".lora_B.weight") else tensor)
        for name, tensor in tensors.items()
    }


def make_tensor_lora_request(
    request_class: Type[Any],
    *,
    name: str,
    request_id: int,
    peft_config,
    tensors,
    load_device: str = "cuda",
):
    """Construct a request with a non-empty placeholder path for vLLM."""

    return request_class(
        lora_name=name,
        lora_int_id=int(request_id),
        lora_path="agentmerge://in-memory/{}".format(name),
        peft_config=dict(peft_config),
        lora_tensors=tensors,
        load_device=str(load_device),
    )


__all__ = [
    "install_tensor_lora_support",
    "make_tensor_lora_request",
    "protect_resident_source_tensors",
]
