"""Optional vLLM backend."""

from .backend import (
    VllmCandidateFusion,
    VllmCausalLMBackend,
    VllmFusionAdapterFactory,
)
from .materialization import (
    MaterializedTensorLoRA,
    VllmResidentAdapterPool,
    VllmTensorAdapterMaterializer,
)
from .transport import VllmTensorLoraTransport

__all__ = [
    "MaterializedTensorLoRA",
    "VllmResidentAdapterPool",
    "VllmCandidateFusion",
    "VllmCausalLMBackend",
    "VllmFusionAdapterFactory",
    "VllmTensorAdapterMaterializer",
    "VllmTensorLoraTransport",
]
