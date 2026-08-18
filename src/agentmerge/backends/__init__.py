"""Model backend implementations."""

from .algebra import (
    ExactDeltaFusionAlgebra,
    FactorComposition,
    FactorLinearFusionAlgebra,
    FusionAlgebra,
    FusionExecutionPlan,
    create_fusion_algebra,
    register_fusion_algebra,
)
from .base import ModelBackend, OptionalDependencyError
from .factory import create_model_backend
from .materialization import (
    AdapterMaterializer,
    PeftCandidateFusion,
    PeftDirectAdapterMaterializer,
    PeftFusionAdapterFactory,
)
from .mock import MockModelBackend
from .peft import PeftCausalLMBackend
from .peft_pool import PeftResidentAdapterPool
from .pool import (
    AdapterPool,
    AdapterPoolReleaseReport,
    AdapterPoolState,
    ResidentAdapterPool,
)
from .resources import (
    AdapterResource,
    FusedAdapterResource,
    FusionAdapterFactory,
    SourceAdapterResource,
)
from .vllm import VllmCausalLMBackend

__all__ = [
    "AdapterMaterializer",
    "AdapterPool",
    "AdapterPoolReleaseReport",
    "AdapterPoolState",
    "AdapterResource",
    "ExactDeltaFusionAlgebra",
    "FactorComposition",
    "FactorLinearFusionAlgebra",
    "FusionAlgebra",
    "FusedAdapterResource",
    "FusionAdapterFactory",
    "FusionExecutionPlan",
    "MockModelBackend",
    "ModelBackend",
    "OptionalDependencyError",
    "PeftCandidateFusion",
    "PeftResidentAdapterPool",
    "PeftDirectAdapterMaterializer",
    "PeftFusionAdapterFactory",
    "PeftCausalLMBackend",
    "ResidentAdapterPool",
    "SourceAdapterResource",
    "VllmCausalLMBackend",
    "create_fusion_algebra",
    "create_model_backend",
    "register_fusion_algebra",
]
