"""Backend-neutral LoRA fusion algebra and compiled execution plans.

The algebra decides *what* fused delta a policy represents.  PEFT and vLLM
remain parallel tensor runtimes which only execute the immutable plan emitted
here; neither backend owns the fusion mathematics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Mapping, Sequence, Tuple, Type

from ..core.config import ModelConfig
from ..core.policies import (
    FusionPolicy,
    ModuleTarget,
    active_adapter_names,
)


@dataclass(frozen=True)
class SourceAdapterDescriptor:
    """Backend-independent metadata needed by a fusion algebra."""

    name: str
    rank: int
    native_scale: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Source adapter name cannot be empty")
        if self.rank <= 0:
            raise ValueError("Source adapter rank must be positive")


@dataclass(frozen=True)
class FusionModuleSpec:
    """Stable identity of one adapter-bearing module."""

    key: str
    target: ModuleTarget


@dataclass(frozen=True)
class FactorSlice:
    """One source adapter's rank interval in a fused LoRA."""

    source_name: str
    rank: int
    start: int
    end: int


class FactorComposition(str, Enum):
    """How source factors sharing a target rank interval are combined."""

    CONCATENATE = "concatenate"
    LINEAR_SUM = "linear_sum"


@dataclass(frozen=True)
class ModuleFusionInstruction:
    """Per-module coefficients aligned with ``FusionExecutionPlan.slices``."""

    module_key: str
    a_coefficients: Tuple[float, ...]
    b_coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class FusionExecutionPlan:
    """Immutable algebra result consumed by a tensor backend."""

    algebra: str
    policy_fingerprint: str
    factor_composition: FactorComposition
    slices: Tuple[FactorSlice, ...]
    modules: Tuple[ModuleFusionInstruction, ...]
    total_rank: int

    def __post_init__(self) -> None:
        try:
            composition = FactorComposition(self.factor_composition)
        except ValueError as exc:
            raise ValueError(
                "Unknown factor composition {!r}".format(self.factor_composition)
            ) from exc
        object.__setattr__(self, "factor_composition", composition)
        if self.total_rank <= 0:
            raise ValueError("Fusion execution plan rank must be positive")
        source_count = len(self.slices)
        for factor_slice in self.slices:
            if (
                factor_slice.rank <= 0
                or factor_slice.start < 0
                or factor_slice.end - factor_slice.start != factor_slice.rank
                or factor_slice.end > self.total_rank
            ):
                raise ValueError(
                    "Invalid factor slice for adapter {!r}".format(
                        factor_slice.source_name
                    )
                )
        for instruction in self.modules:
            if (
                len(instruction.a_coefficients) != source_count
                or len(instruction.b_coefficients) != source_count
            ):
                raise ValueError(
                    "Module coefficient count must match the source slice count"
                )

    @property
    def active_adapter_names(self) -> Tuple[str, ...]:
        return tuple(item.source_name for item in self.slices)


class FusionAlgebra(ABC):
    """Translate a fusion policy into backend-neutral tensor instructions."""

    name: str

    @abstractmethod
    def compile(
        self,
        policy: FusionPolicy,
        sources: Sequence[SourceAdapterDescriptor],
        modules: Sequence[FusionModuleSpec] = (),
    ) -> FusionExecutionPlan:
        """Compile one policy without allocating backend tensors."""

    def estimate_rank(
        self,
        policy: FusionPolicy,
        sources: Sequence[SourceAdapterDescriptor],
    ) -> int:
        if not active_adapter_names(policy):
            return 0
        return self.compile(policy, sources).total_rank


class ExactDeltaFusionAlgebra(FusionAlgebra):
    """Represent exactly ``sum_i w_i * scale_i * B_i @ A_i``.

    Exactness is obtained by assigning every active source its own rank slice:
    ``A_fused = cat(A_i, dim=0)`` and
    ``B_fused = cat(w_i * scale_i * B_i, dim=1)``.  The temporary LoRA uses
    scale one, so expanding the product recovers the weighted delta exactly.
    """

    name = "exact_delta"

    def compile(
        self,
        policy: FusionPolicy,
        sources: Sequence[SourceAdapterDescriptor],
        modules: Sequence[FusionModuleSpec] = (),
    ) -> FusionExecutionPlan:
        source_by_name = _source_map(sources)
        active_names = active_adapter_names(policy)
        if not active_names:
            raise ValueError("Cannot compile an empty adapter policy")

        unknown = tuple(name for name in active_names if name not in source_by_name)
        if unknown:
            raise KeyError(
                "Fusion policy references unknown source adapter(s): {}".format(
                    ", ".join(unknown)
                )
            )

        slices = []
        offset = 0
        for name in active_names:
            source = source_by_name[name]
            end = offset + source.rank
            slices.append(FactorSlice(name, source.rank, offset, end))
            offset = end

        instructions = []
        seen_module_keys = set()
        for module in modules:
            if module.key in seen_module_keys:
                raise ValueError(
                    "Duplicate fusion module key {!r}".format(module.key)
                )
            seen_module_keys.add(module.key)
            weights = policy.mixture_for(module.target).as_dict()
            instructions.append(
                ModuleFusionInstruction(
                    module_key=module.key,
                    a_coefficients=tuple(1.0 for _ in slices),
                    b_coefficients=tuple(
                        float(weights.get(item.source_name, 0.0))
                        * source_by_name[item.source_name].native_scale
                        for item in slices
                    ),
                )
            )

        return FusionExecutionPlan(
            algebra=self.name,
            policy_fingerprint=policy.fingerprint(),
            factor_composition=FactorComposition.CONCATENATE,
            slices=tuple(slices),
            modules=tuple(instructions),
            total_rank=offset,
        )


class FactorLinearFusionAlgebra(FusionAlgebra):
    """Blend LoRA factors linearly, following LoRAHub/PEM semantics.

    For each module this algebra constructs

    ``A' = sum_i w_i * sqrt(scale_i) * pad(A_i)`` and
    ``B' = sum_i w_i * sqrt(scale_i) * pad(B_i)``.

    The temporary LoRA has scale one.  When source scales are equal, this is
    exactly the usual LoRAHub/PEM parameter-wise weighted sum followed by the
    source LoRA scale.  Splitting heterogeneous native scales symmetrically
    between A and B extends the same convention without privileging a factor.

    Unlike :class:`ExactDeltaFusionAlgebra`, multiplying ``B' @ A'`` includes
    cross-adapter terms.  This is intentional: factor-linear composition is a
    different fusion algebra, not an approximation of exact delta summation.
    Different source ranks are zero-padded to the largest active rank.
    """

    name = "factor_linear"

    def compile(
        self,
        policy: FusionPolicy,
        sources: Sequence[SourceAdapterDescriptor],
        modules: Sequence[FusionModuleSpec] = (),
    ) -> FusionExecutionPlan:
        source_by_name = _source_map(sources)
        active_names = active_adapter_names(policy)
        if not active_names:
            raise ValueError("Cannot compile an empty adapter policy")

        unknown = tuple(name for name in active_names if name not in source_by_name)
        if unknown:
            raise KeyError(
                "Fusion policy references unknown source adapter(s): {}".format(
                    ", ".join(unknown)
                )
            )

        active_sources = tuple(source_by_name[name] for name in active_names)
        for source in active_sources:
            if not math.isfinite(source.native_scale) or source.native_scale < 0.0:
                raise ValueError(
                    "Factor-linear fusion requires a finite non-negative native "
                    "scale for adapter {!r}".format(source.name)
                )

        total_rank = max(source.rank for source in active_sources)
        # Slices deliberately overlap: every source contributes to the same
        # leading rank coordinates and the unused tail is implicit zero-padding.
        slices = tuple(
            FactorSlice(source.name, source.rank, 0, source.rank)
            for source in active_sources
        )

        instructions = []
        seen_module_keys = set()
        for module in modules:
            if module.key in seen_module_keys:
                raise ValueError(
                    "Duplicate fusion module key {!r}".format(module.key)
                )
            seen_module_keys.add(module.key)
            weights = policy.mixture_for(module.target).as_dict()
            coefficients = tuple(
                float(weights.get(source.name, 0.0))
                * math.sqrt(source.native_scale)
                for source in active_sources
            )
            instructions.append(
                ModuleFusionInstruction(
                    module_key=module.key,
                    a_coefficients=coefficients,
                    b_coefficients=coefficients,
                )
            )

        return FusionExecutionPlan(
            algebra=self.name,
            policy_fingerprint=policy.fingerprint(),
            factor_composition=FactorComposition.LINEAR_SUM,
            slices=slices,
            modules=tuple(instructions),
            total_rank=total_rank,
        )


_ALGEBRA_FACTORIES = {
    ExactDeltaFusionAlgebra.name: ExactDeltaFusionAlgebra,
    FactorLinearFusionAlgebra.name: FactorLinearFusionAlgebra,
}  # type: Dict[str, Type[FusionAlgebra]]


def register_fusion_algebra(
    name: str,
    implementation: Type[FusionAlgebra],
) -> None:
    """Register an algebra without coupling it to either model backend."""

    normalized = _normalize_name(name)
    if not issubclass(implementation, FusionAlgebra):
        raise TypeError("Fusion algebra implementation must subclass FusionAlgebra")
    _ALGEBRA_FACTORIES[normalized] = implementation


def create_fusion_algebra(name: str = "exact_delta") -> FusionAlgebra:
    normalized = _normalize_name(name)
    try:
        implementation = _ALGEBRA_FACTORIES[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unknown fusion algebra {!r}; available: {}".format(
                name, ", ".join(sorted(_ALGEBRA_FACTORIES))
            )
        ) from exc
    return implementation()


def fusion_algebra_from_model_config(model_config: ModelConfig) -> FusionAlgebra:
    return create_fusion_algebra(
        str(model_config.options.get("fusion_algebra", "exact_delta"))
    )


def _source_map(
    sources: Sequence[SourceAdapterDescriptor],
) -> Mapping[str, SourceAdapterDescriptor]:
    result = {}  # type: Dict[str, SourceAdapterDescriptor]
    for source in sources:
        if source.name in result:
            raise ValueError(
                "Duplicate source adapter descriptor {!r}".format(source.name)
            )
        result[source.name] = source
    return result


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Fusion algebra name cannot be empty")
    return normalized


__all__ = [
    "ExactDeltaFusionAlgebra",
    "FactorComposition",
    "FactorLinearFusionAlgebra",
    "FactorSlice",
    "FusionAlgebra",
    "FusionExecutionPlan",
    "FusionModuleSpec",
    "ModuleFusionInstruction",
    "SourceAdapterDescriptor",
    "create_fusion_algebra",
    "fusion_algebra_from_model_config",
    "register_fusion_algebra",
]
