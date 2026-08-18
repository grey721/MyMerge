"""Optimizer package registry and construction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

from ..core.config import ExperimentConfig
from ..evaluation.benchmarks import FusionPlanEvaluator
from .base import Optimizer, SearchObserver


@dataclass(frozen=True)
class OptimizerBuildContext:
    """Shared dependencies supplied to an optimizer package."""

    evaluator: FusionPlanEvaluator
    observer: SearchObserver


OptimizerBuilder = Callable[
    [ExperimentConfig, OptimizerBuildContext], Optimizer
]
OptimizerValidator = Callable[[ExperimentConfig], Mapping[str, object]]


class OptimizerRegistry:
    """Map algorithm names to independent package-level builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, OptimizerBuilder] = {}
        self._validators: Dict[str, OptimizerValidator] = {}

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._builders))

    def register(
        self,
        name: str,
        builder: OptimizerBuilder,
        *,
        validator: Optional[OptimizerValidator] = None,
        replace: bool = False,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Optimizer name cannot be empty")
        if normalized in self._builders and not replace:
            raise ValueError(
                "Optimizer {!r} is already registered".format(normalized)
            )
        self._builders[normalized] = builder
        if validator is not None:
            self._validators[normalized] = validator
        elif replace:
            self._validators.pop(normalized, None)

    def validate(
        self, name: str, config: ExperimentConfig
    ) -> Mapping[str, object]:
        normalized = name.strip().lower()
        if normalized not in self._builders:
            available = ", ".join(self.names) or "<none>"
            raise ValueError(
                "Unknown search algorithm {!r}; registered: {}".format(
                    name, available
                )
            )
        validator = self._validators.get(normalized)
        if validator is not None:
            return validator(config)
        return {}

    def create(
        self,
        name: str,
        config: ExperimentConfig,
        context: OptimizerBuildContext,
    ) -> Optimizer:
        normalized = name.strip().lower()
        try:
            builder = self._builders[normalized]
        except KeyError:
            available = ", ".join(self.names) or "<none>"
            raise ValueError(
                "Unknown search algorithm {!r}; registered: {}".format(
                    name, available
                )
            )
        return builder(config, context)


def create_default_optimizer_registry() -> OptimizerRegistry:
    """Create a fresh registry containing the built-in optimizers."""

    from .nsga2 import build_nsga2_optimizer, validate_nsga2_experiment
    from .two_stage_nsga2 import (
        build_two_stage_nsga2_optimizer,
        validate_two_stage_nsga2_experiment,
    )
    from .de import build_de_optimizer, validate_de_experiment
    from .two_stage_de import (
        build_two_stage_de_optimizer,
        validate_two_stage_de_experiment,
    )
    from .progressive_de import (
        build_progressive_de_optimizer,
        validate_progressive_de_experiment,
    )
    from .adaptive_resolution_de import (
        build_adaptive_resolution_de_optimizer,
        validate_adaptive_resolution_de_experiment,
    )

    registry = OptimizerRegistry()
    registry.register(
        "nsga2",
        build_nsga2_optimizer,
        validator=validate_nsga2_experiment,
    )
    registry.register(
        "two_stage_nsga2",
        build_two_stage_nsga2_optimizer,
        validator=validate_two_stage_nsga2_experiment,
    )
    registry.register(
        "de",
        build_de_optimizer,
        validator=validate_de_experiment,
    )
    registry.register(
        "two_stage_de",
        build_two_stage_de_optimizer,
        validator=validate_two_stage_de_experiment,
    )
    registry.register(
        "progressive_de",
        build_progressive_de_optimizer,
        validator=validate_progressive_de_experiment,
    )
    registry.register(
        "adaptive_resolution_de",
        build_adaptive_resolution_de_optimizer,
        validator=validate_adaptive_resolution_de_experiment,
    )
    return registry


__all__ = [
    "OptimizerBuildContext",
    "OptimizerBuilder",
    "OptimizerRegistry",
    "create_default_optimizer_registry",
]
