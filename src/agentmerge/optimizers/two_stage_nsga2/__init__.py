"""Global-to-scoped Two-stage NSGA-II optimizer package."""

from __future__ import annotations

from ...core.config import ConfigError, ExperimentConfig
from ..base import Optimizer
from ..nsga2 import build_nsga2_components
from ..fitness import FitnessSpace
from ..registry import OptimizerBuildContext
from .optimizer import TwoStageNSGA2Optimizer


_SCOPED_GRANULARITIES = {
    "layer",
    "layer_group",
    "module_type",
    "layer_module",
}


def build_two_stage_nsga2_optimizer(
    config: ExperimentConfig,
    context: OptimizerBuildContext,
) -> Optimizer:
    stages = _build_stages(config)
    return TwoStageNSGA2Optimizer(
        evaluator=context.evaluator,
        fitness_space=FitnessSpace(config.objectives),
        stages=stages,
        observer=context.observer,
    )


def validate_two_stage_nsga2_experiment(config: ExperimentConfig):
    stages = _build_stages(config)
    return {
        "algorithm": "two_stage_nsga2",
        "stages": [
            {
                "index": index,
                "granularity": stage.name,
                "population_size": stage.settings.population_size,
                "generations": stage.settings.generations,
                "dimension": stage.search_space.dimension,
                "translator": type(stage.translator).__name__,
                "initializer": (
                    "AdmissionAwareInitializer" if index == 0 else "ParetoProjector"
                ),
                "initial_genome_count": len(stage.initial_genomes),
                "active_adapter_range": [
                    stage.settings.min_active_adapters,
                    stage.settings.max_active_adapters,
                ],
            }
            for index, stage in enumerate(stages)
        ],
    }


def _build_stages(config: ExperimentConfig):
    granularities = config.search.granularities
    if len(granularities) != 2:
        raise ConfigError(
            "two_stage_nsga2 requires exactly two --search_granularity values"
        )
    if granularities[0] not in ("global", "adapter"):
        raise ConfigError("Two-stage NSGA-II must start with global fusion")
    if granularities[1] not in _SCOPED_GRANULARITIES:
        raise ConfigError(
            "Two-stage NSGA-II must finish with a scoped granularity"
        )
    return tuple(
        build_nsga2_components(
            config,
            stage_index,
            option_namespace="two_stage_nsga2",
            include_seed_genomes=(stage_index == 0),
        )
        for stage_index in range(2)
    )


__all__ = [
    "TwoStageNSGA2Optimizer",
    "build_two_stage_nsga2_optimizer",
    "validate_two_stage_nsga2_experiment",
]
