"""Adapter-to-layer two-stage decomposition-based DE package."""

from __future__ import annotations

from ...core.config import ConfigError, ExperimentConfig
from ..base import Optimizer
from ..de import build_de_components
from ..fitness import FitnessSpace
from ..registry import OptimizerBuildContext
from .optimizer import TwoStageDEOptimizer


def build_two_stage_de_optimizer(
    config: ExperimentConfig,
    context: OptimizerBuildContext,
) -> Optimizer:
    return TwoStageDEOptimizer(
        evaluator=context.evaluator,
        fitness_space=FitnessSpace(config.objectives),
        stages=_build_stages(config),
        observer=context.observer,
    )


def validate_two_stage_de_experiment(config: ExperimentConfig):
    stages = _build_stages(config)
    return {
        "algorithm": "two_stage_de",
        "stages": [
            {
                "index": index,
                "granularity": stage.name,
                "population_size": stage.settings.population_size,
                "generations": stage.settings.generations,
                "dimension": stage.search_space.dimension,
                "translator": type(stage.translator).__name__,
                "initializer": (
                    "AdmissionAwareInitializer"
                    if index == 0
                    else "ParetoProjector"
                ),
                "initial_genome_count": len(stage.initial_genomes),
                "active_adapter_range": [
                    stage.settings.min_active_adapters,
                    stage.settings.max_active_adapters,
                ],
                "multi_objective_strategy": "decomposition",
                "decomposition": stage.settings.decomposition,
                "de_variant": "DE/current/1/bin",
                "neighborhood_size": stage.settings.neighborhood_size,
                "max_replacements": stage.settings.max_replacements,
            }
            for index, stage in enumerate(stages)
        ],
    }


def _build_stages(config: ExperimentConfig):
    granularities = config.search.granularities
    if len(granularities) != 2:
        raise ConfigError(
            "two_stage_de requires exactly two --search_granularity values"
        )
    if granularities[0] != "adapter":
        raise ConfigError("Two-stage DE must start with adapter granularity")
    if granularities[1] != "layer":
        raise ConfigError("Two-stage DE must finish with layer granularity")
    return tuple(
        build_de_components(
            config,
            stage_index,
            option_namespace="two_stage_de",
            include_seed_genomes=(stage_index == 0),
        )
        for stage_index in range(2)
    )


__all__ = [
    "TwoStageDEOptimizer",
    "build_two_stage_de_optimizer",
    "validate_two_stage_de_experiment",
]
