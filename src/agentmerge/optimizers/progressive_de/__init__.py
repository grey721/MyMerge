"""Progressive recursive layer-refinement MOEA/D-DE package."""

from __future__ import annotations

from ...core.config import ExperimentConfig
from ..base import Optimizer
from ..fitness import FitnessSpace
from ..registry import OptimizerBuildContext
from .components import build_progressive_components
from .config import ProgressiveDEConfig
from .optimizer import ProgressiveDEOptimizer
from .partition import RecursiveLayerPartition, build_refinement_partitions


def build_progressive_de_optimizer(
    config: ExperimentConfig,
    context: OptimizerBuildContext,
) -> Optimizer:
    settings = ProgressiveDEConfig.from_experiment_config(config)
    return ProgressiveDEOptimizer(
        evaluator=context.evaluator,
        fitness_space=FitnessSpace(config.objectives),
        stages=build_progressive_components(config, settings),
        settings=settings,
        observer=context.observer,
    )


def validate_progressive_de_experiment(config: ExperimentConfig):
    settings = ProgressiveDEConfig.from_experiment_config(config)
    stages = build_progressive_components(config, settings)
    return {
        "algorithm": "progressive_de",
        "adapter_pool_mode": "fixed_global_pool",
        "adapter_count": len(config.adapters),
        "transition": "full_population_chromosome_state_transfer",
        "stages": [
            {
                "index": index,
                "granularity": stage.name,
                "group_count": settings.refinement_groups[index],
                "population_size": stage.settings.population_size,
                "generations": stage.settings.generations,
                "dimension": stage.search_space.dimension,
                "translator": type(stage.translator).__name__,
                "initializer": (
                    "AdmissionAwareInitializer"
                    if index == 0
                    else "InheritedPopulationState"
                ),
                "multi_objective_strategy": "decomposition",
                "decomposition": stage.settings.decomposition,
                "de_variant": "DE/current/1/bin",
            }
            for index, stage in enumerate(stages)
        ],
    }


__all__ = [
    "ProgressiveDEConfig",
    "ProgressiveDEOptimizer",
    "RecursiveLayerPartition",
    "build_progressive_de_optimizer",
    "build_refinement_partitions",
    "validate_progressive_de_experiment",
]
