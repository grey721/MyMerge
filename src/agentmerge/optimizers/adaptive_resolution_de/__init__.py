"""Adaptive budget-aware nested-resolution MOEA/D-DE package."""

from __future__ import annotations

from ...core.config import ExperimentConfig
from ..base import Optimizer
from ..fitness import FitnessSpace
from ..registry import OptimizerBuildContext
from .components import build_adaptive_resolution_components
from .config import AdaptiveResolutionConfig
from .controller import BudgetAwareResolutionController, MonitorStats
from .hierarchy import (
    LayerGroup,
    ResolutionManager,
    build_nested_layer_groups,
)
from .monitor import DirectionalMonitor
from .optimizer import AdaptiveResolutionDEOptimizer


def _resolution(
    config: ExperimentConfig,
    settings: AdaptiveResolutionConfig,
) -> ResolutionManager:
    return ResolutionManager(
        settings.num_layers,
        {
            agent_name: len(config.adapters)
            for agent_name in config.agent_names
        },
        first_split=settings.first_split,
    )


def build_adaptive_resolution_de_optimizer(
    config: ExperimentConfig,
    context: OptimizerBuildContext,
) -> Optimizer:
    settings = AdaptiveResolutionConfig.from_experiment_config(config)
    resolution = _resolution(config, settings)
    return AdaptiveResolutionDEOptimizer(
        evaluator=context.evaluator,
        fitness_space=FitnessSpace(config.objectives),
        stages=build_adaptive_resolution_components(
            config, settings, resolution
        ),
        settings=settings,
        resolution=resolution,
        observer=context.observer,
    )


def validate_adaptive_resolution_de_experiment(config: ExperimentConfig):
    settings = AdaptiveResolutionConfig.from_experiment_config(config)
    resolution = _resolution(config, settings)
    stages = build_adaptive_resolution_components(config, settings, resolution)
    return {
        "algorithm": "adaptive_resolution_de",
        "adapter_pool_mode": "fixed_global_pool",
        "adapter_count": len(config.adapters),
        "transition": "lossless_population_and_archive_state_transfer",
        "total_budget": settings.total_budget,
        "min_stage_fe": list(settings.min_stage_fe),
        "promotion_triggers": [
            "search_saturation",
            "zero_progress_fallback",
            "budget_reserve",
        ],
        "stages": [
            {
                "index": index,
                "granularity": stage.name,
                "group_count": len(resolution.stages[index]),
                "layer_intervals": [
                    list(group.interval) for group in resolution.stages[index]
                ],
                "population_size": stage.settings.population_size,
                "generations": "adaptive_real_fe_budget",
                "min_stage_fe": settings.min_stage_fe[index],
                "dimension": stage.search_space.dimension,
                "translator": type(stage.translator).__name__,
                "initializer": (
                    "AdmissionAwareInitializer"
                    if index == 0
                    else "LosslessInheritedPopulationAndArchive"
                ),
                "multi_objective_strategy": "decomposition",
                "decomposition": stage.settings.decomposition,
                "de_variant": "DE/current/1/bin",
            }
            for index, stage in enumerate(stages)
        ],
    }


__all__ = [
    "AdaptiveResolutionConfig",
    "AdaptiveResolutionDEOptimizer",
    "BudgetAwareResolutionController",
    "DirectionalMonitor",
    "LayerGroup",
    "MonitorStats",
    "ResolutionManager",
    "build_adaptive_resolution_de_optimizer",
    "build_nested_layer_groups",
    "validate_adaptive_resolution_de_experiment",
]
