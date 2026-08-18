"""Single-stage NSGA-II optimizer package."""

from __future__ import annotations

from ...core.config import ConfigError, ExperimentConfig
from ..base import Optimizer
from ..pareto import (
    assign_crowding_distance,
    dominates,
    fast_non_dominated_sort,
)
from ..registry import OptimizerBuildContext
from .components import NSGA2Components
from .config import NSGA2Config
from ..fitness import FitnessSpace
from ..initialization import create_initial_genomes
from .optimizer import NSGA2Optimizer
from ..representation import create_default_search_space_registry
from ..translation import create_plan_translator
from .variation import NSGA2Variation


def build_nsga2_components(
    config: ExperimentConfig,
    stage_index: int = 0,
    *,
    option_namespace: str = "nsga2",
    include_seed_genomes: bool = True,
) -> NSGA2Components:
    """Assemble one stage without exposing NSGA-II private internals."""

    settings = NSGA2Config.from_experiment_config(
        config, stage_index, option_namespace=option_namespace
    )
    search_space = create_default_search_space_registry().create(
        settings.representation,
        config,
        stage_index,
        settings=settings,
    )
    translator = create_plan_translator(search_space, config)
    variation = NSGA2Variation(
        search_space=search_space,
        crossover_probability=settings.crossover_probability,
        mutation_probability=settings.mutation_probability,
        sparsity_mutation_probability=settings.sparsity_mutation_probability,
        mutation_sigma=settings.mutation_sigma,
    )
    return NSGA2Components(
        name=settings.representation,
        settings=settings,
        search_space=search_space,
        translator=translator,
        variation=variation,
        initial_genomes=(
            tuple(create_initial_genomes(config, search_space))
            if include_seed_genomes
            else ()
        ),
    )


def build_nsga2_optimizer(
    config: ExperimentConfig,
    context: OptimizerBuildContext,
) -> Optimizer:
    """Build a single-stage NSGA-II optimizer."""

    if config.search.stage_count != 1:
        raise ConfigError(
            "nsga2 accepts exactly one --search_granularity; use "
            "--algorithm two_stage_nsga2 for global-to-layer search"
        )
    stage = build_nsga2_components(config)
    return NSGA2Optimizer(
        evaluator=context.evaluator,
        fitness_space=FitnessSpace(config.objectives),
        search_space=stage.search_space,
        translator=stage.translator,
        variation=stage.variation,
        population_size=stage.settings.population_size,
        generations=stage.settings.generations,
        seed=stage.settings.seed,
        workers=stage.settings.workers,
        observer=context.observer,
        initial_genomes=stage.initial_genomes,
    )


def validate_nsga2_experiment(config: ExperimentConfig):
    if config.search.stage_count != 1:
        raise ConfigError(
            "nsga2 accepts exactly one --search_granularity"
        )
    stage = build_nsga2_components(config)
    return {
        "algorithm": "nsga2",
        "stages": [_stage_description(stage, 0)],
    }


def _stage_description(stage: NSGA2Components, index: int):
    return {
        "index": index,
        "granularity": stage.name,
        "population_size": stage.settings.population_size,
        "generations": stage.settings.generations,
        "dimension": stage.search_space.dimension,
        "translator": type(stage.translator).__name__,
        "initializer": "AdmissionAwareInitializer",
        "initial_genome_count": len(stage.initial_genomes),
        "active_adapter_range": [
            stage.settings.min_active_adapters,
            stage.settings.max_active_adapters,
        ],
    }


__all__ = [
    "NSGA2Components",
    "NSGA2Config",
    "NSGA2Optimizer",
    "NSGA2Variation",
    "assign_crowding_distance",
    "build_nsga2_components",
    "build_nsga2_optimizer",
    "create_plan_translator",
    "create_initial_genomes",
    "dominates",
    "fast_non_dominated_sort",
]
