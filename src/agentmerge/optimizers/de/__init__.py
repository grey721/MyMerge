"""Single-stage decomposition-based multi-objective DE package."""

from __future__ import annotations

from ...core.config import ConfigError, ExperimentConfig
from ..base import Optimizer
from ..registry import OptimizerBuildContext
from ..fitness import FitnessSpace
from ..initialization import create_initial_genomes
from ..representation import create_default_search_space_registry
from ..translation import create_plan_translator
from .components import DEComponents
from .config import DEConfig
from .decomposition import create_decomposition
from .optimizer import DecompositionDEOptimizer
from .survival import DecompositionUpdate
from .variation import DifferentialVariation


def build_de_components(
    config: ExperimentConfig,
    stage_index: int = 0,
    *,
    option_namespace: str = "de",
    include_seed_genomes: bool = True,
) -> DEComponents:
    settings = DEConfig.from_experiment_config(
        config, stage_index, option_namespace=option_namespace
    )
    search_space = create_default_search_space_registry().create(
        settings.representation,
        config,
        stage_index,
        settings=settings,
    )
    translator = create_plan_translator(search_space, config)
    variation = DifferentialVariation(
        search_space=search_space,
        differential_weight=settings.differential_weight,
        crossover_probability=settings.crossover_probability,
        mutation_probability=settings.mutation_probability,
        mutation_distribution_index=settings.mutation_distribution_index,
    )
    update_strategy = DecompositionUpdate(
        create_decomposition(
            settings.decomposition, pbi_theta=settings.pbi_theta
        ),
        max_replacements=settings.max_replacements,
    )
    return DEComponents(
        name=settings.representation,
        settings=settings,
        search_space=search_space,
        translator=translator,
        variation=variation,
        update_strategy=update_strategy,
        initial_genomes=(
            tuple(create_initial_genomes(config, search_space))
            if include_seed_genomes
            else ()
        ),
    )


def create_stage_optimizer(
    config: ExperimentConfig,
    context: OptimizerBuildContext,
    stage: DEComponents,
) -> DecompositionDEOptimizer:
    return DecompositionDEOptimizer(
        evaluator=context.evaluator,
        fitness_space=FitnessSpace(config.objectives),
        search_space=stage.search_space,
        translator=stage.translator,
        variation=stage.variation,
        update_strategy=stage.update_strategy,
        population_size=stage.settings.population_size,
        generations=stage.settings.generations,
        neighborhood_size=stage.settings.neighborhood_size,
        neighborhood_mating_probability=(
            stage.settings.neighborhood_mating_probability
        ),
        seed=stage.settings.seed,
        workers=stage.settings.workers,
        observer=context.observer,
        initial_genomes=stage.initial_genomes,
    )


def build_de_optimizer(
    config: ExperimentConfig,
    context: OptimizerBuildContext,
) -> Optimizer:
    if config.search.stage_count != 1:
        raise ConfigError(
            "de accepts exactly one --search_granularity; use "
            "--algorithm two_stage_de for adapter-to-layer search"
        )
    return create_stage_optimizer(config, context, build_de_components(config))


def validate_de_experiment(config: ExperimentConfig):
    if config.search.stage_count != 1:
        raise ConfigError("de accepts exactly one --search_granularity")
    stage = build_de_components(config)
    return {"algorithm": "de", "stages": [_stage_description(stage, 0)]}


def _stage_description(stage: DEComponents, index: int):
    settings = stage.settings
    return {
        "index": index,
        "granularity": stage.name,
        "population_size": settings.population_size,
        "generations": settings.generations,
        "dimension": stage.search_space.dimension,
        "translator": type(stage.translator).__name__,
        "initializer": "AdmissionAwareInitializer",
        "initial_genome_count": len(stage.initial_genomes),
        "active_adapter_range": [
            settings.min_active_adapters,
            settings.max_active_adapters,
        ],
        "multi_objective_strategy": "decomposition",
        "decomposition": settings.decomposition,
        "de_variant": "DE/current/1/bin",
        "neighborhood_size": settings.neighborhood_size,
        "max_replacements": settings.max_replacements,
    }


__all__ = [
    "DEComponents",
    "DEConfig",
    "DecompositionDEOptimizer",
    "DifferentialVariation",
    "build_de_components",
    "build_de_optimizer",
    "create_stage_optimizer",
    "validate_de_experiment",
]
