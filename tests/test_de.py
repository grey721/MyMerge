from dataclasses import replace
import random

import pytest

from agentmerge.core.config import (
    ConfigError,
    ObjectiveConfig,
    apply_experiment_overrides,
    load_experiment_config,
)
from agentmerge.evaluation.benchmarks import SyntheticMultiObjectiveEvaluator
from agentmerge.optimizers.base import NullSearchObserver
from agentmerge.optimizers.de import DEConfig, DecompositionDEOptimizer
from agentmerge.optimizers.de.decomposition import (
    TchebycheffDecomposition,
    reference_neighborhoods,
    uniform_reference_directions,
)
from agentmerge.optimizers.de.survival import DecompositionUpdate
from agentmerge.optimizers.de.variation import DifferentialVariation
from agentmerge.optimizers.fitness import FitnessSpace
from agentmerge.optimizers.genome import ScopedFusionGenome
from agentmerge.optimizers.representation import FusionSearchSpace
from agentmerge.optimizers.translation import create_plan_translator
from agentmerge.optimizers.two_stage_de import TwoStageDEOptimizer
from agentmerge.runtime.experiment import ComponentFactory


def test_reference_directions_and_neighborhoods_have_requested_size():
    directions = uniform_reference_directions(12, 3)
    neighborhoods = reference_neighborhoods(directions, 3)

    assert len(directions) == 12
    assert all(sum(direction) == pytest.approx(1.0) for direction in directions)
    assert all(len(neighborhood) == 3 for neighborhood in neighborhoods)
    assert all(index in neighborhoods[index] for index in range(12))


def test_de_variation_respects_search_space_repair():
    space = FusionSearchSpace(("agent",), ("a", "b", "c"), 0, 3)
    variation = DifferentialVariation(space, 0.5, 1.0, 0.0, 20.0)
    target = space.seed_from_spec({"agent": {"a": 1.0}})
    left = space.seed_from_spec({"agent": {"b": 1.0}})
    right = space.seed_from_spec({"agent": {"c": 1.0}})

    trial = variation.create(target, left, right, random.Random(7))
    values = space.encode(trial)

    assert all(0.0 <= value <= 1.0 for value in values)
    assert sum(values) == pytest.approx(1.0)
    assert trial.fingerprint() != target.fingerprint()


def test_single_stage_de_runs_through_generic_ask_tell_contract():
    agents = ("planner", "critic")
    adapters = ("a", "b", "c")
    space = FusionSearchSpace(agents, adapters, 0, 3)
    fitness_space = FitnessSpace(
        (
            ObjectiveConfig("quality", "maximize"),
            ObjectiveConfig("active_adapters", "minimize"),
        )
    )
    optimizer = DecompositionDEOptimizer(
        evaluator=SyntheticMultiObjectiveEvaluator(agents, adapters, seed=3),
        fitness_space=fitness_space,
        search_space=space,
        translator=create_plan_translator(space),
        variation=DifferentialVariation(space, 0.5, 1.0, None, 20.0),
        update_strategy=DecompositionUpdate(TchebycheffDecomposition(), 2),
        population_size=6,
        generations=3,
        neighborhood_size=3,
        neighborhood_mating_probability=0.9,
        seed=3,
    )

    result = optimizer.optimize()

    assert result.generations_completed == 3
    assert len(result.population) == 6
    assert result.pareto_front
    assert 6 <= result.unique_evaluations <= 18


def test_de_config_owns_decomposition_options_and_unlimited_pool_default():
    config = load_experiment_config("experiments/smoke.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="de",
        search_granularity=("adapter",),
        differential_weight=0.7,
        neighborhood_size=4,
        decomposition="pbi",
        pbi_theta=4.0,
    )
    settings = DEConfig.from_experiment_config(config)

    assert settings.differential_weight == pytest.approx(0.7)
    assert settings.neighborhood_size == 4
    assert settings.decomposition == "pbi"
    assert settings.pbi_theta == pytest.approx(4.0)
    assert settings.max_active_adapters == len(config.adapters)

    invalid = replace(
        config,
        search=replace(config.search, options={"de": {"neighborhood_size": 99}}),
    )
    with pytest.raises(ConfigError, match="neighborhood_size"):
        DEConfig.from_experiment_config(invalid)


def test_two_stage_de_projects_adapter_anchors_into_layer_space():
    config = load_experiment_config("experiments/smoke.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="two_stage_de",
        search_granularity=("adapter", "layer"),
        populations=(6, 6),
        generations=(2, 2),
        num_layers=2,
        neighborhood_size=3,
    )
    evaluator = SyntheticMultiObjectiveEvaluator(
        config.agent_names, config.adapter_names, seed=3
    )
    optimizer = ComponentFactory().create_optimizer(
        config, evaluator, NullSearchObserver()
    )

    assert isinstance(optimizer, TwoStageDEOptimizer)
    result = optimizer.optimize()

    assert result.generations_completed == 4
    assert result.pareto_front
    assert all(
        isinstance(individual.genome, ScopedFusionGenome)
        for individual in result.population
    )


def test_two_stage_de_requires_adapter_to_layer_transition():
    config = load_experiment_config("experiments/smoke.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="two_stage_de",
        search_granularity=("global", "layer"),
        populations=(6, 6),
        generations=(1, 1),
        num_layers=2,
    )
    with pytest.raises(ConfigError, match="adapter granularity"):
        ComponentFactory().validate_search(config)
