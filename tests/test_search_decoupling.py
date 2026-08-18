from dataclasses import replace
import random

import pytest

from agentmerge.core.config import (
    ObjectiveConfig,
    apply_experiment_overrides,
    load_experiment_config,
)
from agentmerge.evaluation.benchmarks import SyntheticMultiObjectiveEvaluator
from agentmerge.optimizers.representation import (
    AgentSpecificScopedFusionSearchSpace,
    FusionSearchSpace,
    ScopedFusionSearchSpace,
    create_default_search_space_registry,
)
from agentmerge.optimizers.fitness import FitnessSpace
from agentmerge.optimizers.nsga2 import (
    NSGA2Optimizer,
    NSGA2Variation,
    create_plan_translator,
)
from agentmerge.optimizers.two_stage_nsga2 import TwoStageNSGA2Optimizer
from agentmerge.optimizers.genome import ScopedFusionGenome
from agentmerge.optimizers.base import NullSearchObserver
from agentmerge.runtime.experiment import ComponentFactory
from agentmerge.core.policies import (
    LayerGroupGranularity,
    ModuleTarget,
    ScopedAdapterFusionPolicy,
)


def test_scoped_space_round_trips_as_bounded_real_vector():
    space = ScopedFusionSearchSpace(
        agent_names=("planner",),
        adapter_names=("math", "code"),
        granularity=LayerGroupGranularity(num_layers=4, group_size=2),
        min_active=1,
        max_active=2,
    )
    genome = space.seed_from_spec(
        {
            "planner": {
                "group:0": {"math": 1.0},
                "group:1": {"code": 1.0},
            }
        }
    )

    assert space.dimension == 4
    assert space.decode(space.encode(genome)).to_dict() == genome.to_dict()
    policy = create_plan_translator(space).plan_for(genome).policy_for(
        "planner"
    )
    assert policy.mixture_for(ModuleTarget(layer_index=0)).as_dict() == {
        "math": 1.0
    }
    assert policy.mixture_for(ModuleTarget(layer_index=3)).as_dict() == {
        "code": 1.0
    }


def test_agent_specific_scoped_space_uses_ragged_adaptive_coordinates():
    template = ScopedFusionSearchSpace(
        agent_names=("planner", "critic"),
        adapter_names=("a", "b", "c", "d"),
        granularity=LayerGroupGranularity(num_layers=4, group_size=2),
        min_active=0,
        max_active=4,
    )
    # Deliberately non-prefix and unequal pools. The compressed layout only
    # consumes masks; it neither assumes TopK nor a common K across Agents.
    space = AgentSpecificScopedFusionSearchSpace.from_masks(
        template,
        {
            "planner": (True, False, True, False),
            "critic": (False, True, True, True),
        },
    )
    genome = space.broadcast_agent_weights(
        {
            "planner": (0.25, 0.0, 0.75, 0.0),
            "critic": (0.0, 0.2, 0.3, 0.5),
        }
    )

    assert space.selected_adapter_indices("planner") == (0, 2)
    assert space.selected_adapter_indices("critic") == (1, 2, 3)
    assert space.dimension == 2 * (2 + 3) == 10
    assert space.coefficient_groups == ((0, 2), (2, 4), (4, 7), (7, 10))
    encoded = space.encode(genome)
    assert len(encoded) == 10
    assert space.decode(encoded).to_dict() == genome.to_dict()

    plan = create_plan_translator(space).plan_for(genome)
    planner = plan.policy_for("planner").mixture_for(
        ModuleTarget(layer_index=0)
    )
    critic = plan.policy_for("critic").mixture_for(
        ModuleTarget(layer_index=3)
    )
    assert planner.as_dict() == {"a": 0.25, "c": 0.75}
    assert critic.as_dict() == {"b": 0.2, "c": 0.3, "d": 0.5}


def test_agent_specific_scoped_space_rejects_mass_outside_dynamic_pool():
    template = ScopedFusionSearchSpace(
        agent_names=("agent",),
        adapter_names=("a", "b", "c"),
        granularity=LayerGroupGranularity(num_layers=2, group_size=1),
        min_active=0,
        max_active=3,
    )
    space = AgentSpecificScopedFusionSearchSpace.from_masks(
        template, {"agent": (False, True, True)}
    )
    invalid = ScopedFusionGenome(
        {
            "agent": {
                "group:0": [0.5, 0.5, 0.0],
                "group:1": [0.0, 0.5, 0.5],
            }
        }
    )

    with pytest.raises(ValueError, match="outside"):
        space.encode(invalid)
    repaired = space.repair(invalid)
    assert repaired.agent_scope_weights["agent"]["group:0"] == [0.0, 1.0, 0.0]


def test_nsga2_uses_generic_scoped_search_contract():
    agents = ("planner", "critic")
    adapters = ("a", "b")
    space = ScopedFusionSearchSpace(
        agents,
        adapters,
        LayerGroupGranularity(num_layers=4, group_size=2),
        min_active=1,
        max_active=2,
    )
    optimizer = NSGA2Optimizer(
        evaluator=SyntheticMultiObjectiveEvaluator(agents, adapters, seed=3),
        fitness_space=FitnessSpace(
            (
                ObjectiveConfig("quality", "maximize"),
                ObjectiveConfig("active_adapters", "minimize"),
            )
        ),
        search_space=space,
        translator=create_plan_translator(space),
        variation=NSGA2Variation(space, 0.9, 0.3, 0.2, 0.15),
        population_size=6,
        generations=1,
        seed=3,
    )

    result = optimizer.optimize()

    assert len(result.population) == 6
    assert result.pareto_front
    assert all(
        len(space.encode(individual.genome)) == space.dimension
        for individual in result.population
    )


def test_representation_registry_is_owned_by_nsga2_config():
    config = load_experiment_config("experiments/smoke.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="nsga2",
        search_granularity=("layer_group",),
        num_layers=8,
        group_size=2,
    )

    space = create_default_search_space_registry().create("layer_group", config)

    assert isinstance(space, ScopedFusionSearchSpace)
    assert space.scope_names == ("group:0", "group:1", "group:2", "group:3")
    sampled = space.sample(random.Random(4))
    assert len(space.encode(sampled)) == space.dimension


def test_layer_policy_has_stable_runtime_fingerprint():
    policy = ScopedAdapterFusionPolicy(
        adapter_names=("a", "b"),
        granularity=LayerGroupGranularity(4, 2),
        scope_weights=(
            ("group:0", (1.0, 0.0)),
            ("group:1", (0.0, 1.0)),
        ),
    )

    assert policy.fingerprint() == policy.fingerprint()
    assert len(policy.fingerprint()) == 16


def test_two_stage_nsga2_switches_search_space_and_translator():
    config = load_experiment_config("experiments/smoke.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="two_stage_nsga2",
        search_granularity=("global", "layer"),
        populations=(4, 4),
        generations=(1, 1),
        num_layers=2,
        min_active_adapters=0,
        max_active_adapters=2,
    )
    evaluator = SyntheticMultiObjectiveEvaluator(
        config.agent_names, config.adapter_names, seed=3
    )
    optimizer = ComponentFactory().create_optimizer(
        config, evaluator, NullSearchObserver()
    )

    assert isinstance(optimizer, TwoStageNSGA2Optimizer)
    assert isinstance(optimizer.stages[0].search_space, FusionSearchSpace)
    assert isinstance(optimizer.stages[1].search_space, ScopedFusionSearchSpace)
    assert optimizer.stages[0].translator is not optimizer.stages[1].translator

    result = optimizer.optimize()

    assert result.generations_completed == 2
    assert result.unique_evaluations >= 8
    assert all(
        isinstance(individual.genome, ScopedFusionGenome)
        for individual in result.population
    )
