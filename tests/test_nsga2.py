import random

from agentmerge.core.config import ObjectiveConfig
from agentmerge.core.domain import EvaluationResult
from agentmerge.evaluation.benchmarks import SyntheticMultiObjectiveEvaluator
from agentmerge.optimizers.genome import FusionGenome
from agentmerge.optimizers.genome import NSGA2Individual
from agentmerge.optimizers.representation import FusionSearchSpace
from agentmerge.optimizers.fitness import FitnessSpace
from agentmerge.optimizers.nsga2 import (
    NSGA2Optimizer,
    NSGA2Variation,
    create_plan_translator,
    fast_non_dominated_sort,
)


def _individual(first, second):
    individual = NSGA2Individual(FusionGenome({"agent": [1.0]}))
    individual.evaluation = EvaluationResult({"x": first, "y": second})
    individual.fitness = (first, second)
    return individual


def test_variation_returns_valid_genome():
    rng = random.Random(11)
    space = FusionSearchSpace(
        agent_names=("planner", "solver"),
        adapter_names=("a", "b", "c"),
        min_active=1,
        max_active=2,
    )
    variation = NSGA2Variation(space, 1.0, 1.0, 1.0, 0.2)
    child = variation.crossover(space.sample(rng), space.sample(rng), rng)
    child = variation.mutate(child, rng)
    for vector in child.agent_weights.values():
        assert abs(sum(vector) - 1.0) < 1e-9
        assert 1 <= sum(value > 0 for value in vector) <= 2


def test_non_dominated_sort():
    population = [
        _individual(1, 3),
        _individual(2, 2),
        _individual(3, 1),
        _individual(4, 4),
    ]
    fronts = fast_non_dominated_sort(population)
    assert len(fronts[0]) == 3
    assert fronts[1] == [population[3]]


def test_nsga2_smoke():
    agents = ("planner", "solver", "critic")
    adapters = ("a", "b", "c", "d")
    space = FusionSearchSpace(agents, adapters, 1, 3)
    variation = NSGA2Variation(space, 0.9, 0.3, 0.2, 0.15)
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
        variation=variation,
        population_size=12,
        generations=3,
        seed=3,
        workers=2,
    )
    result = optimizer.optimize()
    assert len(result.population) == 12
    assert result.pareto_front
    assert result.unique_evaluations >= 12
    assert all(individual.rank == 0 for individual in result.pareto_front)


def test_nsga2_exposes_ask_tell_lifecycle():
    agents = ("agent",)
    adapters = ("a", "b")
    space = FusionSearchSpace(agents, adapters, 1, 2)
    optimizer = NSGA2Optimizer(
        evaluator=SyntheticMultiObjectiveEvaluator(agents, adapters, seed=3),
        fitness_space=FitnessSpace(
            (ObjectiveConfig("quality", "maximize"),)
        ),
        search_space=space,
        translator=create_plan_translator(space),
        variation=NSGA2Variation(space, 0.9, 0.3, 0.2, 0.15),
        population_size=4,
        generations=2,
        seed=3,
    )

    initial = optimizer.ask()
    assert len(initial) == 4
    assert not any(candidate.evaluated for candidate in initial)
    optimizer.evaluate_candidates(initial)
    optimizer.tell(initial)
    assert not optimizer.done

    offspring = optimizer.ask()
    optimizer.evaluate_candidates(offspring)
    optimizer.tell(offspring)

    assert optimizer.done
    assert optimizer.result().generations_completed == 2


def test_nsga2_seeds_base_when_zero_active_adapters_are_allowed():
    agents = ("planner", "critic")
    adapters = ("a", "b")
    space = FusionSearchSpace(agents, adapters, 0, 2)
    variation = NSGA2Variation(space, 0.9, 0.3, 0.2, 0.15)
    optimizer = NSGA2Optimizer(
        evaluator=SyntheticMultiObjectiveEvaluator(agents, adapters, seed=3),
        fitness_space=FitnessSpace(
            (ObjectiveConfig("quality", "maximize"),)
        ),
        search_space=space,
        translator=create_plan_translator(space),
        variation=variation,
        population_size=4,
        generations=1,
        seed=3,
    )

    initial = optimizer._initial_population()

    assert all(
        sum(weights) == 0.0
        for weights in initial[0].genome.agent_weights.values()
    )


def test_nsga2_prioritizes_unique_explicit_seed_genomes():
    agents = ("planner", "critic")
    adapters = ("a", "b")
    space = FusionSearchSpace(agents, adapters, 0, 2)
    variation = NSGA2Variation(space, 0.9, 0.3, 0.2, 0.15)
    seed = FusionGenome(
        {"planner": [0.0, 0.0], "critic": [0.0, 1.0]}
    )
    optimizer = NSGA2Optimizer(
        evaluator=SyntheticMultiObjectiveEvaluator(agents, adapters, seed=3),
        fitness_space=FitnessSpace(
            (ObjectiveConfig("quality", "maximize"),)
        ),
        search_space=space,
        translator=create_plan_translator(space),
        variation=variation,
        population_size=4,
        generations=1,
        seed=3,
        initial_genomes=(seed, seed.clone()),
    )

    initial = optimizer._initial_population()

    assert initial[0].genome.fingerprint() == seed.fingerprint()
    assert len({item.genome.fingerprint() for item in initial}) == 4
