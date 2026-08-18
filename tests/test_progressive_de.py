from dataclasses import dataclass, field

from agentmerge.core.config import apply_experiment_overrides, load_experiment_config
from agentmerge.core.domain import EvaluationResult
from agentmerge.core.policies import ModuleTarget
from agentmerge.evaluation.benchmarks import SyntheticMultiObjectiveEvaluator
from agentmerge.optimizers.base import NullSearchObserver
from agentmerge.optimizers.de.genome import DEIndividual
from agentmerge.optimizers.genome import ScopedFusionGenome
from agentmerge.optimizers.progressive_de import (
    ProgressiveDEOptimizer,
    build_refinement_partitions,
)
from agentmerge.optimizers.progressive_de.transition import (
    project_to_refined_space,
    transfer_population,
)
from agentmerge.optimizers.representation import (
    FusionSearchSpace,
    ScopedFusionSearchSpace,
)
from agentmerge.runtime.experiment import ComponentFactory


def test_recursive_partitions_are_nested_and_finish_layerwise():
    partitions = build_refinement_partitions(5, (1, 2, 4, 5))

    assert [len(item.intervals) for item in partitions] == [1, 2, 4, 5]
    assert partitions[-1].intervals == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    )
    for parent, child in zip(partitions, partitions[1:]):
        for start, end in child.intervals:
            assert any(
                left <= start and end <= right
                for left, right in parent.intervals
            )


def test_refinement_projection_only_rebuilds_the_chromosome():
    agents = ("agent",)
    adapters = ("a", "b", "c")
    global_space = FusionSearchSpace(agents, adapters, 0, 3)
    partitions = build_refinement_partitions(4, (1, 2, 4))
    two_group_space = ScopedFusionSearchSpace(
        agents, adapters, partitions[1], 0, 3
    )
    four_group_space = ScopedFusionSearchSpace(
        agents, adapters, partitions[2], 0, 3
    )
    global_genome = global_space.seed_from_spec(
        {"agent": {"a": 0.25, "b": 0.75}}
    )

    two_group = project_to_refined_space(
        global_genome, global_space, two_group_space
    )
    four_group = project_to_refined_space(
        two_group, two_group_space, four_group_space
    )

    assert all(
        row == [0.25, 0.75, 0.0]
        for row in four_group.agent_scope_weights["agent"].values()
    )
    assert two_group_space.granularity.scope_for(
        ModuleTarget(layer_index=3)
    ) == two_group_space.scope_names[1]


def test_population_transfer_is_one_to_one_and_inherits_fitness():
    agents = ("agent",)
    adapters = ("a", "b", "c")
    global_space = FusionSearchSpace(agents, adapters, 0, 3)
    scoped_space = ScopedFusionSearchSpace(
        agents,
        adapters,
        build_refinement_partitions(4, (1, 2, 4))[1],
        0,
        3,
    )

    evaluations = (
        EvaluationResult({"quality": 0.4}),
        EvaluationResult({"quality": 0.7}),
    )
    population = (
        DEIndividual(
            global_space.seed_from_spec({"agent": {"a": 0.5, "b": 0.5}}),
            fitness=(-0.4,),
            evaluation=evaluations[0],
        ),
        DEIndividual(
            global_space.seed_from_spec({"agent": {"b": 0.5, "c": 0.5}}),
            fitness=(-0.7,),
            evaluation=evaluations[1],
        ),
    )
    transferred = transfer_population(
        population,
        global_space,
        scoped_space,
        expected_size=2,
    )

    assert len(transferred) == 2
    assert [item.fitness for item in transferred] == [(-0.4,), (-0.7,)]
    assert [item.evaluation for item in transferred] == list(evaluations)
    assert all(
        isinstance(item.genome, ScopedFusionGenome) for item in transferred
    )
    assert all(
        len(set(tuple(row) for row in item.genome.agent_scope_weights["agent"].values()))
        == 1
        for item in transferred
    )


@dataclass
class RecordingObserver(NullSearchObserver):
    events: list = field(default_factory=list)
    evaluated_genome_ids: set = field(default_factory=set)

    def on_evaluation(self, individual, cached=False):
        self.evaluated_genome_ids.add(individual.genome.fingerprint())

    def on_event(self, event_type, payload=None):
        self.events.append((event_type, dict(payload or {})))


def test_progressive_de_runs_three_stages_without_personalized_pool():
    config = load_experiment_config("experiments/smoke.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="progressive_de",
        refinement_groups=(1, 2, 4),
        num_layers=4,
        populations=(6,),
        generations=(2,),
        neighborhood_size=3,
    )
    observer = RecordingObserver()
    evaluator = SyntheticMultiObjectiveEvaluator(
        config.agent_names, config.adapter_names, seed=3
    )
    optimizer = ComponentFactory().create_optimizer(config, evaluator, observer)

    assert isinstance(optimizer, ProgressiveDEOptimizer)
    assert [stage.search_space.dimension for stage in optimizer.stages] == [
        len(config.agent_names) * len(config.adapters),
        len(config.agent_names) * 2 * len(config.adapters),
        len(config.agent_names) * 4 * len(config.adapters),
    ]
    assert all(
        tuple(stage.search_space.adapter_names) == config.adapter_names
        for stage in optimizer.stages
    )

    result = optimizer.optimize()

    assert result.generations_completed == 6
    assert result.pareto_front
    assert all(
        isinstance(individual.genome, ScopedFusionGenome)
        for individual in result.population
    )
    assert {
        individual.genome.fingerprint() for individual in result.population
    } <= observer.evaluated_genome_ids
    assert {
        individual.genome.fingerprint() for individual in result.pareto_front
    } <= observer.evaluated_genome_ids
    starts = [payload for event, payload in observer.events if event == "search_stage_started"]
    assert [item["group_count"] for item in starts] == [1, 2, 4]
    assert starts[0]["inherited_population_count"] == 0
    assert starts[0]["fitness_inherited"] is False
    assert starts[1]["inherited_population_count"] == 6
    assert starts[1]["fitness_inherited"] is True
    assert starts[2]["inherited_population_count"] == 6
    generations = [
        payload
        for event, payload in observer.events
        if event == "generation_started"
    ]
    assert len(generations) == 6
    assert all(item["candidate_count"] == 6 for item in generations)
    objective_summaries = [
        payload
        for event, payload in observer.events
        if event == "objective_best"
    ]
    generation_summaries = [
        item for item in objective_summaries if item["scope"] == "generation"
    ]
    stage_summaries = [
        item for item in objective_summaries if item["scope"] == "stage"
    ]
    assert len(generation_summaries) == 6 * len(config.objectives)
    assert len(stage_summaries) == 3 * len(config.objectives)
    assert {item["objective"] for item in stage_summaries} == {
        objective.metric for objective in config.objectives
    }
    assert not any(event == "adapter_pool_release" for event, _ in observer.events)
