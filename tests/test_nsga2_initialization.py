from agentmerge.core.config import apply_experiment_overrides, load_experiment_config
from agentmerge.optimizers.nsga2 import build_nsga2_components
from agentmerge.optimizers.two_stage_nsga2 import TwoStageNSGA2Optimizer
from agentmerge.optimizers.base import NullSearchObserver
from agentmerge.evaluation.benchmarks import SyntheticMultiObjectiveEvaluator
from agentmerge.runtime.experiment import ComponentFactory


def test_nsga2_builds_role_seeds_from_adapter_admission_metadata():
    config = load_experiment_config("experiments/qwen3_pilot_v01.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="nsga2",
        search_granularity=("global",),
        min_active_adapters=0,
        max_active_adapters=2,
    )

    stage = build_nsga2_components(config)
    seeds = [genome.to_dict() for genome in stage.initial_genomes]

    assert len(seeds) == 5
    assert {tuple(name for name, weight in zip(config.adapter_names, seed["critic"]) if weight) for seed in seeds if sum(seed["critic"])} == {
        ("qasper_rag",),
        ("narrativeqa_rag",),
        ("knapsack_runtime",),
    }
    assert {tuple(name for name, weight in zip(config.adapter_names, seed["solver"]) if weight) for seed in seeds if sum(seed["solver"])} == {
        ("robot_planner",),
        ("platformio",),
    }


def test_two_stage_uses_admission_initializer_only_in_first_stage():
    config = load_experiment_config("experiments/qwen3_pilot_v01.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="two_stage_nsga2",
        search_granularity=("global", "layer"),
        populations=(8, 8),
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
    assert len(optimizer.stages[0].initial_genomes) == 5
    assert optimizer.stages[1].initial_genomes == ()
