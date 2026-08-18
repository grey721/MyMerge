import pytest
from pathlib import Path

from agentmerge.core.config import (
    ObjectiveConfig,
    apply_experiment_overrides,
    load_experiment_config,
)
from agentmerge.evaluation.metrics import BenchmarkMetricCatalog
from agentmerge.optimizers.fitness import FitnessSpace
from agentmerge.runtime.experiment import ComponentFactory


ROOT = Path(__file__).resolve().parents[1]


def test_task_metric_catalog_aggregates_and_advertises_every_group():
    catalog = BenchmarkMetricCatalog.from_groups(
        ("hotpotqa", "math500", "hotpotqa")
    )

    metrics = catalog.aggregate_quality(
        {
            "math500": (1.0, 0.0),
            "hotpotqa": (1.0, 1.0, 0.0),
        }
    )

    assert catalog.task_names == ("hotpotqa", "math500")
    assert "task_quality.math500" in catalog.names
    assert "task_quality.hotpotqa" in catalog.names
    assert metrics["quality"] == pytest.approx(0.6)
    assert metrics["worst_group_quality"] == pytest.approx(0.5)
    assert metrics["task_quality.math500"] == pytest.approx(0.5)
    assert metrics["task_quality.hotpotqa"] == pytest.approx(2.0 / 3.0)


def test_metric_catalog_aggregates_configured_quality_domains():
    catalog = BenchmarkMetricCatalog.from_groups(
        ("math", "code_a", "code_b"),
        quality_domains={
            "mathematics": ("math",),
            "code_logic": ("code_a", "code_b"),
        },
    )

    metrics = catalog.aggregate_quality(
        {
            "math": (1.0, 0.0),
            "code_a": (1.0, 1.0),
            "code_b": (0.0,),
        }
    )

    assert "domain_quality.mathematics" in catalog.names
    assert "domain_quality.code_logic" in catalog.names
    assert metrics["domain_quality.mathematics"] == pytest.approx(0.5)
    # Domain quality is a macro-average of task means, not a sample-weighted
    # micro-average: mean([1.0, 0.0]) = 0.5.
    assert metrics["domain_quality.code_logic"] == pytest.approx(0.5)


def test_metric_catalog_rejects_domain_with_unknown_task():
    with pytest.raises(ValueError, match="unknown task.*missing"):
        BenchmarkMetricCatalog.from_groups(
            ("math",),
            quality_domains={"reasoning": ("math", "missing")},
        )


def test_fitness_space_rejects_unavailable_metrics_before_search():
    space = FitnessSpace(
        (ObjectiveConfig("task_quality.unknown", "maximize"),)
    )

    with pytest.raises(ValueError, match="Unknown objective metric.*unknown"):
        space.validate_metrics(("quality", "task_quality.math500"))


def test_fitness_space_encodes_directions_and_restores_objective_values():
    space = FitnessSpace(
        (
            ObjectiveConfig("quality", "maximize"),
            ObjectiveConfig("latency_seconds", "minimize"),
        )
    )

    fitness = space.encode({"quality": 0.75, "latency_seconds": 2.5})

    assert fitness == (-0.75, 2.5)
    assert space.decode(fitness) == {
        "quality": 0.75,
        "latency_seconds": 2.5,
    }


def test_real_benchmark_task_objectives_validate_without_loading_model():
    config = load_experiment_config(
        str(ROOT / "experiments" / "qwen3_vllm_research_v01.yaml")
    )
    config = apply_experiment_overrides(
        config,
        algorithm="de",
        search_granularity=("adapter",),
        populations=(12,),
        generations=(1,),
        objectives=(
            "task_quality.math500:maximize",
            "task_quality.hotpotqa:maximize",
            "fusion_rank_proxy:minimize",
        ),
    )

    description = ComponentFactory().validate_search(config)

    assert "task_quality.math500" in description["available_objective_metrics"]
    assert "task_quality.hotpotqa" in description["available_objective_metrics"]


def test_research_v02_uses_four_domain_quality_objectives():
    config = load_experiment_config(
        str(
            ROOT
            / "experiments"
            / "qwen3_vllm_factor_linear_research_v02.yaml"
        )
    )
    config = apply_experiment_overrides(
        config,
        algorithm="de",
        search_granularity=("adapter",),
        populations=(12,),
        generations=(1,),
    )

    description = ComponentFactory().validate_search(config)
    expected = (
        "domain_quality.math",
        "domain_quality.code_logic",
        "domain_quality.multi_hop_qa",
        "domain_quality.planning_game",
    )

    assert tuple(objective.metric for objective in config.objectives) == expected
    assert set(expected).issubset(description["available_objective_metrics"])


def test_research_v03_keeps_tool_use_available_but_not_default():
    config = load_experiment_config(
        str(
            ROOT
            / "experiments"
            / "qwen3_vllm_factor_linear_research_v03.yaml"
        )
    )
    config = apply_experiment_overrides(
        config,
        algorithm="de",
        search_granularity=("adapter",),
        populations=(12,),
        generations=(1,),
    )

    description = ComponentFactory().validate_search(config)
    expected = (
        "domain_quality.language_reasoning",
        "domain_quality.mathematics",
        "domain_quality.code",
    )

    assert tuple(objective.metric for objective in config.objectives) == expected
    assert set(expected).issubset(description["available_objective_metrics"])
    assert "domain_quality.tool_use" in (
        description["available_objective_metrics"]
    )
