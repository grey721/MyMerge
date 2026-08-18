from dataclasses import replace
from pathlib import Path

import pytest

from agentmerge.core.config import (
    ConfigError,
    apply_experiment_overrides,
    load_experiment_config,
)
from agentmerge.evaluation.benchmarks import SyntheticMultiObjectiveEvaluator
from agentmerge.runtime.experiment import ComponentFactory
from agentmerge.optimizers.base import Optimizer, SearchResult
from agentmerge.optimizers.registry import OptimizerRegistry
from agentmerge.optimizers.base import NullSearchObserver
from agentmerge.optimizers.nsga2 import NSGA2Config, NSGA2Optimizer
from agentmerge.optimizers.de import DecompositionDEOptimizer


ROOT = Path(__file__).resolve().parents[1]


class MarkerOptimizer(Optimizer):
    def optimize(self) -> SearchResult:
        return SearchResult((), (), 0, 0, ())


def test_nsga2_owns_and_validates_operator_options():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    config = apply_experiment_overrides(
        config, mutation_probability=0.25, mutation_sigma=0.15
    )
    settings = NSGA2Config.from_experiment_config(config)
    assert settings.mutation_probability == pytest.approx(0.25)
    assert settings.mutation_sigma == pytest.approx(0.15)

    invalid = replace(
        config,
        search=replace(
            config.search,
            options={"nsga2": {"mutation_probability": 2.0}},
        ),
    )
    with pytest.raises(ConfigError, match="mutation_probability"):
        NSGA2Config.from_experiment_config(invalid)


def test_nsga2_default_does_not_limit_active_adapter_count():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    settings = NSGA2Config.from_experiment_config(config)

    assert settings.min_active_adapters == 0
    assert settings.max_active_adapters == len(config.adapters)


def test_component_factory_accepts_optimizer_without_core_changes():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    config = replace(
        config,
        search=replace(
            config.search,
            algorithm="marker",
            population_size=1,
            options={"marker": {"custom_parameter": 7}},
        ),
    )
    received = {}

    def build_marker(experiment, context):
        received["options"] = experiment.search.options["marker"]
        received["agent_names"] = experiment.agent_names
        return MarkerOptimizer()

    registry = OptimizerRegistry()
    registry.register("marker", build_marker)
    evaluator = SyntheticMultiObjectiveEvaluator(
        config.agent_names, config.adapter_names, seed=3
    )
    optimizer = ComponentFactory(registry).create_optimizer(
        config, evaluator, NullSearchObserver()
    )

    assert isinstance(optimizer, MarkerOptimizer)
    assert received["options"] == {"custom_parameter": 7}
    assert received["agent_names"] == config.agent_names


def test_default_registry_builds_nsga2():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    evaluator = SyntheticMultiObjectiveEvaluator(
        config.agent_names, config.adapter_names, seed=3
    )
    optimizer = ComponentFactory().create_optimizer(
        config, evaluator, NullSearchObserver()
    )
    assert isinstance(optimizer, NSGA2Optimizer)


def test_default_registry_builds_de():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    config = apply_experiment_overrides(
        config,
        algorithm="de",
        search_granularity=("adapter",),
        populations=(6,),
        generations=(1,),
        neighborhood_size=3,
    )
    evaluator = SyntheticMultiObjectiveEvaluator(
        config.agent_names, config.adapter_names, seed=3
    )
    optimizer = ComponentFactory().create_optimizer(
        config, evaluator, NullSearchObserver()
    )
    assert isinstance(optimizer, DecompositionDEOptimizer)


def test_registry_rejects_duplicate_names():
    registry = OptimizerRegistry()
    registry.register("marker", lambda search, context: MarkerOptimizer())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("MARKER", lambda search, context: MarkerOptimizer())
