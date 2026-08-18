from contextlib import contextmanager

from agentmerge.backends.base import ModelBackend
from agentmerge.backends.materialization import BASE_ADAPTER_NAME
from agentmerge.core.config import (
    AdapterConfig,
    AgentConfig,
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TopologyConfig,
)
from agentmerge.evaluation.diagnostics import BackendSmokeRunner
from agentmerge.core.domain import AdapterMixture, GenerationResult


class _SmokeBackend(ModelBackend):
    def __init__(self):
        self._adapters = (
            AdapterConfig("a", "a", rank=8),
            AdapterConfig("b", "b", rank=16),
        )
        self._in_session = False
        self._temporary = []

    @property
    def adapter_names(self):
        return tuple(item.name for item in self._adapters)

    @property
    def adapter_specs(self):
        return self._adapters

    def generate(self, request, mixture: AdapterMixture):
        if not mixture.active:
            adapter = BASE_ADAPTER_NAME
        elif len(mixture.active) == 1:
            adapter = mixture.active[0][0]
        else:
            adapter = "candidate_fusion_test"
            if self._in_session and adapter not in self._temporary:
                self._temporary.append(adapter)
        return GenerationResult(
            text="4",
            latency_seconds=0.01,
            prompt_tokens=4,
            completion_tokens=1,
            metadata={
                "adapter": adapter,
                "activation_latency_seconds": (
                    0.03 if self._temporary else 0.001
                ),
                "model_generation_latency_seconds": 0.008,
                "fusion_cleanup_latency_seconds": 0.001,
            },
        )

    @contextmanager
    def candidate_session(self):
        self._in_session = True
        try:
            yield
        finally:
            self._temporary = []
            self._in_session = False

    def runtime_snapshot(self):
        loaded = list(self.adapter_names) + list(self._temporary)
        return {
            "loaded_adapter_names": loaded,
            "temporary_adapter_names": list(self._temporary),
            "candidate_adapter_names": list(self._temporary),
            "fusion": {
                "materializer": {
                    "materialization_count": int(bool(self._temporary)),
                    "materialization_seconds": 0.03,
                    "release_count": int(not self._temporary),
                    "release_seconds": 0.002,
                }
            },
        }


def _config():
    return ExperimentConfig(
        name="smoke",
        output_dir="outputs",
        model=ModelConfig(backend="peft", name_or_path="unused"),
        adapters=(
            AdapterConfig("a", "a", rank=8),
            AdapterConfig("b", "b", rank=16),
        ),
        agents=(AgentConfig("agent", "solver", "solve"),),
        topology=TopologyConfig(
            edges=(), output_agent="agent", kind="fixed"
        ),
        benchmark=BenchmarkConfig(kind="synthetic"),
        objectives=(ObjectiveConfig("quality", "maximize"),),
    )


def test_backend_smoke_runner_checks_candidate_scoped_lifecycle():
    backend = _SmokeBackend()
    report = BackendSmokeRunner(
        _config(), backend_factory=lambda: backend
    ).run(max_new_tokens=4)

    assert report.passed
    assert [item.label for item in report.generations] == [
        "base",
        "one_hot:a",
        "fusion:first",
        "fusion:reused",
    ]
    assert all(report.lifecycle_checks.values())
    assert report.after_candidate_session["temporary_adapter_names"] == []
