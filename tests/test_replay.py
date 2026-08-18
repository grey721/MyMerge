import json
from contextlib import contextmanager

from agentmerge.backends.base import ModelBackend
from agentmerge.core.config import (
    AdapterConfig,
    AgentConfig,
    BenchmarkConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    ObjectiveConfig,
    RuntimeConfig,
    TopologyConfig,
)
from agentmerge.core.domain import GenerationResult
from agentmerge.evaluation.replay import CandidateReplayRunner


class _ReplayBackend(ModelBackend):
    def __init__(self):
        self._adapters = (
            AdapterConfig("a", "a", rank=8),
            AdapterConfig("b", "b", rank=16),
        )
        self.closed = False
        self.sessions = 0

    @property
    def adapter_names(self):
        return tuple(item.name for item in self._adapters)

    @property
    def adapter_specs(self):
        return self._adapters

    def generate(self, request, mixture):
        return GenerationResult(
            text="4",
            latency_seconds=0.01,
            prompt_tokens=4,
            completion_tokens=1,
            metadata={"adapter": "candidate"},
        )

    @contextmanager
    def candidate_session(self):
        self.sessions += 1
        yield

    def runtime_snapshot(self):
        return {
            "backend": "test",
            "sessions": self.sessions,
            "temporary_adapter_names": [],
        }

    def close(self):
        self.closed = True


def _config(benchmark_path):
    return ExperimentConfig(
        name="replay",
        output_dir="outputs",
        model=ModelConfig(
            backend="peft",
            name_or_path="unused",
            generation=GenerationConfig(max_new_tokens=4),
        ),
        adapters=(
            AdapterConfig("a", "a", rank=8),
            AdapterConfig("b", "b", rank=16),
        ),
        agents=(AgentConfig("solver", "solver", "solve"),),
        topology=TopologyConfig(
            edges=(), output_agent="solver", kind="fixed"
        ),
        benchmark=BenchmarkConfig(
            kind="jsonl",
            path=str(benchmark_path),
            options={"scorer": "numeric", "inference_batch_size": 1},
        ),
        objectives=(ObjectiveConfig("quality", "maximize"),),
        runtime=RuntimeConfig(mode="multi_gpu", gpu_ids=("0",)),
    )


def test_candidate_replay_reuses_production_evaluator_without_answers(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {"id": "case-1", "input": "2+2", "target": "4", "group": "math"}
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "genome_id": "known",
                "genotype": {"solver": [0.5, 0.5]},
            }
        ),
        encoding="utf-8",
    )
    backend = _ReplayBackend()

    report = CandidateReplayRunner(
        _config(benchmark_path),
        visible_device="3",
        backend_factory=lambda: backend,
    ).run(str(candidate_path), repeats=2)

    assert report.passed
    assert report.successful_repeats == 2
    assert backend.sessions == 2
    assert backend.closed
    assert all(attempt.metrics["quality"] == 1.0 for attempt in report.attempts)
    serialized = json.dumps(report.to_dict())
    assert "prediction" not in serialized
    assert "2+2" not in serialized


def test_candidate_replay_rejects_scoped_candidate(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps({"input": "x", "target": "4"}) + "\n",
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {"genotype": {"solver": {"layer.0": [0.5, 0.5]}}}
        ),
        encoding="utf-8",
    )

    runner = CandidateReplayRunner(
        _config(benchmark_path),
        visible_device="3",
        backend_factory=_ReplayBackend,
    )
    try:
        runner.run(str(candidate_path))
    except ValueError as exc:
        assert "global Agent" in str(exc)
    else:
        raise AssertionError("Expected a scoped-candidate validation failure")
