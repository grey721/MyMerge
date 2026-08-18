import json
import os
import time
from dataclasses import replace
from pathlib import Path

from agentmerge.core.config import (
    BenchmarkConfig,
    ObjectiveConfig,
    RuntimeConfig,
    load_experiment_config,
)
from agentmerge.runtime.experiment import ExperimentRunner
from agentmerge.runtime.distributed import (
    MultiGpuFusionPlanEvaluator,
    _configure_vllm_worker_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def test_vllm_workers_receive_stable_isolated_compile_caches(
    tmp_path, monkeypatch
):
    base = tmp_path / "vllm-cache"
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(base))
    first = _configure_vllm_worker_cache("0")

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(base))
    second = _configure_vllm_worker_cache("1")

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(base))
    repeated = _configure_vllm_worker_cache("0")

    assert first == repeated
    assert first != second
    assert Path(first).parent == base / "agentmerge-workers"
    assert Path(second).parent == base / "agentmerge-workers"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(repeated)
    assert os.environ["TRITON_CACHE_DIR"].startswith(repeated)


def test_multi_gpu_process_pool_with_mock_backend(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "one",
                        "input": "Return [expected: 42]",
                        "target": "42",
                        "group": "a",
                    }
                ),
                json.dumps(
                    {
                        "id": "two",
                        "input": "Return [expected: 7]",
                        "target": "7",
                        "group": "b",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    config = replace(
        config,
        output_dir=str(tmp_path / "outputs"),
        benchmark=BenchmarkConfig(
            kind="jsonl",
            path=str(benchmark_path),
            options={"scorer": "exact_match"},
        ),
        objectives=(
            ObjectiveConfig("task_quality.a", "maximize"),
            ObjectiveConfig("task_quality.b", "maximize"),
            ObjectiveConfig("active_adapters", "minimize"),
        ),
        runtime=RuntimeConfig(
            mode="multi_gpu",
            gpu_ids=("mock-gpu-0", "mock-gpu-1"),
            start_method="spawn",
            preload_workers=True,
            evaluation_timeout_seconds=30,
            max_retries=0,
            cpu_threads_per_worker=1,
        ),
        search=replace(
            config.search,
            population_size=4,
            generations=1,
            workers=1,
        ),
    )

    summary = ExperimentRunner(config).run()
    checkpoint = json.loads(
        (
            Path(summary.run_dir)
            / "checkpoints"
            / "generation_0000.json"
        ).read_text(encoding="utf-8")
    )
    run_dir = Path(summary.run_dir)
    evaluations = [
        json.loads(
            (run_dir / "checkpoints" / item["evaluation_ref"])
            .resolve()
            .read_text(encoding="utf-8")
        )
        for item in checkpoint["population"]
    ]
    used_gpus = {
        item["runtime"]["gpu_id"]
        for item in evaluations
    }
    assert used_gpus == {"mock-gpu-0", "mock-gpu-1"}
    assert all(
        "fusion" in item["runtime"]
        for item in evaluations
    )
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed_events = [
        event for event in events if event["event"] == "candidate_completed"
    ]
    assert completed_events
    assert all("fusion" in event["data"]["runtime"] for event in completed_events)
    runtime = json.loads(
        (Path(summary.run_dir) / "runtime.json").read_text(encoding="utf-8")
    )
    assert runtime["configured_gpu_ids"] == ["mock-gpu-0", "mock-gpu-1"]
    assert runtime["controller_pid"] > 0
    assert runtime["process_group_id"] > 0
    assert runtime["worker_parent_death_signal"] == "SIGTERM"
    assert len(runtime["workers"]) == 2
    assert all(
        item["metrics"]["quality"] == 1.0
        for item in evaluations
    )
    assert all(
        item["metrics"]["task_quality.a"] == 1.0
        and item["metrics"]["task_quality.b"] == 1.0
        for item in evaluations
    )

    evaluator = MultiGpuFusionPlanEvaluator(config)
    try:
        report = evaluator.release_unreferenced_adapters(
            {
                adapter.name: int(coordinate_index < 2)
                for coordinate_index, adapter in enumerate(config.adapters)
            }
        )
    finally:
        evaluator.close()
    assert report["worker_count"] == 2
    assert report["supported"] is True
    assert report["released_adapter_keys"] == [
        adapter.name for adapter in config.adapters[2:]
    ]
    assert len(report["workers"]) == 2


def test_multi_gpu_abort_terminates_worker_processes(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "id": "one",
                "input": "Return [expected: 42]",
                "target": "42",
                "group": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    config = replace(
        config,
        benchmark=BenchmarkConfig(
            kind="jsonl",
            path=str(benchmark_path),
            options={"scorer": "exact_match"},
        ),
        runtime=RuntimeConfig(
            mode="multi_gpu",
            gpu_ids=("mock-gpu-0",),
            start_method="spawn",
            preload_workers=True,
            evaluation_timeout_seconds=30,
            max_retries=0,
            cpu_threads_per_worker=1,
        ),
    )

    evaluator = MultiGpuFusionPlanEvaluator(config)
    worker_pid = evaluator.statuses[0].process_id
    evaluator.abort()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("GPU worker remained alive after evaluator.abort()")
