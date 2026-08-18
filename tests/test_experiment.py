from dataclasses import replace
import json
from pathlib import Path

from agentmerge.core.config import load_experiment_config
import pytest

from agentmerge.runtime.experiment import ExperimentRunner, assert_output_safe


ROOT = Path(__file__).resolve().parents[1]


def test_output_boundary_rejects_model_and_answer_payloads():
    assert_output_safe({"genotype": {"agent": [0.5]}})
    with pytest.raises(ValueError, match="Unsafe output field"):
        assert_output_safe({"adapter_weights": [1, 2, 3]})
    with pytest.raises(ValueError, match="Unsafe output field"):
        assert_output_safe({"prediction": "answer"})


def test_experiment_writes_pareto_outputs(tmp_path):
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    config = replace(
        config,
        output_dir=str(tmp_path),
        search=replace(config.search, population_size=8, generations=2),
    )
    summary = ExperimentRunner(config).run()
    run_dir = Path(summary.run_dir)
    assert summary.pareto_size > 0
    resolved_path = run_dir / "resolved_config.yaml"
    assert resolved_path.exists()
    resolved = load_experiment_config(str(resolved_path))
    assert resolved.name == config.name
    assert "search:" not in resolved_path.read_text(encoding="utf-8")
    assert (run_dir / "pareto_front.json").exists()
    assert (run_dir / "final_population.json").exists()
    manifest = json.loads(
        (run_dir / "output_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["persistence_policy"] == {
        "stores_model_weights": False,
        "stores_adapter_weights": False,
        "stores_predictions": False,
        "stores_targets": False,
        "stores_genotypes": True,
    }
    assert manifest["search"]["stages"][0]["population_size"] == 8
    assert "data_sha256" not in manifest["benchmark"]
    log_path = run_dir / "run.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "run_started" in log_text
    assert "GEN 1/2 | Started (8)" in log_text
    assert "GEN 1/2 | Completed (8)" in log_text
    assert "GEN 2/2 | Completed" in log_text
    assert "best_quality=" in log_text
    for objective in config.objectives:
        assert "OBJ: {}".format(objective.metric) in log_text
    assert "run_completed" in log_text
    assert "CAND " in log_text
    assert "metrics={" not in log_text
    assert "runtime={" not in log_text
    assert "optimizer_options" not in log_text
    checkpoint_path = run_dir / "checkpoints" / "generation_0001.json"
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == 2
    assert all(
        set(item) == {
            "genome_id",
            "evaluation_ref",
            "rank",
            "crowding_distance",
        }
        for item in checkpoint["population"]
    )
    assert all(
        (checkpoint_path.parent / item["evaluation_ref"]).resolve().exists()
        for item in checkpoint["population"]
    )

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )
    event_types = {event["event"] for event in events}
    assert {
        "run_started",
        "generation_started",
        "candidate_completed",
        "checkpoint_written",
        "run_completed",
    } <= event_types
    evaluation_files = tuple((run_dir / "evaluations").glob("*.json"))
    assert len(evaluation_files) == summary.unique_evaluations
    evaluation = json.loads(evaluation_files[0].read_text(encoding="utf-8"))
    assert "genotype" in evaluation
    assert "details" not in evaluation
    assert "weights" not in evaluation
    assert "prediction" not in json.dumps(evaluation)
