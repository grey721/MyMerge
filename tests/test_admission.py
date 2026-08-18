from dataclasses import replace
import json
from pathlib import Path

from agentmerge.evaluation.admission import AdapterAdmissionRunner
from agentmerge.core.config import load_experiment_config


def test_adapter_admission_builds_base_global_and_rolewise_variants(tmp_path):
    config = load_experiment_config("experiments/smoke.yaml")
    config = replace(config, output_dir=str(tmp_path))
    runner = AdapterAdmissionRunner(config)

    global_variants = runner._variants(include_rolewise=False)
    rolewise_variants = runner._variants(include_rolewise=True)

    assert len(global_variants) == 1 + len(config.adapters)
    assert len(rolewise_variants) == 1 + len(config.adapters) * (
        1 + len(config.agents)
    )
    assert global_variants[0].name == "base"
    assert all(
        sum(policy.weights) == 0.0
        for _, policy in global_variants[0].plan.agent_policies
    )


def test_adapter_admission_runs_without_optimizer(tmp_path):
    config = load_experiment_config("experiments/smoke.yaml")
    config = replace(config, output_dir=str(tmp_path))

    summary = AdapterAdmissionRunner(config).run(include_rolewise=False)

    assert summary.variant_count == 1 + len(config.adapters)
    assert summary.best_variant
    assert (Path(summary.run_dir) / "admission_results.json").exists()
    log_text = (Path(summary.run_dir) / "run.log").read_text(encoding="utf-8")
    assert "admission_started" in log_text
    assert "admission_completed" in log_text
    report = json.loads(
        (Path(summary.run_dir) / "admission_results.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(report)
    assert report["schema_version"] == 2
    assert "metric_order" in report
    assert "base_metrics" not in report
    assert "details" not in serialized
    assert "prediction" not in serialized
