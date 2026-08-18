from agentmerge.core.logger import ExperimentLogger


def test_experiment_logger_formats_progress_without_nested_payloads(tmp_path):
    logger = ExperimentLogger(
        tmp_path,
        "logger-test",
        console=False,
        generation_width=3,
    )
    logger.event(
        "base_model_loading_started",
        {"backend": "vllm", "engine_mode": "V1", "devices": ["1", "2"]},
    )
    logger.event(
        "base_model_loading_completed",
        {
            "elapsed_seconds": 12.3456,
            "worker_count": 2,
            "evaluator": "MultiGpuFusionPlanEvaluator",
        },
    )
    logger.event(
        "search_validated",
        {
            "algorithm": "progressive_de",
            "adapter_pool_mode": "fixed_global_pool",
            "adapter_count": 6,
            "transition": "recursive_bisection_with_inherited_jitter",
            "stages": [{"large_payload": True}],
        },
    )
    logger.event(
        "generation_started",
        {"generation": 0, "generation_count": 100, "candidate_count": 12},
    )
    logger.event(
        "candidate_completed",
        {
            "plan_id": "abcdef1234567890",
            "metrics": {
                "quality": 0.51234,
                "worst_group_quality": 0.4,
                "latency_seconds": 3.12345,
                "total_tokens": 456.789,
            },
            "runtime": {"gpu_id": "2", "large_payload": {"x": 1}},
        },
    )
    logger.event(
        "generation_completed",
        {
            "generation": 0,
            "generation_count": 100,
            "unique_evaluations": 12,
            "primary_objective": "quality",
            "best_metrics": {
                "quality": 0.81234,
                "worst_group_quality": 0.67891,
            },
        },
    )
    logger.event(
        "generation_completed",
        {
            "generation": 2,
            "generation_count": 100,
            "unique_evaluations": 36,
            "primary_objective": "domain_quality.math",
            "best_metrics": {
                "domain_quality.math": 0.75,
                "domain_quality.code_logic": 0.625,
            },
        },
    )
    logger.event(
        "generation_completed",
        {
            "generation": 1,
            "generation_count": 100,
            "unique_evaluations": 24,
            "primary_objective": "task_quality.math500",
            "best_metrics": {
                "task_quality.math500": 0.875,
                "task_quality.hotpotqa": 0.625,
                "fusion_rank_proxy": 64.0,
            },
        },
    )
    logger.event(
        "generation_started",
        {
            "generation": 101,
            "stage_generation": 1,
            "generation_count": 12,
            "candidate_count": 12,
            "stage_index": 1,
            "stage_count": 12,
        },
    )
    logger.event(
        "generation_completed",
        {
            "generation": 101,
            "stage_generation": 1,
            "generation_count": 12,
            "unique_evaluations": 24,
            "stage_index": 1,
            "stage_count": 12,
        },
    )
    logger.event(
        "candidate_completed",
        {"plan_id": "stagecandidate123", "stage_index": 1},
    )
    logger.event(
        "objective_best",
        {
            "scope": "generation",
            "generation": 1,
            "generation_count": 12,
            "stage_index": 1,
            "stage_count": 12,
            "objective": "domain_quality.code",
            "objective_label": "code",
            "objective_width": 18,
            "candidate_id": "abcdef1234567890",
            "quality": 0.81234,
            "worst_group_quality": 0.75,
            "latency_seconds": 1.6314,
            "total_tokens": 1420.68,
            "device": "4",
        },
    )
    logger.event(
        "adapter_pool_release",
        {
            "supported": True,
            "released_source_count": 3,
            "resident_source_count": 5,
            "released_source_bytes": 786432000,
            "resident_source_bytes": 1310720000,
            "worker_count": 6,
            "workers": [{"large_payload": True}],
        },
    )
    logger.close()

    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert (
        "Base model | Loading started backend=vllm engine_mode=V1 "
        "devices=[\"1\",\"2\"]"
    ) in text
    assert "Base model | Loading completed elapsed_seconds=12.346" in text
    assert "search_validated | algorithm=progressive_de" in text
    assert "adapter_count=6" in text
    assert "GEN   1/100 | Started (12)" in text
    assert "CAND abcdef123456 | Completed quality=0.512" in text
    assert "worst_group=0.400" in text
    assert "latency_s=3.123" in text
    assert "tokens=456.789" in text
    assert (
        "GEN   1/100 | Completed (12) best_quality=0.812 worst_group=0.679"
        in text
    )
    assert (
        "GEN   2/100 | Completed (24) best_math500=0.875 "
        "hotpotqa=0.625 rank=64.000"
    ) in text
    assert (
        "GEN   3/100 | Completed (36) best_math=0.750 code_logic=0.625"
        in text
    )
    assert "Stage  2/12 | GEN  2/12 | Started (12)" in text
    assert "Stage  2/12 | GEN  2/12 | Completed (24)" in text
    assert "CAND stagecandida | Completed" in text
    assert "CAND S2" not in text
    assert (
        "Stage  2/12 | GEN  2/12 | OBJ: code               | "
        "Best CAND: abcdef123456 | Quality: [0.750, 0.812] | "
        "Latency: 1.631 s | Tokens: 1420.680 | Device: 4"
    ) in text
    assert "metrics={" not in text
    assert "runtime={" not in text
    assert "large_payload" not in text
    assert (
        "Adapter pool | Released released_source_count=3 "
        "resident_source_count=5 released_source_bytes=786432000 "
        "resident_source_bytes=1310720000 worker_count=6"
    ) in text
