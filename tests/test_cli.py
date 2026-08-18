from pathlib import Path

import pytest

from agentmerge.cli import build_parser, main
from agentmerge.core.config import resolve_device_ids


ROOT = Path(__file__).resolve().parents[1]


def test_validate_accepts_whitelisted_cli_overrides(capsys):
    code = main(
        (
            "validate",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--populations",
            "8",
            "--seed",
            "7",
            "--mutation_probability",
            "0.33",
            "--device",
            "0",
            "2",
            "--batch_size",
            "4",
        )
    )

    assert code == 0
    assert '"valid": true' in capsys.readouterr().out


def test_cli_has_no_generic_optimizer_parameter():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            (
            "validate",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--optimizer-param",
            "mutation_probability=0.3",
            )
        )


def test_cli_has_no_seed_genome_parameter():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            (
                "validate",
                str(ROOT / "experiments" / "smoke.yaml"),
                "--seed_genome",
                '{"critic":{"adapter":1.0}}',
            )
        )


def test_cli_selects_two_search_granularities(capsys):
    code = main(
        (
            "validate",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--algorithm",
            "two_stage_nsga2",
            "--search_granularity",
            "global",
            "layer",
            "--populations",
            "8",
            "6",
            "--generations",
            "2",
            "3",
            "--num_layers",
            "4",
        )
    )
    assert code == 0
    output = capsys.readouterr().out
    assert '"granularities": [' in output
    assert '"layer"' in output


def test_cli_selects_de_and_its_direct_parameters(capsys):
    code = main(
        (
            "validate",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--algorithm",
            "de",
            "--search_granularity",
            "adapter",
            "--populations",
            "8",
            "--generations",
            "3",
            "--differential_weight",
            "0.6",
            "--neighborhood_size",
            "3",
            "--decomposition",
            "tchebycheff",
        )
    )

    assert code == 0
    output = capsys.readouterr().out
    assert '"algorithm": "de"' in output
    assert '"de_variant": "DE/current/1/bin"' in output
    assert '"multi_objective_strategy": "decomposition"' in output


def test_cli_defaults_two_stage_de_to_adapter_then_layer(capsys):
    code = main(
        (
            "validate",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--algorithm",
            "two_stage_de",
            "--populations",
            "6",
            "--generations",
            "1",
            "--num_layers",
            "2",
        )
    )

    assert code == 0
    output = capsys.readouterr().out
    assert '"granularities": [' in output
    assert '"adapter"' in output
    assert '"layer"' in output


def test_cli_validates_progressive_de_refinement_schedule(capsys):
    code = main(
        (
            "validate",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--algorithm",
            "progressive_de",
            "--refinement_groups",
            "1",
            "2",
            "3",
            "--populations",
            "4",
            "--generations",
            "2",
            "--num_layers",
            "3",
        )
    )

    assert code == 0
    output = capsys.readouterr().out
    assert '"algorithm": "progressive_de"' in output
    assert '"group_count": 1' in output
    assert '"group_count": 2' in output
    assert '"group_count": 3' in output
    assert '"adapter_pool_mode": "fixed_global_pool"' not in output
    assert '"initializer": "InheritedPopulationState"' in output


def test_cli_rejects_removed_progressive_transition_heuristics():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            (
                "validate",
                str(ROOT / "experiments" / "smoke.yaml"),
                "--transition_jitter",
                "0.1",
            )
        )


def test_smoke_device_is_not_treated_as_multi_gpu_override():
    args = build_parser().parse_args(
        (
            "smoke-real",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--device",
            "10",
        )
    )

    assert args.smoke_device == "10"
    assert not hasattr(args, "device")


def test_replay_device_is_not_treated_as_multi_gpu_override():
    args = build_parser().parse_args(
        (
            "replay-candidate",
            str(ROOT / "experiments" / "smoke.yaml"),
            "--candidate",
            "candidate.json",
            "--device",
            "3",
            "--report",
            "report.json",
        )
    )

    assert args.replay_device == "3"
    assert not hasattr(args, "device")


def test_auto_device_is_resolved_without_mixing_explicit_ids(monkeypatch):
    monkeypatch.setattr(
        "agentmerge.core.config._discover_cuda_device_ids",
        lambda: ("1", "2", "4", "5"),
    )
    assert resolve_device_ids(("auto",)) == ("1", "2", "4", "5")
    with pytest.raises(Exception, match="cannot be combined"):
        resolve_device_ids(("auto", "1"))
