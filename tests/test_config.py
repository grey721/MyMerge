from pathlib import Path

import pytest

from agentmerge.core.config import (
    ConfigError,
    apply_experiment_overrides,
    experiment_config_from_dict,
    load_experiment_config,
)
from agentmerge.optimizers.nsga2 import NSGA2Config


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_valid():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    assert config.model.name_or_path == "Qwen/Qwen3-8B"
    assert config.agent_names == ("planner", "solver", "critic")
    assert len(config.adapter_names) == 5
    assert config.search.algorithm == "nsga2"
    assert config.search.granularities == ("global",)
    assert config.search.options == {}


def test_tool_loop_topology_is_parsed_and_validated():
    raw = {
        "model": {"backend": "mock", "name_or_path": "mock"},
        "adapters": [{"name": "a", "path": "a"}],
        "agents": [
            {"name": "planner", "role": "planner"},
            {"name": "arguments", "role": "argument generator"},
            {"name": "final", "role": "final answerer"},
        ],
        "topology": {
            "loop_agents": ["planner", "arguments"],
            "max_steps": 2,
            "max_context_tokens": 2048,
            "output_agent": "final",
        },
        "benchmark": {"kind": "synthetic"},
        "objectives": [{"metric": "quality", "direction": "maximize"}],
    }

    config = experiment_config_from_dict(raw)
    config.validate()

    assert config.topology.kind == "tool_loop"
    assert config.topology.loop_agents == ("planner", "arguments")
    assert config.topology.max_steps == 2
    assert config.topology.max_context_tokens == 2048
    serialized = config.to_experiment_dict()["topology"]
    assert "edges" not in serialized
    assert serialized["loop_agents"] == ("planner", "arguments")


def test_tool_loop_rejects_dag_edges():
    raw = {
        "model": {"backend": "mock", "name_or_path": "mock"},
        "adapters": [{"name": "a", "path": "a"}],
        "agents": [
            {"name": "planner", "role": "planner"},
            {"name": "arguments", "role": "argument generator"},
            {"name": "final", "role": "final answerer"},
        ],
        "topology": {
            "kind": "tool_loop",
            "edges": [["planner", "arguments"]],
            "loop_agents": ["planner", "arguments"],
            "output_agent": "final",
        },
        "benchmark": {"kind": "synthetic"},
        "objectives": [{"metric": "quality", "direction": "maximize"}],
    }

    with pytest.raises(ConfigError, match="loop_agents instead of edges"):
        experiment_config_from_dict(raw).validate()


def test_shared_local_model_path_keeps_hub_identity(tmp_path):
    source = (ROOT / "experiments" / "smoke.yaml").read_text(encoding="utf-8")
    source = source.replace(
        "  name_or_path: Qwen/Qwen3-8B\n",
        "  name_or_path: ../../shared-models/Qwen3-8B\n"
        "  base_model_id: Qwen/Qwen3-8B\n",
        1,
    )
    path = tmp_path / "shared-model.yaml"
    path.write_text(source, encoding="utf-8")

    config = load_experiment_config(str(path))
    assert config.model.name_or_path.endswith("/shared-models/Qwen3-8B")
    assert config.model.base_model_identity == "Qwen/Qwen3-8B"


def test_experiment_yaml_rejects_search_configuration(tmp_path):
    source = (ROOT / "experiments" / "smoke.yaml").read_text(encoding="utf-8")
    source += "\nsearch:\n  algorithm: nsga2\n"
    path = tmp_path / "invalid-optimizer-options.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigError, match="Unknown key.*search"):
        load_experiment_config(str(path))


def test_scoped_granularity_is_resolved_from_cli():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    config = apply_experiment_overrides(
        config,
        search_granularity=("layer_group",),
        num_layers=36,
        group_size=6,
    )
    assert config.search.granularities == ("layer_group",)
    assert config.search.granularity_options == {
        "num_layers": 36,
        "group_size": 6,
    }


def test_unknown_model_field_is_rejected(tmp_path):
    source = (ROOT / "experiments" / "smoke.yaml").read_text(encoding="utf-8")
    source = source.replace(
        "  backend: mock\n",
        "  backend: mock\n  unexpected_setting: true\n",
        1,
    )
    path = tmp_path / "unknown-model-field.yaml"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ConfigError, match="Unknown key.*unexpected_setting"):
        load_experiment_config(str(path))


def test_stage_budget_count_must_match_granularities():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))
    with pytest.raises(ConfigError, match="exactly 2 stage values"):
        apply_experiment_overrides(
            config,
            search_granularity=("global", "layer"),
            generations=(1, 2, 3),
        )


def test_cycle_is_rejected(tmp_path):
    source = (ROOT / "experiments" / "smoke.yaml").read_text(encoding="utf-8")
    source = source.replace(
        "    - [solver, critic]\n",
        "    - [solver, critic]\n    - [critic, planner]\n",
    )
    path = tmp_path / "cycle.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match="acyclic"):
        load_experiment_config(str(path))


def test_cli_overrides_only_change_run_controls(tmp_path):
    original = load_experiment_config(
        str(ROOT / "experiments" / "smoke.yaml")
    )
    resolved = apply_experiment_overrides(
        original,
        run_name="trial-7",
        output_dir=str(tmp_path),
        seed=7,
        algorithm="nsga2",
        search_granularity=("global", "layer"),
        populations=(8, 6),
        generations=(2, 3),
        workers=3,
        batch_size=4,
        device=("0", "2"),
        mutation_probability=0.33,
    )

    assert resolved.name == "trial-7"
    assert resolved.seed == resolved.search.seed == 7
    assert resolved.model.generation.seed == 7
    assert resolved.search.granularities == ("global", "layer")
    assert resolved.search.stage_population_sizes == (8, 6)
    assert resolved.search.stage_generations == (2, 3)
    assert resolved.search.total_generation_batches == 5
    assert resolved.search.final_generation_index == 4
    assert resolved.search.workers == 3
    assert resolved.runtime.mode == "multi_gpu"
    assert resolved.runtime.gpu_ids == ("0", "2")
    assert resolved.benchmark.options["inference_batch_size"] == 4
    assert resolved.benchmark.inference_batch_size == 4
    assert resolved.search.options["nsga2"]["mutation_probability"] == 0.33
    assert resolved.agents == original.agents
    assert resolved.topology == original.topology
    assert resolved.adapters == original.adapters


def test_cli_can_replace_objectives_without_changing_benchmark():
    original = load_experiment_config(
        str(ROOT / "experiments" / "qwen3_vllm_research_v01.yaml")
    )

    resolved = apply_experiment_overrides(
        original,
        objectives=(
            "task_quality.math500:max",
            "task_quality.hotpotqa:maximize",
            "fusion_rank_proxy:min",
        ),
    )

    assert tuple(
        (item.metric, item.direction) for item in resolved.objectives
    ) == (
        ("task_quality.math500", "maximize"),
        ("task_quality.hotpotqa", "maximize"),
        ("fusion_rank_proxy", "minimize"),
    )
    assert resolved.benchmark == original.benchmark


def test_cli_objective_requires_explicit_direction():
    config = load_experiment_config(str(ROOT / "experiments" / "smoke.yaml"))

    with pytest.raises(ConfigError, match="METRIC:DIRECTION"):
        apply_experiment_overrides(config, objectives=("quality",))


def test_vllm_rejects_independent_max_num_seqs():
    raw = {
        "model": {
            "backend": "vllm",
            "name_or_path": "Qwen/Qwen3-8B",
            "options": {"vllm": {"max_num_seqs": 32}},
        },
        "adapters": [{"name": "a", "path": "/tmp/a"}],
        "agents": [{"name": "agent", "role": "agent"}],
        "topology": {
            "kind": "fixed",
            "edges": [],
            "output_agent": "agent",
        },
        "benchmark": {
            "kind": "jsonl",
            "path": "/tmp/data.jsonl",
            "options": {"inference_batch_size": 8},
        },
        "objectives": [{"metric": "quality", "direction": "maximize"}],
    }

    with pytest.raises(ConfigError, match="derived.*inference_batch_size"):
        experiment_config_from_dict(raw).validate()


def test_vllm_factory_derives_sequence_capacity_from_benchmark(monkeypatch):
    config = load_experiment_config(
        str(ROOT / "experiments" / "qwen3_vllm_research_v01.yaml")
    )
    config = apply_experiment_overrides(config, batch_size=16)
    captured = {}

    def fake_backend(model, adapters, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "agentmerge.backends.factory.VllmCausalLMBackend", fake_backend
    )
    from agentmerge.backends.factory import create_model_backend

    create_model_backend(config)

    assert captured["inference_batch_size"] == 16
    assert "max_num_seqs" not in config.model.options["vllm"]


def test_experiment_yaml_dictionary_excludes_search_request():
    config = load_experiment_config(
        str(ROOT / "experiments" / "smoke.yaml")
    )

    assert "search" not in config.to_experiment_dict()
    assert "model" in config.to_experiment_dict()
    assert "agents" in config.to_experiment_dict()
    assert "loop_agents" not in config.to_experiment_dict()["topology"]
    assert "max_steps" not in config.to_experiment_dict()["topology"]
    assert "max_context_tokens" not in (
        config.to_experiment_dict()["topology"]
    )
