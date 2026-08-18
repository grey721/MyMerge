import json

import pytest

from agentmerge.backends.algebra import FactorLinearFusionAlgebra
from agentmerge.backends.resources import FusedAdapterResource, FusionAdapterFactory
from agentmerge.backends.vllm.backend import VllmCandidateFusion, VllmCausalLMBackend
from agentmerge.backends.vllm.materialization import (
    VllmResidentAdapterPool,
    VllmTensorAdapterMaterializer,
    supported_rank_ceiling,
)
from agentmerge.backends.vllm.tensor_lora import protect_resident_source_tensors
from agentmerge.backends.vllm.transport import (
    VllmTensorLoraTransport,
    configure_vllm_engine_mode,
    normalize_engine_mode,
)
from agentmerge.core.config import AdapterConfig, ConfigError, experiment_config_from_dict
from agentmerge.core.domain import AdapterMixture
from agentmerge.core.policies import LayerGranularity, ScopedAdapterFusionPolicy


torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")


def _write_adapter(directory, name, rank, alpha, layer_values):
    directory.mkdir()
    config = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": ["q_proj"],
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    (directory / "adapter_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    tensors = {}
    for layer, value in enumerate(layer_values):
        prefix = "base_model.model.model.layers.{}.self_attn.q_proj".format(layer)
        tensors[prefix + ".lora_A.weight"] = torch.full(
            (rank, 2), float(value), dtype=torch.float32
        )
        tensors[prefix + ".lora_B.weight"] = torch.full(
            (2, rank), float(value + 10), dtype=torch.float32
        )
    safetensors.save_file(tensors, str(directory / "adapter_model.safetensors"))
    return AdapterConfig(
        name=name,
        path=str(directory),
        base_model="Qwen/Qwen3-8B",
        rank=rank,
        target_modules=("q_proj",),
    )


def _pool(tmp_path):
    first = _write_adapter(tmp_path / "first", "first", 1, 2, (1, 2))
    second = _write_adapter(tmp_path / "second", "second", 2, 2, (3, 4))
    return VllmResidentAdapterPool((first, second))


def test_tensor_materializer_preserves_scoped_fusion_math(tmp_path):
    pool = _pool(tmp_path)
    policy = ScopedAdapterFusionPolicy(
        adapter_names=("first", "second"),
        granularity=LayerGranularity(2),
        scope_weights=(
            ("layer:0", (1.0, 0.0)),
            ("layer:1", (0.25, 0.75)),
        ),
    )
    materializer = VllmTensorAdapterMaterializer(pool)
    fused = materializer.materialize(policy, "candidate")

    layer0 = "base_model.model.model.layers.0.self_attn.q_proj"
    layer1 = "base_model.model.model.layers.1.self_attn.q_proj"
    assert fused.rank == 3
    assert fused.peft_config["r"] == fused.peft_config["lora_alpha"] == 3
    assert fused.tensors[layer0 + ".lora_A.weight"].shape == (3, 2)
    # first scale=alpha/r=2; second is inactive in layer 0.
    assert torch.allclose(
        fused.tensors[layer0 + ".lora_B.weight"],
        torch.tensor([[22.0, 0.0, 0.0], [22.0, 0.0, 0.0]]),
    )
    # layer 1: first=(12 * 2 * .25), second=(14 * 1 * .75).
    assert torch.allclose(
        fused.tensors[layer1 + ".lora_B.weight"],
        torch.tensor([[6.0, 10.5, 10.5], [6.0, 10.5, 10.5]]),
    )
    profile = materializer.runtime_snapshot()
    assert profile["algebra"] == "exact_delta"
    assert profile["precompiled_module_count"] == 2
    assert profile["target_allocation_count"] == 4
    assert profile["last_materialization"]["zero_coefficient_slices"] == 1
    fused.release()
    pool.close()


def test_tensor_materializer_executes_factor_linear_fusion(tmp_path):
    pool = _pool(tmp_path)
    policy = ScopedAdapterFusionPolicy(
        adapter_names=("first", "second"),
        granularity=LayerGranularity(2),
        scope_weights=(
            ("layer:0", (1.0, 0.0)),
            ("layer:1", (0.25, 0.75)),
        ),
    )
    materializer = VllmTensorAdapterMaterializer(
        pool, algebra=FactorLinearFusionAlgebra()
    )
    fused = materializer.materialize(policy, "candidate")

    layer0 = "base_model.model.model.layers.0.self_attn.q_proj"
    layer1 = "base_model.model.model.layers.1.self_attn.q_proj"
    root_two = 2.0**0.5
    assert fused.rank == 2
    assert fused.tensors[layer0 + ".lora_A.weight"].shape == (2, 2)
    assert torch.allclose(
        fused.tensors[layer0 + ".lora_A.weight"],
        torch.tensor([[root_two, root_two], [0.0, 0.0]]),
    )
    assert torch.allclose(
        fused.tensors[layer0 + ".lora_B.weight"],
        torch.tensor([[11.0 * root_two, 0.0], [11.0 * root_two, 0.0]]),
    )
    assert torch.allclose(
        fused.tensors[layer1 + ".lora_A.weight"],
        torch.tensor(
            [
                [3.0 + 0.5 * root_two, 3.0 + 0.5 * root_two],
                [3.0, 3.0],
            ]
        ),
    )
    assert torch.allclose(
        fused.tensors[layer1 + ".lora_B.weight"],
        torch.tensor(
            [
                [10.5 + 3.0 * root_two, 10.5],
                [10.5 + 3.0 * root_two, 10.5],
            ]
        ),
    )
    profile = materializer.runtime_snapshot()
    assert profile["algebra"] == "factor_linear"
    assert profile["last_materialization"]["total_rank"] == 2
    fused.release()
    pool.close()


class _Request:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Engine:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_lora(self, request):
        self.added.append(request)
        return True

    def remove_lora(self, request_id):
        self.removed.append(request_id)
        return True


class _FailFirstRemovalEngine(_Engine):
    def __init__(self):
        super().__init__()
        self._failed = False

    def remove_lora(self, request_id):
        self.removed.append(request_id)
        if not self._failed:
            self._failed = True
            raise RuntimeError("injected vLLM removal failure")
        return True


def _bound_transport(engine, mode="v1"):
    transport = VllmTensorLoraTransport(mode, _Request)
    transport.bind_engine(engine)
    return transport


def test_candidate_lora_is_reused_then_released(tmp_path):
    pool = _pool(tmp_path)
    engine = _Engine()
    transport = _bound_transport(engine)
    pool.bind_transport(transport)
    fusion = VllmCandidateFusion(transport, pool)
    mixture = AdapterMixture(("first", "second"), (0.4, 0.6))

    with fusion.candidate_session():
        first_request, _ = fusion.request_for(mixture)
        second_request, _ = fusion.request_for(mixture)
        assert first_request is second_request
        assert len(engine.added) == 1
        assert fusion.runtime_snapshot()["resident_temporary_count"] == 1

    assert engine.removed == [first_request.lora_int_id]
    assert first_request.lora_tensors == {}
    assert fusion.runtime_snapshot()["resident_temporary_count"] == 0
    pool.close()


def test_vllm_factory_returns_one_independently_closable_fused_resource(tmp_path):
    pool = _pool(tmp_path)
    engine = _Engine()
    transport = _bound_transport(engine)
    pool.bind_transport(transport)
    fusion = VllmCandidateFusion(transport, pool)
    policy = AdapterMixture(("first", "second"), (0.4, 0.6))

    assert isinstance(fusion.factory, FusionAdapterFactory)
    resource = fusion.factory.create(policy)
    assert isinstance(resource, FusedAdapterResource)
    assert resource.source_keys == ("first", "second")
    assert resource.policy_fingerprint == policy.fingerprint()
    assert resource.resident_bytes > 0

    resource.close()
    resource.close()
    assert resource.closed is True
    assert len(engine.removed) == 1
    pool.close()


def test_vllm_failed_candidate_release_is_retained_and_retried(tmp_path):
    pool = _pool(tmp_path)
    engine = _FailFirstRemovalEngine()
    transport = _bound_transport(engine)
    pool.bind_transport(transport)
    fusion = VllmCandidateFusion(transport, pool)
    policy = AdapterMixture(("first", "second"), (0.4, 0.6))

    with pytest.raises(RuntimeError, match="injected vLLM removal failure"):
        with fusion.candidate_session():
            request, _ = fusion.request_for(policy)

    assert request.lora_tensors
    assert fusion.runtime_snapshot()["resident_temporary_count"] == 1

    with fusion.candidate_session():
        pass
    assert request.lora_tensors == {}
    assert fusion.runtime_snapshot()["resident_temporary_count"] == 0
    assert engine.removed == [request.lora_int_id, request.lora_int_id]
    pool.close()


def test_one_hot_source_lora_stays_registered_across_candidates(tmp_path):
    pool = _pool(tmp_path)
    engine = _Engine()
    transport = _bound_transport(engine)
    pool.bind_transport(transport)
    fusion = VllmCandidateFusion(transport, pool)
    one_hot = AdapterMixture(("first", "second"), (1.0, 0.0))

    with fusion.candidate_session():
        request, name = fusion.request_for(one_hot)
    with fusion.candidate_session():
        same_request, _ = fusion.request_for(one_hot)

    assert name == "first"
    assert request is same_request
    assert len(engine.added) == 1
    assert engine.removed == []
    fusion.close()
    pool.close()
    assert engine.removed == [request.lora_int_id]
    snapshot = pool.runtime_snapshot()
    assert snapshot["state"] == "closed"
    assert snapshot["resident_source_count"] == 0
    assert snapshot["unregistration_count"] == 1


def test_vllm_pool_releases_registered_zero_reference_source_by_key(tmp_path):
    pool = _pool(tmp_path)
    engine = _Engine()
    transport = _bound_transport(engine)
    pool.bind_transport(transport)
    second_request = pool.request_for_source("second")
    second_tensors = second_request.lora_tensors

    report = pool.release_unreferenced({"first": 2, "second": 0})

    assert report.released_adapter_keys == ("second",)
    assert report.released_coordinate_indices == (1,)
    assert report.resident_adapter_keys == ("first",)
    assert report.released_source_bytes > 0
    assert report.resident_source_bytes > 0
    assert engine.removed == [2]
    assert second_tensors == {}
    assert pool.sources[0].spec.name == "first"
    pool.validate_policy(AdapterMixture(("first", "second"), (1.0, 0.0)))
    with pytest.raises(RuntimeError, match="released source"):
        pool.validate_policy(AdapterMixture(("first", "second"), (0.0, 1.0)))

    pool.close()


def test_vllm_pool_can_append_sources_before_sealing(tmp_path):
    first = _write_adapter(tmp_path / "first", "first", 1, 2, (1, 2))
    second = _write_adapter(tmp_path / "second", "second", 2, 2, (3, 4))
    pool = VllmResidentAdapterPool((first,), auto_seal=False)

    pool.add_source(second)
    assert pool.coordinate_index("second") == 1
    pool.seal()
    assert pool.adapter_keys == ("first", "second")
    pool.close()


def test_vllm_batch_packing_counts_lora_ids_but_not_base_requests():
    backend = VllmCausalLMBackend.__new__(VllmCausalLMBackend)
    backend._max_loras = 1
    first = _Request(lora_int_id=1)
    second = _Request(lora_int_id=2)

    batches = backend._pack_lora_batches(
        ((None, "base"), (first, "a"), (first, "a"), (second, "b"))
    )

    assert batches == ((0, 1, 2), (3,))


def test_supported_rank_ceiling_matches_vllm_rank_set():
    assert supported_rank_ceiling(176) == 256
    assert supported_rank_ceiling(256) == 256
    with pytest.raises(ValueError, match="exceeds"):
        supported_rank_ceiling(513)


def test_native_scaling_cannot_mutate_resident_source_b_factors():
    source_a = torch.ones((2, 2))
    source_b = torch.full((2, 2), 3.0)
    tensors = {
        "layer.lora_A.weight": source_a,
        "layer.lora_B.weight": source_b,
    }

    protected = protect_resident_source_tensors(tensors, scaling=2.0)
    protected["layer.lora_B.weight"].mul_(2.0)

    assert protected["layer.lora_A.weight"] is source_a
    assert protected["layer.lora_B.weight"] is not source_b
    assert torch.all(source_b == 3.0)


def test_unit_scaling_keeps_candidate_zero_copy():
    tensors = {"layer.lora_B.weight": torch.ones((2, 2))}
    assert protect_resident_source_tensors(tensors, scaling=1.0) is tensors


def test_vllm_v1_transport_selects_inprocess_engine(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)

    assert configure_vllm_engine_mode("V1") == "v1"
    assert normalize_engine_mode("v0") == "v0"
    assert __import__("os").environ["VLLM_USE_V1"] == "1"
    assert __import__("os").environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    with pytest.raises(ValueError, match="engine_mode"):
        normalize_engine_mode("future")


def test_vllm_transport_owns_registration_boundary():
    engine = _Engine()
    transport = _bound_transport(engine, mode="v1")
    request = transport.make_request(
        name="candidate",
        request_id=7,
        peft_config={"r": 1},
        tensors={"weight": torch.ones(1)},
    )

    assert transport.register(request) is True
    assert transport.unregister(7) is True
    assert engine.added == [request]
    assert engine.removed == [7]
    assert transport.runtime_snapshot()["engine_mode"] == "V1"
    assert transport.runtime_snapshot()["process_boundary"] == "in_process"


def test_vllm_config_rejects_tensor_parallelism_inside_gpu_worker():
    raw = {
        "model": {
            "backend": "vllm",
            "name_or_path": "Qwen/Qwen3-8B",
            "options": {"vllm": {"tensor_parallel_size": 2}},
        },
        "adapters": [
            {"name": "a", "path": "/tmp/a", "base_model": "Qwen/Qwen3-8B"}
        ],
        "agents": [{"name": "agent", "role": "agent"}],
        "topology": {
            "kind": "fixed",
            "edges": [],
            "output_agent": "agent",
        },
        "benchmark": {"kind": "jsonl", "path": "/tmp/data.jsonl"},
        "objectives": [{"metric": "quality", "direction": "maximize"}],
    }
    config = experiment_config_from_dict(raw)
    with pytest.raises(ConfigError, match="tensor_parallel_size=1"):
        config.validate()


def test_vllm_config_rejects_unknown_engine_mode():
    raw = {
        "model": {
            "backend": "vllm",
            "name_or_path": "Qwen/Qwen3-8B",
            "options": {"vllm": {"engine_mode": "v2"}},
        },
        "adapters": [
            {"name": "a", "path": "/tmp/a", "base_model": "Qwen/Qwen3-8B"}
        ],
        "agents": [{"name": "agent", "role": "agent"}],
        "topology": {
            "kind": "fixed",
            "edges": [],
            "output_agent": "agent",
        },
        "benchmark": {"kind": "jsonl", "path": "/tmp/data.jsonl"},
        "objectives": [{"metric": "quality", "direction": "maximize"}],
    }
    config = experiment_config_from_dict(raw)

    with pytest.raises(ConfigError, match="engine_mode"):
        config.validate()


def test_vllm_config_reserves_source_and_candidate_lora_cache():
    raw = {
        "model": {
            "backend": "vllm",
            "name_or_path": "Qwen/Qwen3-8B",
            "options": {"vllm": {"max_cpu_loras": 2}},
        },
        "adapters": [
            {"name": "a", "path": "/tmp/a", "base_model": "Qwen/Qwen3-8B"},
            {"name": "b", "path": "/tmp/b", "base_model": "Qwen/Qwen3-8B"},
        ],
        "agents": [
            {"name": "first", "role": "first"},
            {"name": "second", "role": "second"},
        ],
        "topology": {
            "kind": "fixed",
            "edges": [["first", "second"]],
            "output_agent": "second",
        },
        "benchmark": {"kind": "jsonl", "path": "/tmp/data.jsonl"},
        "objectives": [{"metric": "quality", "direction": "maximize"}],
    }
    config = experiment_config_from_dict(raw)

    with pytest.raises(ConfigError, match="source pool 2.*Agent 2"):
        config.validate()
