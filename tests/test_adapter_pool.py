import pytest

from agentmerge.backends.peft_pool import PeftResidentAdapterPool
from agentmerge.backends.pool import AdapterPool, AdapterPoolState
from agentmerge.backends.mock import MockModelBackend
from agentmerge.core.config import AdapterConfig
from agentmerge.core.domain import AdapterMixture


def _adapters():
    return (
        AdapterConfig("a", "/a", rank=2),
        AdapterConfig("b", "/b", rank=4),
    )


def test_logical_adapter_pool_owns_order_lookup_and_idempotent_lifecycle():
    pool = AdapterPool(_adapters())
    policy = AdapterMixture(("a", "b"), (0.25, 0.75))

    pool.validate_policy(policy)
    assert pool.adapter_names == ("a", "b")
    assert pool.adapter_keys == ("a", "b")
    assert pool.coordinate_index("a") == 0
    assert pool.adapter_key(1) == "b"
    assert pool.get("a").coordinate_index == 0
    assert pool.spec("b").rank == 4
    assert pool.runtime_snapshot()["resource_kind"] == "logical"
    with pytest.raises(KeyError, match="Unknown Adapter key"):
        pool.get(0)

    pool.close()
    pool.close()
    snapshot = pool.runtime_snapshot()
    assert snapshot["state"] == "closed"
    assert snapshot["close_count"] == 1


class _Config:
    target_modules = {"q_proj"}
    rank_pattern = {}
    alpha_pattern = {}
    use_rslora = False

    def __init__(self, rank, alpha):
        self.r = rank
        self.lora_alpha = alpha


class _Weight:
    device = "cpu"


class _Embeddings:
    weight = _Weight()


class _PeftModel:
    def __init__(self):
        self.peft_config = {
            "a": _Config(2, 4),
            "b": _Config(4, 4),
            "temporary": _Config(6, 6),
        }
        self.active = None
        self.deleted = []

    def get_input_embeddings(self):
        return _Embeddings()

    def modules(self):
        return ()

    def set_adapter(self, name):
        self.active = name

    def delete_adapter(self, name):
        self.deleted.append(name)
        self.peft_config.pop(name)


def test_peft_resident_pool_reports_resources_and_cleans_stray_temporary():
    model = _PeftModel()
    pool = PeftResidentAdapterPool(_adapters(), model)

    snapshot = pool.runtime_snapshot()
    assert snapshot["resource_kind"] == "peft_model"
    assert snapshot["resident_source_count"] == 2
    assert snapshot["device"] == "cpu"
    assert pool.descriptor("a").native_scale == 2.0
    assert pool.temporary_adapter_names == ("temporary",)

    pool.close()

    assert model.active == "a"
    assert model.deleted == ["temporary", "a", "b"]
    assert pool.runtime_snapshot()["resident_source_count"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = pool.model


def test_adapter_pool_rejects_wrong_policy_axis():
    pool = AdapterPool(_adapters())
    wrong = AdapterMixture(("b", "a"), (0.5, 0.5))

    with pytest.raises(ValueError, match="does not match"):
        pool.validate_policy(wrong)


def test_peft_pool_releases_only_zero_reference_id_and_keeps_global_axis():
    model = _PeftModel()
    model.peft_config.pop("temporary")
    pool = PeftResidentAdapterPool(_adapters(), model)

    report = pool.release_unreferenced({"a": 2, "b": 0})

    assert report.released_adapter_keys == ("b",)
    assert report.released_coordinate_indices == (1,)
    assert report.resident_adapter_keys == ("a",)
    assert pool.state == AdapterPoolState.PRUNED
    assert model.deleted == ["b"]
    # Policies keep the stable global axis; a released zero coordinate is fine.
    pool.validate_policy(AdapterMixture(("a", "b"), (1.0, 0.0)))
    with pytest.raises(RuntimeError, match="released source"):
        pool.validate_policy(AdapterMixture(("a", "b"), (0.0, 1.0)))

    pool.close()
    assert model.deleted == ["b", "a"]


def test_pool_building_add_seal_and_single_release_are_explicit():
    pool = AdapterPool((_adapters()[0],), auto_seal=False)

    assert pool.state == AdapterPoolState.BUILDING
    added = pool.add_source(_adapters()[1])
    assert added.adapter_key == "b"
    assert added.coordinate_index == 1
    pool.seal()
    assert pool.state == AdapterPoolState.SEALED
    with pytest.raises(RuntimeError, match="expected building"):
        pool.add_source(AdapterConfig("c", "/c", rank=8))

    report = pool.release("b")
    assert report.released_adapter_keys == ("b",)
    assert report.released_coordinate_indices == (1,)
    assert pool.adapter_keys == ("a", "b")
    with pytest.raises(KeyError, match="not resident"):
        pool.get("b")


def test_backend_resident_pool_can_build_then_freeze_coordinates():
    model = _PeftModel()
    model.peft_config.pop("temporary")
    pool = PeftResidentAdapterPool(
        (_adapters()[0],), model, auto_seal=False
    )

    pool.add_source(_adapters()[1])
    assert pool.state == AdapterPoolState.BUILDING
    assert pool.coordinate_index("b") == 1
    pool.seal()
    assert pool.state == AdapterPoolState.SEALED
    pool.close()


def test_model_backend_exposes_the_unified_pool_interface():
    backend = MockModelBackend(_adapters())

    assert backend.adapter_pool.adapter_names == ("a", "b")
    assert backend.runtime_snapshot()["adapter_pool"]["state"] == "sealed"
    backend.close()
    assert backend.adapter_pool.closed is True
