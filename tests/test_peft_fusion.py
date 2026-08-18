import pytest

from agentmerge.backends.algebra import FactorLinearFusionAlgebra
from agentmerge.backends.materialization import (
    AdapterMaterializer,
    PeftCandidateFusion,
    PeftDirectAdapterMaterializer,
    PeftFusionAdapterFactory,
)
from agentmerge.backends.peft_pool import PeftResidentAdapterPool
from agentmerge.backends.pool import AdapterPool
from agentmerge.backends.resources import FusedAdapterResource, FusionAdapterFactory
from agentmerge.core.config import AdapterConfig
from agentmerge.core.domain import AdapterMixture
from agentmerge.core.policies import (
    LayerGranularity,
    ScopedAdapterFusionPolicy,
    TransformerModuleLocator,
)


class _Model:
    def __init__(self):
        self.final_adapter = None

    def set_adapter(self, name):
        self.final_adapter = name


class _RecordingMaterializer(AdapterMaterializer):
    def __init__(self):
        self.materialized = []
        self.released = []

    def materialize(self, model, mixture, adapter_name):
        self.materialized.append((mixture.fingerprint(), adapter_name))
        return adapter_name

    def release(self, model, adapter_name):
        self.released.append(adapter_name)


class _FailFirstReleaseMaterializer(_RecordingMaterializer):
    def __init__(self):
        super().__init__()
        self._failed = False

    def release(self, model, adapter_name):
        self.released.append(adapter_name)
        if not self._failed:
            self._failed = True
            raise RuntimeError("injected release failure")


def _adapters():
    return (
        AdapterConfig("a", "a", rank=8),
        AdapterConfig("b", "b", rank=16),
    )


def test_candidate_fusion_reuses_every_mixture_until_candidate_exit():
    materializer = _RecordingMaterializer()
    strategy = PeftCandidateFusion(
        AdapterPool(_adapters()), materializer=materializer
    )
    model = _Model()
    first = AdapterMixture(("a", "b"), (0.25, 0.75))
    second = AdapterMixture(("a", "b"), (0.5, 0.5))

    with strategy.candidate_session(model):
        with strategy.activate(model, first) as first_name:
            pass
        with strategy.activate(model, second) as second_name:
            pass
        with strategy.activate(model, first) as reused_name:
            assert reused_name == first_name

        assert second_name != first_name
        assert len(materializer.materialized) == 2
        assert materializer.released == []
        assert strategy.resident_adapter_names == (first_name, second_name)

    assert materializer.released == [first_name, second_name]
    assert strategy.resident_adapter_names == ()


def test_peft_factory_returns_one_independently_closable_fused_resource():
    materializer = _RecordingMaterializer()
    model = _Model()
    factory = PeftFusionAdapterFactory(model, materializer)
    policy = AdapterMixture(("a", "b"), (0.25, 0.75))

    assert isinstance(factory, FusionAdapterFactory)
    resource = factory.create(policy)
    assert isinstance(resource, FusedAdapterResource)
    assert resource.source_keys == ("a", "b")
    assert resource.policy_fingerprint == policy.fingerprint()
    assert resource.closed is False

    resource.close()
    resource.close()
    assert resource.closed is True
    assert materializer.released == [resource.adapter_key]


def test_mixed_adapter_fusion_requires_candidate_session():
    materializer = _RecordingMaterializer()
    strategy = PeftCandidateFusion(
        AdapterPool(_adapters()), materializer=materializer
    )
    mixture = AdapterMixture(("a", "b"), (0.25, 0.75))

    with pytest.raises(RuntimeError, match="candidate_session"):
        with strategy.activate(_Model(), mixture):
            pass

    assert materializer.materialized == []
    assert materializer.released == []


def test_one_hot_activation_uses_resident_source_adapter_without_materializing():
    materializer = _RecordingMaterializer()
    strategy = PeftCandidateFusion(
        AdapterPool(_adapters()), materializer=materializer
    )
    model = _Model()
    mixture = AdapterMixture(("a", "b"), (1.0, 0.0))

    with strategy.activate(model, mixture) as adapter_name:
        assert adapter_name == "a"
        assert model.final_adapter == "a"

    assert materializer.materialized == []


def test_candidate_cleanup_attempts_every_adapter_and_recovers_next_session():
    materializer = _FailFirstReleaseMaterializer()
    strategy = PeftCandidateFusion(
        AdapterPool(_adapters()), materializer=materializer
    )
    model = _Model()
    first = AdapterMixture(("a", "b"), (0.25, 0.75))
    second = AdapterMixture(("a", "b"), (0.5, 0.5))

    with pytest.raises(RuntimeError, match="injected release failure"):
        with strategy.candidate_session(model):
            with strategy.activate(model, first):
                pass
            with strategy.activate(model, second):
                pass

    assert len(materializer.released) == 2
    snapshot = strategy.runtime_snapshot()
    assert snapshot["resident_temporary_count"] == 1
    assert len(snapshot["unreleased_adapter_names"]) == 1

    with strategy.candidate_session(model):
        pass
    assert strategy.resident_adapter_names == ()
    assert len(materializer.released) == 3


class _FakePeftConfig:
    r = 2
    lora_alpha = 2
    target_modules = {"q_proj"}
    rank_pattern = {}
    alpha_pattern = {}
    use_dora = False
    use_rslora = False
    modules_to_save = None
    bias = "none"
    lora_bias = False
    lora_dropout = 0.0
    init_lora_weights = False


class _RegistrationModel:
    def __init__(self):
        self.peft_config = {"a": _FakePeftConfig(), "b": _FakePeftConfig()}
        self.deleted = []

    def add_adapter(self, adapter_name, peft_config, **kwargs):
        self.peft_config[adapter_name] = peft_config

    def delete_adapter(self, adapter_name):
        self.deleted.append(adapter_name)
        self.peft_config.pop(adapter_name, None)


class _FailingAllocationMaterializer(PeftDirectAdapterMaterializer):
    @staticmethod
    def _materialize_empty_adapter_weights(model, active_names, temporary_name):
        raise RuntimeError("injected allocation failure")


def test_materialization_failure_removes_partially_registered_adapter():
    materializer = _FailingAllocationMaterializer()
    model = _RegistrationModel()
    mixture = AdapterMixture(("a", "b"), (0.25, 0.75))

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        materializer.materialize(model, mixture, "temporary")

    assert "temporary" not in model.peft_config
    assert model.deleted == ["temporary"]
    assert materializer.runtime_snapshot()["release_count"] == 1


def test_real_peft_candidate_fusion_matches_peft_cat_reference():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(1)
    base_model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    first_config = peft.LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )
    second_config = peft.LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )
    model = peft.get_peft_model(base_model, first_config, adapter_name="x")
    model.add_adapter("y", second_config)
    model.eval()

    adapters = (
        AdapterConfig("x", "x", rank=2),
        AdapterConfig("y", "y", rank=4),
    )
    mixture = AdapterMixture(("x", "y"), (0.25, 0.75))
    pool = PeftResidentAdapterPool(adapters, model)
    strategy = PeftCandidateFusion(pool)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        with strategy.candidate_session(model):
            with strategy.activate(model, mixture):
                candidate_logits = model(input_ids).logits

        model.add_weighted_adapter(
            adapters=["x", "y"],
            weights=[0.25, 0.75],
            adapter_name="test_reference_cat",
            combination_type="cat",
        )
        model.set_adapter("test_reference_cat")
        reference_logits = model(input_ids).logits

    profile = strategy.runtime_snapshot()["materializer"]
    assert profile["materialization_count"] == 1
    assert profile["release_count"] == 1
    assert profile["last_materialization"]["total_rank"] == 6
    assert profile["last_materialization"]["copied_module_count"] > 0
    assert profile["last_materialization"]["factor_copy_operations"] > 0
    assert profile["factor_bytes_copied"] > 0
    assert strategy.runtime_snapshot()["resident_temporary_count"] == 0
    assert torch.allclose(
        candidate_logits, reference_logits, rtol=1e-5, atol=1e-6
    )
    model.set_adapter("x")
    model.delete_adapter("test_reference_cat")


def test_real_peft_scoped_candidate_fusion_bakes_each_layer_coefficients():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(2)
    base_model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    first_config = peft.LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )
    second_config = peft.LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )
    model = peft.get_peft_model(base_model, first_config, adapter_name="x")
    model.add_adapter("y", second_config)
    model.eval()

    adapters = (
        AdapterConfig("x", "x", rank=2),
        AdapterConfig("y", "y", rank=4),
    )
    policy = ScopedAdapterFusionPolicy(
        adapter_names=("x", "y"),
        granularity=LayerGranularity(2),
        scope_weights=(
            ("layer:0", (1.0, 0.0)),
            ("layer:1", (0.0, 1.0)),
        ),
    )
    pool = PeftResidentAdapterPool(adapters, model)
    strategy = PeftCandidateFusion(pool)

    with strategy.candidate_session(model):
        with strategy.activate(model, policy) as temporary_name:
            checked_modules = _assert_factor_layout(
                torch, model, policy, temporary_name, ("x", "y")
            )

    assert checked_modules > 0
    assert strategy.estimate_rank(policy) == 6.0


def test_real_peft_factor_linear_fusion_accumulates_shared_rank_slices():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(3)
    base_model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    first_config = peft.LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )
    second_config = peft.LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        init_lora_weights=False,
    )
    model = peft.get_peft_model(base_model, first_config, adapter_name="x")
    model.add_adapter("y", second_config)
    model.eval()

    adapters = (
        AdapterConfig("x", "x", rank=2),
        AdapterConfig("y", "y", rank=4),
    )
    policy = AdapterMixture(("x", "y"), (0.25, 0.75))
    pool = PeftResidentAdapterPool(adapters, model)
    strategy = PeftCandidateFusion(pool, algebra=FactorLinearFusionAlgebra())

    checked_modules = 0
    with strategy.candidate_session(model):
        with strategy.activate(model, policy) as temporary_name:
            for module in model.modules():
                lora_a = getattr(module, "lora_A", None)
                lora_b = getattr(module, "lora_B", None)
                if (
                    lora_a is None
                    or lora_b is None
                    or temporary_name not in lora_a
                    or temporary_name not in lora_b
                ):
                    continue
                source_x_a = lora_a["x"].weight
                source_y_a = lora_a["y"].weight
                source_x_b = lora_b["x"].weight
                source_y_b = lora_b["y"].weight
                target_a = lora_a[temporary_name].weight
                target_b = lora_b[temporary_name].weight
                expected_a = source_y_a * 0.75
                expected_a[:2].add_(source_x_a, alpha=0.25)
                expected_b = source_y_b * 0.75
                expected_b[:, :2].add_(source_x_b, alpha=0.25)
                assert target_a.shape[0] == target_b.shape[1] == 4
                assert torch.allclose(target_a, expected_a)
                assert torch.allclose(target_b, expected_b)
                assert module.scaling[temporary_name] == 1.0
                checked_modules += 1

    assert checked_modules > 0
    assert strategy.estimate_rank(policy) == 4.0
    assert strategy.runtime_snapshot()["materializer"]["algebra"] == "factor_linear"


def _assert_factor_layout(torch, model, policy, temporary_name, source_names):
    locator = TransformerModuleLocator()
    checked = 0
    for module_path, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if (
            lora_a is None
            or lora_b is None
            or temporary_name not in lora_a
            or temporary_name not in lora_b
        ):
            continue
        weights = policy.mixture_for(locator.locate(module_path, module)).as_dict()
        target_a = lora_a[temporary_name].weight
        target_b = lora_b[temporary_name].weight
        offset = 0
        for source_name in source_names:
            source_a = lora_a[source_name].weight
            source_b = lora_b[source_name].weight
            rank = source_a.shape[0]
            end = offset + rank
            native_scale = module.scaling[source_name]
            assert torch.allclose(target_a[offset:end], source_a)
            assert torch.allclose(
                target_b[:, offset:end],
                source_b * (weights.get(source_name, 0.0) * native_scale),
            )
            offset = end
        assert offset == target_a.shape[0] == target_b.shape[1]
        assert module.scaling[temporary_name] == 1.0
        checked += 1
    return checked
