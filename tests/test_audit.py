from types import SimpleNamespace

from agentmerge.backends.audit import PeftAdapterPoolAuditor, _snapshot_peft_config
from agentmerge.core.config import AdapterConfig


class _EnumLike:
    def __init__(self, value):
        self.value = value


class _FakePeftConfig:
    base_model_name_or_path = "Qwen/Qwen3-8B"
    peft_type = _EnumLike("LORA")
    task_type = _EnumLike("CAUSAL_LM")
    r = 16
    target_modules = {"q_proj", "v_proj"}
    modules_to_save = None
    rank_pattern = {}
    use_dora = False
    bias = "none"
    lora_bias = False
    revision = "abc123"


def test_snapshot_peft_config_is_json_serializable_shape():
    snapshot = _snapshot_peft_config(_FakePeftConfig())
    assert snapshot["peft_type"] == "LORA"
    assert snapshot["task_type"] == "CAUSAL_LM"
    assert snapshot["rank"] == 16
    assert snapshot["target_modules"] == ("q_proj", "v_proj")
    assert snapshot["rank_pattern"] == {}
    assert snapshot["use_dora"] is False
    assert snapshot["bias"] == "none"


def test_auditor_rejects_every_materializer_incompatibility():
    config = SimpleNamespace(
        model=SimpleNamespace(
            base_model_identity="Qwen/Qwen3-8B",
            base_model_aliases=("Qwen/Qwen3-8B",),
        )
    )
    adapter = AdapterConfig(
        "bad",
        "bad",
        rank=16,
        metadata={"license_audited": True, "data_audited": True},
    )
    snapshot = {
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "rank": 16,
        "rank_pattern": {"q_proj": 8},
        "use_dora": True,
        "modules_to_save": ("lm_head",),
        "bias": "all",
        "lora_bias": True,
    }
    findings = []

    PeftAdapterPoolAuditor()._audit_one(config, adapter, snapshot, findings)

    assert {finding.code for finding in findings} == {
        "rank_pattern_unsupported",
        "dora_unsupported",
        "modules_to_save_unsupported",
        "adapter_bias_unsupported",
    }
    assert all(finding.severity == "error" for finding in findings)


def test_auditor_rejects_heterogeneous_target_modules():
    findings = []
    resolved = {
        "a": {"target_modules": ("q_proj", "v_proj")},
        "b": {"target_modules": ("q_proj",)},
    }

    PeftAdapterPoolAuditor()._audit_pool(resolved, findings)

    assert [(item.code, item.severity) for item in findings] == [
        ("heterogeneous_target_modules", "error")
    ]
