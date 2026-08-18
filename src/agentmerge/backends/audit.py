"""Preflight checks for a global PEFT adapter pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..backends.peft import OptionalDependencyError
from ..core.config import AdapterConfig, ExperimentConfig


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    code: str
    message: str
    adapter: Optional[str] = None


@dataclass(frozen=True)
class AdapterPoolAudit:
    passed: bool
    findings: Tuple[AuditFinding, ...]
    resolved_adapters: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "findings": [
                {
                    "severity": finding.severity,
                    "code": finding.code,
                    "message": finding.message,
                    "adapter": finding.adapter,
                }
                for finding in self.findings
            ],
            "resolved_adapters": {
                name: dict(value)
                for name, value in self.resolved_adapters.items()
            },
        }


class PeftAdapterPoolAuditor:
    """Load only PEFT configs; no base-model weights are allocated."""

    def audit(self, config: ExperimentConfig) -> AdapterPoolAudit:
        try:
            from peft import PeftConfig
        except ImportError as exc:
            raise OptionalDependencyError(
                "Adapter audit requires PEFT. Run it in the Python 3.10 model "
                "environment after: pip install -e '.[peft]'"
            ) from exc

        findings: List[AuditFinding] = []
        resolved: Dict[str, Mapping[str, Any]] = {}
        for adapter in config.adapters:
            try:
                peft_config = PeftConfig.from_pretrained(adapter.path)
            except Exception as exc:
                findings.append(
                    AuditFinding(
                        "error",
                        "config_load_failed",
                        "{}: {}".format(type(exc).__name__, exc),
                        adapter.name,
                    )
                )
                continue
            snapshot = _snapshot_peft_config(peft_config)
            resolved[adapter.name] = snapshot
            self._audit_one(config, adapter, snapshot, findings)

        self._audit_pool(resolved, findings)
        passed = not any(
            finding.severity == "error" for finding in findings
        )
        return AdapterPoolAudit(
            passed=passed,
            findings=tuple(findings),
            resolved_adapters=resolved,
        )

    def _audit_one(
        self,
        config: ExperimentConfig,
        adapter: AdapterConfig,
        snapshot: Mapping[str, Any],
        findings: List[AuditFinding],
    ) -> None:
        actual_base = snapshot.get("base_model_name_or_path")
        expected_base = config.model.base_model_identity
        if actual_base not in config.model.base_model_aliases:
            findings.append(
                AuditFinding(
                    "error",
                    "base_model_mismatch",
                    "PEFT config declares {!r}; expected {!r}".format(
                        actual_base, expected_base
                    ),
                    adapter.name,
                )
            )
        if str(snapshot.get("peft_type", "")).upper() != "LORA":
            findings.append(
                AuditFinding(
                    "error",
                    "not_lora",
                    "Expected PEFT type LORA, got {!r}".format(
                        snapshot.get("peft_type")
                    ),
                    adapter.name,
                )
            )
        if str(snapshot.get("task_type", "")).upper() != "CAUSAL_LM":
            findings.append(
                AuditFinding(
                    "error",
                    "not_causal_lm",
                    "Expected task type CAUSAL_LM, got {!r}".format(
                        snapshot.get("task_type")
                    ),
                    adapter.name,
                )
            )
        actual_rank = snapshot.get("rank")
        if adapter.rank is not None and actual_rank != adapter.rank:
            findings.append(
                AuditFinding(
                    "error",
                    "rank_mismatch",
                    "Configured rank {} but PEFT config has {}".format(
                        adapter.rank, actual_rank
                    ),
                    adapter.name,
                )
            )
        if snapshot.get("rank_pattern"):
            findings.append(
                AuditFinding(
                    "error",
                    "rank_pattern_unsupported",
                    "Candidate-scoped materialization requires a uniform rank",
                    adapter.name,
                )
            )
        if bool(snapshot.get("use_dora", False)):
            findings.append(
                AuditFinding(
                    "error",
                    "dora_unsupported",
                    "Candidate-scoped materialization does not support DoRA",
                    adapter.name,
                )
            )
        if snapshot.get("modules_to_save"):
            findings.append(
                AuditFinding(
                    "error",
                    "modules_to_save_unsupported",
                    "Candidate-scoped materialization does not support "
                    "modules_to_save",
                    adapter.name,
                )
            )
        if str(snapshot.get("bias", "none")).lower() != "none" or bool(
            snapshot.get("lora_bias", False)
        ):
            findings.append(
                AuditFinding(
                    "error",
                    "adapter_bias_unsupported",
                    "Candidate-scoped materialization supports factor-only "
                    "LoRA adapters with bias='none'",
                    adapter.name,
                )
            )
        metadata = adapter.metadata
        if not bool(metadata.get("license_audited", False)):
            findings.append(
                AuditFinding(
                    "warning",
                    "license_not_audited",
                    "Adapter license has not been marked as audited",
                    adapter.name,
                )
            )
        if not bool(metadata.get("data_audited", False)):
            findings.append(
                AuditFinding(
                    "warning",
                    "training_data_not_audited",
                    "Training data has not been checked for benchmark leakage",
                    adapter.name,
                )
            )

    def _audit_pool(
        self,
        resolved: Mapping[str, Mapping[str, Any]],
        findings: List[AuditFinding],
    ) -> None:
        target_sets = {
            name: tuple(sorted(value.get("target_modules") or ()))
            for name, value in resolved.items()
        }
        unique_targets = set(target_sets.values())
        if len(unique_targets) > 1:
            findings.append(
                AuditFinding(
                    "error",
                    "heterogeneous_target_modules",
                    "Candidate-scoped materialization requires every source "
                    "adapter to target the same module set",
                )
            )


def _snapshot_peft_config(peft_config) -> Mapping[str, Any]:
    def enum_value(value):
        return getattr(value, "value", value)

    target_modules = getattr(peft_config, "target_modules", None)
    if isinstance(target_modules, str):
        targets = (target_modules,)
    else:
        targets = tuple(sorted(target_modules or ()))
    return {
        "base_model_name_or_path": getattr(
            peft_config, "base_model_name_or_path", None
        ),
        "peft_type": enum_value(getattr(peft_config, "peft_type", None)),
        "task_type": enum_value(getattr(peft_config, "task_type", None)),
        "rank": getattr(peft_config, "r", None),
        "target_modules": targets,
        "modules_to_save": tuple(
            sorted(getattr(peft_config, "modules_to_save", None) or ())
        ),
        "rank_pattern": dict(getattr(peft_config, "rank_pattern", None) or {}),
        "use_dora": bool(getattr(peft_config, "use_dora", False)),
        "bias": str(getattr(peft_config, "bias", "none")),
        "lora_bias": bool(getattr(peft_config, "lora_bias", False)),
        "revision": getattr(peft_config, "revision", None),
    }
