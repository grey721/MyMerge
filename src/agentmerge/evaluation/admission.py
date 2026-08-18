"""Deterministic base and one-hot adapter admission evaluation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import yaml

from ..backends import ModelBackend
from ..core.config import ExperimentConfig
from ..core.domain import AdapterMixture, EvaluationResult
from ..core.logger import ExperimentLogger
from ..core.policies import FusionPlan
from .benchmarks import FusionPlanEvaluator
from ..runtime.experiment import (
    ComponentFactory,
    assert_output_safe,
    write_output_manifest,
)


@dataclass(frozen=True)
class AdmissionVariant:
    name: str
    kind: str
    adapter: Optional[str]
    agent: Optional[str]
    plan: FusionPlan


@dataclass(frozen=True)
class AdmissionSummary:
    run_dir: str
    variant_count: int
    base_quality: float
    best_variant: str


class AdapterAdmissionRunner:
    """Evaluate fixed genomes without invoking an evolutionary optimizer."""

    def __init__(
        self,
        config: ExperimentConfig,
        factory: Optional[ComponentFactory] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.factory = factory or ComponentFactory()

    def run(self, include_rolewise: bool = False) -> AdmissionSummary:
        run_dir = self._create_run_dir()
        run_logger = ExperimentLogger(
            run_dir,
            self.config.name,
            generation_width=len(
                str(self.config.search.final_generation_index)
            ),
        )
        try:
            variants = self._variants(include_rolewise)
            run_logger.event(
                "admission_started",
                {
                    "run_dir": str(run_dir),
                    "variant_count": len(variants),
                    "role_wise": include_rolewise,
                    "devices": list(self.config.runtime.gpu_ids),
                },
            )
            backend = None  # type: Optional[ModelBackend]
            evaluator = None  # type: Optional[FusionPlanEvaluator]
            try:
                loads_base_model = (
                    self.config.benchmark.kind.lower() != "synthetic"
                    and self.config.model.backend.lower() in ("peft", "vllm")
                )
                loading_started_at = None
                if loads_base_model:
                    loading_started_at = time.perf_counter()
                    loading_payload = {
                        "backend": self.config.model.backend,
                        "devices": list(self.config.runtime.gpu_ids),
                    }
                    if self.config.model.backend.lower() == "vllm":
                        vllm_options = self.config.model.options.get("vllm", {})
                        if isinstance(vllm_options, Mapping):
                            loading_payload["engine_mode"] = str(
                                vllm_options.get("engine_mode", "v1")
                            ).upper()
                    run_logger.event(
                        "base_model_loading_started",
                        loading_payload,
                    )
                if (
                    self.config.benchmark.kind.lower() != "synthetic"
                    and self.config.runtime.mode == "local"
                ):
                    backend = self.factory.create_backend(self.config)
                evaluator = self.factory.create_evaluator(self.config, backend)
                _atomic_write_yaml(
                    run_dir / "resolved_config.yaml",
                    self.config.to_experiment_dict(),
                )
                write_output_manifest(run_dir, self.config)
                runtime = evaluator.runtime_metadata()
                _atomic_write_json(run_dir / "runtime.json", runtime)
                if loads_base_model and loading_started_at is not None:
                    workers = runtime.get("workers")
                    run_logger.event(
                        "base_model_loading_completed",
                        {
                            "elapsed_seconds": time.perf_counter()
                            - loading_started_at,
                            "worker_count": (
                                len(workers)
                                if isinstance(workers, list)
                                else 1
                            ),
                            "evaluator": runtime.get("evaluator"),
                        },
                    )
                run_logger.event("evaluator_ready", dict(runtime))
                evaluations = evaluator.evaluate_many(
                    tuple(item.plan for item in variants),
                    workers=self.config.search.workers,
                )
            finally:
                if evaluator is not None:
                    evaluator.close()
                if backend is not None:
                    backend.close()

            report = self._report(variants, evaluations)
            assert_output_safe(report)
            _atomic_write_json(run_dir / "admission_results.json", report)
            quality_index = report["metric_order"].index("quality")
            best = max(
                report["variants"],
                key=lambda item: float(item["metric_values"][quality_index]),
            )
            summary = AdmissionSummary(
                run_dir=str(run_dir),
                variant_count=len(variants),
                base_quality=float(
                    report["variants"][report["base_variant_index"]][
                        "metric_values"
                    ][quality_index]
                ),
                best_variant=str(best["name"]),
            )
            run_logger.event(
                "admission_completed",
                {
                    "base_quality": summary.base_quality,
                    "best_variant": summary.best_variant,
                },
            )
            return summary
        except Exception:
            run_logger.exception("Adapter admission terminated unexpectedly")
            raise
        finally:
            run_logger.close()

    def _variants(
        self,
        include_rolewise: bool,
    ) -> Tuple[AdmissionVariant, ...]:
        adapter_count = len(self.config.adapter_names)
        zeros = (0.0,) * adapter_count
        variants = [
            AdmissionVariant(
                name="base",
                kind="base",
                adapter=None,
                agent=None,
                plan=self._plan_for_vectors(
                    {name: zeros for name in self.config.agent_names}
                ),
            )
        ]  # type: List[AdmissionVariant]
        for adapter_index, adapter_name in enumerate(self.config.adapter_names):
            one_hot = tuple(
                1.0 if index == adapter_index else 0.0
                for index in range(adapter_count)
            )
            variants.append(
                AdmissionVariant(
                    name="global:{}".format(adapter_name),
                    kind="global_one_hot",
                    adapter=adapter_name,
                    agent=None,
                    plan=self._plan_for_vectors(
                        {name: one_hot for name in self.config.agent_names}
                    ),
                )
            )
            if include_rolewise:
                for selected_agent in self.config.agent_names:
                    variants.append(
                        AdmissionVariant(
                            name="role:{}:{}".format(
                                selected_agent, adapter_name
                            ),
                            kind="role_one_hot",
                            adapter=adapter_name,
                            agent=selected_agent,
                            plan=self._plan_for_vectors(
                                {
                                    name: one_hot if name == selected_agent else zeros
                                    for name in self.config.agent_names
                                }
                            ),
                        )
                    )
        return tuple(variants)

    def _plan_for_vectors(
        self, vectors: Mapping[str, Sequence[float]]
    ) -> FusionPlan:
        return FusionPlan(
            tuple(
                (
                    agent_name,
                    AdapterMixture(
                        self.config.adapter_names,
                        tuple(float(value) for value in vectors[agent_name]),
                    ),
                )
                for agent_name in self.config.agent_names
            )
        )

    @staticmethod
    def _report(
        variants: Sequence[AdmissionVariant],
        evaluations: Sequence[EvaluationResult],
    ) -> Mapping[str, Any]:
        if len(variants) != len(evaluations):
            raise RuntimeError("Admission evaluator returned a wrong result count")
        metric_order = tuple(evaluations[0].metrics.keys())
        base_metrics = dict(evaluations[0].metrics)
        base_quality = float(base_metrics.get("quality", 0.0))
        rows = []
        for variant, evaluation in zip(variants, evaluations):
            metrics = dict(evaluation.metrics)
            rows.append(
                {
                    "name": variant.name,
                    "kind": variant.kind,
                    "adapter_index": (
                        None
                        if variant.adapter is None
                        else list(variant.plan.policy_for(
                            variant.plan.agent_names[0]
                        ).adapter_names).index(variant.adapter)
                    ),
                    "agent_index": (
                        None
                        if variant.agent is None
                        else list(variant.plan.agent_names).index(variant.agent)
                    ),
                    "genotype": _genotype_matrix(variant.plan),
                    "metric_values": [metrics[name] for name in metric_order],
                    "quality_delta_vs_base": (
                        float(metrics.get("quality", 0.0)) - base_quality
                    ),
                    "statistics": dict(
                        evaluation.details.get("statistics", {})
                    ),
                    "runtime": _compact_runtime(
                        evaluation.details.get("_runtime", {})
                    ),
                }
            )
        return {
            "schema_version": 2,
            "axes": {
                "agents": list(variants[0].plan.agent_names),
                "adapters": list(
                    variants[0].plan.policy_for(
                        variants[0].plan.agent_names[0]
                    ).adapter_names
                ),
            },
            "metric_order": list(metric_order),
            "variant_count": len(rows),
            "base_variant_index": 0,
            "variants": rows,
        }

    def _create_run_dir(self) -> Path:
        root = Path(self.config.output_dir).expanduser().resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = root / "{}-one-hot-admission-{}".format(
            _slug(self.config.name), timestamp
        )
        suffix = 1
        while candidate.exists():
            candidate = root / "{}-one-hot-admission-{}-{}".format(
                _slug(self.config.name), timestamp, suffix
            )
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate


def _atomic_write_json(path: Path, payload: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_yaml(path: Path, payload: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                payload,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _slug(value: str) -> str:
    cleaned = "".join(
        character.lower()
        if character.isalnum() or character in ("-", "_")
        else "-"
        for character in value.strip()
    )
    return cleaned.strip("-") or "experiment"


def _genotype_matrix(plan: FusionPlan) -> List[List[float]]:
    """Encode one-hot admission genes against the manifest axes."""

    return [
        [float(value) for value in plan.policy_for(agent_name).mixture_for().weights]
        for agent_name in plan.agent_names
    ]


def _compact_runtime(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    allowed = {
        "gpu_id",
        "worker_index",
        "process_id",
        "scheduler_attempts",
        "candidate_session_seconds",
        "model_latency_seconds_per_case",
        "fusion",
        "evaluation_seconds",
    }
    return {
        key: value for key, value in runtime.items() if key in allowed
    }


__all__ = ["AdapterAdmissionRunner", "AdmissionSummary", "AdmissionVariant"]
