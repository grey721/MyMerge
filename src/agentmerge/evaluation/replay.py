"""Deterministic replay of a persisted candidate on one local backend."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..backends import ModelBackend, create_model_backend
from ..core.config import ExperimentConfig
from ..evaluation.benchmarks import (
    JsonlBenchmark,
    SystemBenchmarkEvaluator,
    build_scorer,
)
from ..evaluation.system import create_agent_system
from ..optimizers.genome import FusionGenome
from ..optimizers.translation import GlobalFusionPlanTranslator
from ..runtime.lifecycle import graceful_termination_signals, parent_watch_enabled


@dataclass(frozen=True)
class ReplayError:
    phase: str
    error_type: str
    message: str
    traceback: str


@dataclass(frozen=True)
class CandidateReplayAttempt:
    attempt: int
    passed: bool
    elapsed_seconds: float
    metrics: Mapping[str, float]
    statistics: Mapping[str, Any]
    runtime: Mapping[str, Any]
    cuda_before: Mapping[str, Any]
    cuda_after: Mapping[str, Any]
    backend_after: Mapping[str, Any]
    error: Optional[ReplayError] = None


@dataclass(frozen=True)
class CandidateReplayReport:
    schema_version: int
    passed: bool
    candidate_path: str
    genome_id: str
    backend: str
    visible_device: str
    gpu_memory_utilization: Optional[float]
    repeats_requested: int
    successful_repeats: int
    backend_load_seconds: float
    total_seconds: float
    cuda_before_load: Mapping[str, Any]
    cuda_after_load: Mapping[str, Any]
    cuda_after_close: Mapping[str, Any]
    backend_after_load: Mapping[str, Any]
    attempts: Tuple[CandidateReplayAttempt, ...]
    error: Optional[ReplayError] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CandidateReplayRunner:
    """Replay one global Agent × Adapter chromosome without an optimizer.

    A fresh backend is owned by each runner. The same candidate is evaluated
    repeatedly in that backend so temporary LoRA creation, registration,
    reuse and release follow exactly the production evaluator lifecycle.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        visible_device: str,
        gpu_memory_utilization: Optional[float] = None,
        backend_factory: Optional[Callable[[], ModelBackend]] = None,
    ) -> None:
        if config.benchmark.kind.lower() != "jsonl":
            raise ValueError("Candidate replay requires a JSONL benchmark")
        if gpu_memory_utilization is not None and not (
            0.0 < float(gpu_memory_utilization) <= 1.0
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        self.config = _local_replay_config(config, gpu_memory_utilization)
        self.visible_device = str(visible_device)
        self.gpu_memory_utilization = (
            None
            if gpu_memory_utilization is None
            else float(gpu_memory_utilization)
        )
        self._backend_factory = backend_factory or self._create_backend

    def run(self, candidate_path: str, repeats: int = 1) -> CandidateReplayReport:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        started = time.perf_counter()
        path = Path(candidate_path).expanduser().resolve()
        payload = _load_candidate(path)
        genome_id = str(payload.get("genome_id", ""))
        plan = self._plan_from_payload(payload)

        backend = None  # type: Optional[ModelBackend]
        evaluator = None  # type: Optional[SystemBenchmarkEvaluator]
        attempts = []  # type: List[CandidateReplayAttempt]
        overall_error = None  # type: Optional[ReplayError]
        load_seconds = 0.0
        cuda_before_load = _cuda_memory_snapshot()
        cuda_after_load = {}  # type: Mapping[str, Any]
        cuda_after_close = {}  # type: Mapping[str, Any]
        backend_after_load = {}  # type: Mapping[str, Any]

        with graceful_termination_signals(
            watch_parent=parent_watch_enabled()
        ):
            try:
                print(
                    "Replay | Backend load | Started | device={} | "
                    "gpu_memory_utilization={}".format(
                        self.visible_device,
                        self.gpu_memory_utilization,
                    ),
                    flush=True,
                )
                load_started = time.perf_counter()
                backend = self._backend_factory()
                load_seconds = time.perf_counter() - load_started
                cuda_after_load = _cuda_memory_snapshot()
                backend_after_load = dict(backend.runtime_snapshot())
                evaluator = self._create_evaluator(backend)
                print(
                    "Replay | Backend load | Completed | {:.3f}s".format(
                        load_seconds
                    ),
                    flush=True,
                )

                for attempt_index in range(1, repeats + 1):
                    print(
                        "Replay | Attempt {}/{} | Started".format(
                            attempt_index, repeats
                        ),
                        flush=True,
                    )
                    attempt = self._evaluate_once(
                        evaluator, plan, attempt_index
                    )
                    attempts.append(attempt)
                    status = "Completed" if attempt.passed else "Failed"
                    print(
                        "Replay | Attempt {}/{} | {} | {:.3f}s".format(
                            attempt_index,
                            repeats,
                            status,
                            attempt.elapsed_seconds,
                        ),
                        flush=True,
                    )
                    if not attempt.passed:
                        break
            except Exception as exc:
                overall_error = _capture_error("backend_load", exc)
            finally:
                if evaluator is not None:
                    try:
                        evaluator.close()
                    except Exception as exc:
                        if overall_error is None:
                            overall_error = _capture_error(
                                "evaluator_close", exc
                            )
                if backend is not None:
                    try:
                        backend.close()
                    except Exception as exc:
                        if overall_error is None:
                            overall_error = _capture_error("backend_close", exc)
                cuda_after_close = _cuda_memory_snapshot()

        successful = sum(attempt.passed for attempt in attempts)
        passed = successful == repeats and overall_error is None
        return CandidateReplayReport(
            schema_version=1,
            passed=passed,
            candidate_path=str(path),
            genome_id=genome_id,
            backend=self.config.model.backend,
            visible_device=self.visible_device,
            gpu_memory_utilization=self.gpu_memory_utilization,
            repeats_requested=repeats,
            successful_repeats=successful,
            backend_load_seconds=load_seconds,
            total_seconds=time.perf_counter() - started,
            cuda_before_load=cuda_before_load,
            cuda_after_load=cuda_after_load,
            cuda_after_close=cuda_after_close,
            backend_after_load=backend_after_load,
            attempts=tuple(attempts),
            error=overall_error,
        )

    def _evaluate_once(
        self,
        evaluator: SystemBenchmarkEvaluator,
        plan: Any,
        attempt_index: int,
    ) -> CandidateReplayAttempt:
        _reset_cuda_peak_memory()
        cuda_before = _cuda_memory_snapshot()
        started = time.perf_counter()
        try:
            result = evaluator.evaluate(plan)
            elapsed = time.perf_counter() - started
            return CandidateReplayAttempt(
                attempt=attempt_index,
                passed=True,
                elapsed_seconds=elapsed,
                metrics={
                    str(key): float(value)
                    for key, value in result.metrics.items()
                },
                statistics=dict(result.details.get("statistics", {})),
                runtime=dict(result.details.get("_runtime", {})),
                cuda_before=cuda_before,
                cuda_after=_cuda_memory_snapshot(),
                backend_after=dict(evaluator.system.backend.runtime_snapshot()),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return CandidateReplayAttempt(
                attempt=attempt_index,
                passed=False,
                elapsed_seconds=elapsed,
                metrics={},
                statistics={},
                runtime={},
                cuda_before=cuda_before,
                cuda_after=_cuda_memory_snapshot(),
                backend_after=_safe_backend_snapshot(evaluator.system.backend),
                error=_capture_error("evaluation", exc),
            )

    def _plan_from_payload(self, payload: Mapping[str, Any]):
        genotype = payload.get("genotype", payload)
        if not isinstance(genotype, Mapping):
            raise ValueError("Candidate genotype must be a mapping")
        if any(isinstance(value, Mapping) for value in genotype.values()):
            raise ValueError(
                "This replay command currently accepts global Agent × Adapter "
                "genotypes; the selected diagnostic candidate must come from "
                "the adapter-level stage"
            )
        if set(genotype) != set(self.config.agent_names):
            raise ValueError(
                "Candidate agents {} do not match configured agents {}".format(
                    sorted(genotype), sorted(self.config.agent_names)
                )
            )
        genome = FusionGenome.from_dict(genotype)
        return GlobalFusionPlanTranslator(
            self.config.agent_names, self.config.adapter_names
        ).plan_for(genome)

    def _create_backend(self) -> ModelBackend:
        if self.config.model.backend.lower() not in ("peft", "vllm"):
            raise ValueError("Candidate replay requires a PEFT or vLLM backend")
        return create_model_backend(self.config)

    def _create_evaluator(
        self, backend: ModelBackend
    ) -> SystemBenchmarkEvaluator:
        system = create_agent_system(self.config, backend)
        benchmark = JsonlBenchmark(
            self.config.benchmark, build_scorer(self.config.benchmark)
        )
        return SystemBenchmarkEvaluator(system, benchmark)


def _local_replay_config(
    config: ExperimentConfig,
    gpu_memory_utilization: Optional[float],
) -> ExperimentConfig:
    options = dict(config.model.options)
    if gpu_memory_utilization is not None:
        vllm_options = dict(options.get("vllm", {}))
        # Diagnostic utilization sweeps must not be shadowed by a fixed cache
        # setting in the production experiment configuration.
        vllm_options.pop("kv_cache_memory_bytes", None)
        vllm_options["gpu_memory_utilization"] = float(
            gpu_memory_utilization
        )
        options["vllm"] = vllm_options
    resolved = replace(
        config,
        model=replace(config.model, options=options),
        runtime=replace(config.runtime, mode="local", gpu_ids=()),
    )
    resolved.validate()
    return resolved


def _load_candidate(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError("Candidate file does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Candidate file must contain a JSON object")
    return payload


def _cuda_memory_snapshot() -> Mapping[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "available": True,
            "logical_device": int(device),
            "device_name": torch.cuda.get_device_name(device),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "snapshot_error": "{}: {}".format(type(exc).__name__, exc),
        }


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    except Exception:
        pass


def _safe_backend_snapshot(backend: ModelBackend) -> Mapping[str, Any]:
    try:
        return dict(backend.runtime_snapshot())
    except Exception as exc:
        return {
            "snapshot_error": "{}: {}".format(type(exc).__name__, exc)
        }


def _capture_error(phase: str, exc: BaseException) -> ReplayError:
    return ReplayError(
        phase=phase,
        error_type=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
    )


__all__ = [
    "CandidateReplayAttempt",
    "CandidateReplayReport",
    "CandidateReplayRunner",
    "ReplayError",
]
