"""Experiment assembly, lifecycle management, and durable search outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from ..evaluation.system import create_agent_system
from ..backends import ModelBackend, create_model_backend
from ..core.config import ExperimentConfig
from ..core.logger import ExperimentLogger
from ..evaluation.benchmarks import (
    FusionPlanEvaluator,
    JsonlBenchmark,
    SyntheticMultiObjectiveEvaluator,
    SystemBenchmarkEvaluator,
    build_scorer,
)
from ..evaluation.metrics import (
    BenchmarkMetricCatalog,
    SYNTHETIC_METRIC_NAMES,
)
from ..optimizers.base import (
    Optimizer,
    SearchObserver,
    SearchResult,
)
from ..optimizers.fitness import FitnessSpace
from ..optimizers.registry import (
    OptimizerBuildContext,
    OptimizerRegistry,
    create_default_optimizer_registry,
)
from .lifecycle import graceful_termination_signals, parent_watch_enabled


@dataclass(frozen=True)
class ExperimentSummary:
    run_dir: str
    pareto_size: int
    unique_evaluations: int
    generations_completed: int
    objective_names: Tuple[str, ...]


class ComponentFactory:
    """Construction boundary that projects can subclass or replace."""

    def __init__(
        self,
        optimizer_registry: Optional[OptimizerRegistry] = None,
    ) -> None:
        self.optimizer_registry = (
            optimizer_registry or create_default_optimizer_registry()
        )

    def validate_search(self, config: ExperimentConfig) -> Mapping[str, object]:
        description = dict(
            self.optimizer_registry.validate(config.search.algorithm, config)
        )
        metric_names = self.available_metric_names(config)
        FitnessSpace(config.objectives).validate_metrics(metric_names)
        description["available_objective_metrics"] = list(metric_names)
        return description

    def available_metric_names(
        self, config: ExperimentConfig
    ) -> Tuple[str, ...]:
        kind = config.benchmark.kind.lower()
        if kind == "synthetic":
            return SYNTHETIC_METRIC_NAMES
        if kind == "jsonl":
            benchmark = JsonlBenchmark(
                config.benchmark, build_scorer(config.benchmark)
            )
            return BenchmarkMetricCatalog.from_groups(
                tuple(case.group for case in benchmark.cases),
                quality_domains=config.benchmark.options.get(
                    "quality_domains", {}
                ),
            ).names
        raise ValueError("Unknown benchmark kind {!r}".format(kind))

    def create_backend(self, config: ExperimentConfig) -> ModelBackend:
        return create_model_backend(config)

    def create_evaluator(
        self,
        config: ExperimentConfig,
        backend: Optional[ModelBackend] = None,
    ) -> FusionPlanEvaluator:
        kind = config.benchmark.kind.lower()
        if kind == "synthetic":
            return SyntheticMultiObjectiveEvaluator(
                agent_names=config.agent_names,
                adapter_names=config.adapter_names,
                seed=config.seed,
            )
        if kind == "jsonl":
            if config.runtime.mode == "multi_gpu":
                if backend is not None:
                    raise ValueError(
                        "Multi-GPU evaluator owns its worker-local backends"
                    )
                from ..runtime.distributed import MultiGpuFusionPlanEvaluator

                return MultiGpuFusionPlanEvaluator(config)
            if backend is None:
                raise ValueError("JSONL evaluation requires a model backend")
            system = create_agent_system(config, backend)
            benchmark = JsonlBenchmark(
                config.benchmark, build_scorer(config.benchmark)
            )
            return SystemBenchmarkEvaluator(system, benchmark)
        raise ValueError("Unknown benchmark kind {!r}".format(kind))

    def create_optimizer(
        self,
        config: ExperimentConfig,
        evaluator: FusionPlanEvaluator,
        observer: SearchObserver,
    ) -> Optimizer:
        context = OptimizerBuildContext(
            evaluator=evaluator,
            observer=observer,
        )
        return self.optimizer_registry.create(
            config.search.algorithm, config, context
        )


class JsonCheckpointObserver(SearchObserver):
    def __init__(
        self,
        run_dir: Path,
        fitness_space: FitnessSpace,
        checkpoint_every: int,
        logger: Optional[ExperimentLogger] = None,
        agent_names: Sequence[str] = (),
        genotype_encoding: str = "named",
    ) -> None:
        self.run_dir = run_dir
        self.fitness_space = fitness_space
        self.checkpoint_every = max(1, checkpoint_every)
        self.logger = logger
        self.agent_names = tuple(agent_names)
        self.genotype_encoding = genotype_encoding
        self.history_path = run_dir / "history.jsonl"
        self.events_path = run_dir / "events.jsonl"
        self.evaluations_dir = run_dir / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        self._event_lock = threading.Lock()
        self._event_sequence = 0
        self._started_at = time.perf_counter()

    def on_evaluation(
        self, individual: Any, cached: bool = False
    ) -> None:
        if not individual.evaluated:
            raise ValueError("Cannot persist an unevaluated candidate")
        genome_id = individual.genome.fingerprint()
        path = self.evaluations_dir / "{}.json".format(genome_id)
        if path.exists():
            return
        _atomic_write_json(
            path,
            evaluation_to_dict(
                individual,
                self.fitness_space,
                agent_names=self.agent_names,
                genotype_encoding=self.genotype_encoding,
            ),
        )

    def on_event(
        self, event_type: str, payload: Mapping[str, Any] = None
    ) -> None:
        event_payload = dict(payload or {})
        with self._event_lock:
            self._event_sequence += 1
            record = {
                "sequence": self._event_sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - self._started_at,
                "event": event_type,
                "data": event_payload,
            }
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self.logger is not None:
                self.logger.event(event_type, event_payload)
            if event_type == "resolution_trajectory":
                _atomic_write_json(
                    self.run_dir / "resolution_trajectory.json",
                    {
                        "schema_version": 1,
                        "stages": list(event_payload.get("stages", ())),
                    },
                )

    def on_generation(
        self,
        generation: int,
        population: Sequence[Any],
        unique_evaluations: int,
    ) -> None:
        record = {
            "generation": generation,
            "unique_evaluations": unique_evaluations,
            "pareto_size": sum(
                1 for individual in population if individual.rank == 0
            ),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if generation % self.checkpoint_every != 0:
            return
        payload = {
            "schema_version": 2,
            "generation": generation,
            "unique_evaluations": unique_evaluations,
            "population": [
                individual_reference_to_dict(individual)
                for individual in population
            ],
        }
        checkpoint_path = (
            self.run_dir
            / "checkpoints"
            / "generation_{:04d}.json".format(generation)
        )
        _atomic_write_json(checkpoint_path, payload)
        self.on_event(
            "checkpoint_written",
            {
                "generation": generation,
                "path": str(checkpoint_path.relative_to(self.run_dir)),
                "population_size": len(population),
            },
        )


class ExperimentRunner:
    """Own the complete experiment lifecycle while exposing replaceable factories."""

    def __init__(
        self,
        config: ExperimentConfig,
        factory: Optional[ComponentFactory] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.factory = factory or ComponentFactory()

    def run(self) -> ExperimentSummary:
        with graceful_termination_signals(
            watch_parent=parent_watch_enabled()
        ):
            return self._run_guarded()

    def _run_guarded(self) -> ExperimentSummary:
        run_dir = self._create_run_dir()
        run_logger = ExperimentLogger(
            run_dir,
            self.config.name,
            generation_width=len(
                str(self.config.search.final_generation_index)
            ),
        )
        _atomic_write_yaml(
            run_dir / "resolved_config.yaml",
            self.config.to_experiment_dict(),
        )
        write_output_manifest(run_dir, self.config)
        fitness_space = FitnessSpace(self.config.objectives)
        observer = JsonCheckpointObserver(
            run_dir=run_dir,
            fitness_space=fitness_space,
            checkpoint_every=self.config.search.checkpoint_every,
            logger=run_logger,
            agent_names=self.config.agent_names,
            genotype_encoding=_genotype_encoding(self.config),
        )
        adaptive_options = self.config.search.options.get(
            "adaptive_resolution_de", {}
        )
        if not isinstance(adaptive_options, Mapping):
            adaptive_options = {}
        observer.on_event(
            "run_started",
            {
                "experiment": self.config.name,
                "algorithm": self.config.search.algorithm,
                "run_dir": str(run_dir),
                "runtime_mode": self.config.runtime.mode,
                "devices": list(self.config.runtime.gpu_ids),
                "granularities": list(self.config.search.granularities),
                "populations": [
                    self.config.search.population_size_for(index)
                    for index in range(self.config.search.stage_count)
                ],
                "generations": (
                    ["adaptive_real_fe_budget"]
                    if self.config.search.algorithm
                    == "adaptive_resolution_de"
                    else [
                        self.config.search.generations_for(index)
                        for index in range(self.config.search.stage_count)
                    ]
                ),
                "total_budget": adaptive_options.get("total_budget"),
                "min_stage_fe": adaptive_options.get("min_stage_fe"),
                "optimizer_options": dict(self.config.search.options),
            },
        )

        backend = None  # type: Optional[ModelBackend]
        evaluator = None  # type: Optional[FusionPlanEvaluator]
        interrupted = False
        try:
            search_description = self.factory.validate_search(self.config)
            observer.on_event("search_validated", dict(search_description))
            observer.on_event(
                "evaluator_initializing",
                {
                    "benchmark": self.config.benchmark.kind,
                    "backend": self.config.model.backend,
                },
            )
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
                observer.on_event(
                    "base_model_loading_started",
                    loading_payload,
                )
            if (
                self.config.benchmark.kind.lower() != "synthetic"
                and self.config.runtime.mode == "local"
            ):
                backend = self.factory.create_backend(self.config)
            evaluator = self.factory.create_evaluator(self.config, backend)
            runtime = dict(evaluator.runtime_metadata())
            _atomic_write_json(run_dir / "runtime.json", runtime)
            if loads_base_model and loading_started_at is not None:
                workers = runtime.get("workers")
                observer.on_event(
                    "base_model_loading_completed",
                    {
                        "elapsed_seconds": time.perf_counter()
                        - loading_started_at,
                        "worker_count": (
                            len(workers) if isinstance(workers, list) else 1
                        ),
                        "evaluator": runtime.get("evaluator"),
                    },
                )
            observer.on_event("evaluator_ready", runtime)
            optimizer = self.factory.create_optimizer(
                self.config, evaluator, observer
            )
            observer.on_event(
                "optimizer_ready", {"type": type(optimizer).__name__}
            )
            result = optimizer.optimize()
            self._write_result(run_dir, result)
            observer.on_event("results_written", {"run_dir": str(run_dir)})
            self._log_objective_bests(observer, result, fitness_space)
            observer.on_event(
                "run_completed",
                {
                    "generations_completed": result.generations_completed,
                    "unique_evaluations": result.unique_evaluations,
                    "pareto_size": len(result.pareto_front),
                },
            )
        except Exception as exc:
            observer.on_event(
                "run_failed",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
            run_logger.exception("Experiment terminated unexpectedly")
            raise
        except BaseException:
            interrupted = True
            raise
        finally:
            if evaluator is not None:
                if interrupted:
                    evaluator.abort()
                else:
                    evaluator.close()
            if backend is not None:
                backend.close()
            run_logger.info("resources_closed")
            run_logger.close()
        return ExperimentSummary(
            run_dir=str(run_dir),
            pareto_size=len(result.pareto_front),
            unique_evaluations=result.unique_evaluations,
            generations_completed=result.generations_completed,
            objective_names=fitness_space.metric_names,
        )

    @staticmethod
    def _log_objective_bests(
        observer: SearchObserver,
        result: SearchResult,
        fitness_space: FitnessSpace,
    ) -> None:
        candidates = result.pareto_front or result.population
        for payload in fitness_space.objective_best_payloads(candidates):
            observer.on_event(
                "objective_best", dict(payload, scope="search")
            )

    def _create_run_dir(self) -> Path:
        base = Path(self.config.output_dir).expanduser().resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = base / "{}-{}".format(_slug(self.config.name), timestamp)
        suffix = 1
        while candidate.exists():
            candidate = base / "{}-{}-{}".format(
                _slug(self.config.name), timestamp, suffix
            )
            suffix += 1
        (candidate / "checkpoints").mkdir(parents=True, exist_ok=False)
        return candidate

    def _write_result(
        self,
        run_dir: Path,
        result: SearchResult,
    ) -> None:
        _atomic_write_json(
            run_dir / "pareto_front.json",
            {
                "schema_version": 3,
                "generations_completed": result.generations_completed,
                "unique_evaluations": result.unique_evaluations,
                "evaluation_store": "evaluations",
                "individuals": [
                    individual_reference_to_dict(
                        individual,
                        evaluation_ref_prefix="evaluations",
                    )
                    for individual in result.pareto_front
                ],
            },
        )
        _atomic_write_json(
            run_dir / "final_population.json",
            {
                "schema_version": 3,
                "evaluation_store": "evaluations",
                "individuals": [
                    individual_reference_to_dict(
                        individual,
                        evaluation_ref_prefix="evaluations",
                    )
                    for individual in result.population
                ]
            },
        )


def evaluation_to_dict(
    individual: Any,
    fitness_space: FitnessSpace,
    *,
    agent_names: Sequence[str] = (),
    genotype_encoding: str = "named",
) -> Dict[str, Any]:
    """Serialize immutable candidate data once, without generation state."""

    if individual.fitness is None or individual.evaluation is None:
        raise ValueError("Cannot serialize an unevaluated individual")
    payload = {
        "schema_version": 3,
        "genome_id": individual.genome.fingerprint(),
        "genotype": _compact_genotype(
            individual.genome.to_dict(), agent_names, genotype_encoding
        ),
        "metrics": dict(individual.evaluation.metrics),
        "objectives": dict(fitness_space.decode(individual.fitness)),
        "statistics": dict(
            individual.evaluation.details.get("statistics", {})
        ),
        "runtime": _compact_runtime(
            individual.evaluation.details.get("_runtime", {})
        ),
    }
    assert_output_safe(payload)
    return payload


def individual_reference_to_dict(
    individual: Any,
    evaluation_ref_prefix: str = "../evaluations",
) -> Dict[str, Any]:
    """Serialize only generation-varying state plus a candidate-store link."""

    if not individual.evaluated:
        raise ValueError("Cannot checkpoint an unevaluated individual")
    crowding = individual.crowding_distance
    genome_id = individual.genome.fingerprint()
    return {
        "genome_id": genome_id,
        "evaluation_ref": "{}/{}.json".format(
            evaluation_ref_prefix, genome_id
        ),
        "rank": individual.rank,
        "crowding_distance": None if math.isinf(crowding) else crowding,
    }


def _compact_runtime(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    """Persist diagnostics, never arbitrary backend payloads."""

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
        key: value
        for key, value in runtime.items()
        if key in allowed
    }


def write_output_manifest(run_dir: Path, config: ExperimentConfig) -> None:
    """Write compact axes and policy metadata shared by all candidates."""

    benchmark_identity: Dict[str, Any] = {
        "kind": config.benchmark.kind,
        "group_field": config.benchmark.group_field,
        "scorer": config.benchmark.options.get("scorer"),
        "inference_batch_size": config.benchmark.inference_batch_size,
        "max_cases": config.benchmark.max_cases,
    }
    if config.benchmark.path:
        benchmark_path = Path(config.benchmark.path)
        if benchmark_path.is_file():
            benchmark_identity.update(
                {
                    "data_sha256": _sha256_file(benchmark_path),
                    "data_bytes": benchmark_path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": 2,
        "experiment": {"name": config.name, "seed": config.seed},
        "axes": {
            "agents": list(config.agent_names),
            "adapters": list(config.adapter_names),
        },
        "genotype": {
            "encoding": _genotype_encoding(config),
            "agent_axis": "axes.agents",
            "adapter_axis": "axes.adapters",
        },
        "objectives": [
            {"name": item.metric, "direction": item.direction}
            for item in config.objectives
        ],
        "model": {
            "backend": config.model.backend,
            "base_model_id": config.model.base_model_identity,
            "dtype": config.model.dtype,
            "generation": {
                "max_new_tokens": config.model.generation.max_new_tokens,
                "temperature": config.model.generation.temperature,
                "enable_thinking": config.model.generation.enable_thinking,
            },
        },
        "benchmark": benchmark_identity,
        "search": {
            "algorithm": config.search.algorithm,
            "stages": [
                {
                    "index": index,
                    "granularity": config.search.granularities[index],
                    "population_size": config.search.population_size_for(index),
                    "generations": config.search.generations_for(index),
                }
                for index in range(config.search.stage_count)
            ],
            "workers": config.search.workers,
            "checkpoint_every": config.search.checkpoint_every,
            "granularity_options": dict(config.search.granularity_options),
            "algorithm_options": dict(config.search.options),
        },
        "runtime": {
            "mode": config.runtime.mode,
            "devices": list(config.runtime.gpu_ids),
            "start_method": config.runtime.start_method,
        },
        "persistence_policy": {
            "stores_model_weights": False,
            "stores_adapter_weights": False,
            "stores_predictions": False,
            "stores_targets": False,
            "stores_genotypes": True,
        },
    }
    _atomic_write_json(run_dir / "output_manifest.json", manifest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_output_safe(payload: Any) -> None:
    """Reject raw model/adaptor payloads at the persistence boundary."""

    forbidden_keys = {
        "state_dict",
        "model_state_dict",
        "model_weights",
        "adapter_weights",
        "lora_a",
        "lora_b",
        "prediction",
        "target",
        "input_text",
        "prompt",
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    raise ValueError(
                        "Unsafe output field {!r} is not allowed".format(key)
                    )
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)


def _genotype_encoding(config: ExperimentConfig) -> str:
    return (
        "agent_adapter_matrix"
        if config.search.algorithm != "adaptive_resolution_de"
        and config.search.granularities in (("global",), ("adapter",))
        else "named"
    )


def _compact_genotype(
    genotype: Mapping[str, Any],
    agent_names: Sequence[str],
    encoding: str,
) -> Any:
    if (
        encoding == "agent_adapter_matrix"
        and set(genotype.keys()) == set(agent_names)
        and all(isinstance(genotype[name], list) for name in agent_names)
    ):
        return [genotype[name] for name in agent_names]
    return genotype


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, str(path))
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                payload,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        os.replace(temporary_path, str(path))
    except Exception:
        try:
            os.unlink(temporary_path)
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
