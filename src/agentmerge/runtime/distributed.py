"""Persistent process-per-GPU evaluation for candidate-level parallelism."""

from __future__ import annotations

import atexit
import errno
import hashlib
import multiprocessing
import multiprocessing.util
import os
import signal
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from ..evaluation.system import create_agent_system
from ..backends import ModelBackend, create_model_backend
from ..core.config import ExperimentConfig
from ..core.domain import EvaluationResult
from ..core.policies import FusionPlan
from ..evaluation.benchmarks import (
    FusionPlanEvaluator,
    JsonlBenchmark,
    SystemBenchmarkEvaluator,
    build_scorer,
)
from ..evaluation.metrics import BenchmarkMetricCatalog
from .lifecycle import (
    arm_parent_death_signal,
    install_worker_termination_handlers,
)


class MultiGpuEvaluationError(RuntimeError):
    """Raised when a candidate cannot be evaluated by the GPU worker pool."""


@dataclass(frozen=True)
class WorkerStatus:
    worker_index: int
    gpu_id: str
    process_id: int
    backend: str
    vllm_cache_root: Optional[str] = None


@dataclass
class _QueuedEvaluation:
    result_index: int
    plan: FusionPlan
    attempts: int = 0


@dataclass
class _RunningEvaluation:
    task: _QueuedEvaluation
    worker_index: int


_WORKER_EVALUATOR = None  # type: Optional[FusionPlanEvaluator]
_WORKER_BACKEND = None  # type: Optional[ModelBackend]
_WORKER_GPU_ID = None  # type: Optional[str]
_WORKER_FINALIZER = None


def _configure_vllm_worker_cache(gpu_id: str) -> str:
    """Isolate writable compiler artifacts for one persistent GPU worker.

    Each one-GPU AgentMerge worker is rank 0 from vLLM's point of view. If
    workers share VLLM_CACHE_ROOT, concurrent V1 startup processes overwrite
    the same ``rank_0_0`` temporary files. A stable GPU-specific directory
    avoids that race while retaining warm-cache reuse across experiments.
    """

    configured_root = os.environ.get("VLLM_CACHE_ROOT")
    base_root = os.path.abspath(
        os.path.expanduser(configured_root or "~/.cache/vllm")
    )
    gpu_key = hashlib.sha256(str(gpu_id).encode("utf-8")).hexdigest()[:12]
    worker_root = os.path.join(base_root, "agentmerge-workers", gpu_key)
    os.makedirs(worker_root, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = worker_root
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        worker_root, "torchinductor"
    )
    os.environ["TRITON_CACHE_DIR"] = os.path.join(worker_root, "triton")
    return worker_root


def _initialize_gpu_worker(
    config: ExperimentConfig, gpu_id: str, controller_pid: int
) -> None:
    """Initialize exactly once inside a spawned worker process."""

    global _WORKER_BACKEND, _WORKER_EVALUATOR, _WORKER_GPU_ID, _WORKER_FINALIZER
    install_worker_termination_handlers()
    atexit.register(_close_gpu_worker)
    _WORKER_FINALIZER = multiprocessing.util.Finalize(
        None, _close_gpu_worker, exitpriority=20
    )
    if config.runtime.start_method == "spawn":
        arm_parent_death_signal(int(controller_pid))
    else:
        immediate_parent_pid = os.getppid()
        arm_parent_death_signal(immediate_parent_pid)
        _start_controller_monitor(int(controller_pid))
    thread_count = str(config.runtime.cpu_threads_per_worker)
    os.environ.setdefault("OMP_NUM_THREADS", thread_count)
    os.environ.setdefault("MKL_NUM_THREADS", thread_count)
    for key, value in config.runtime.worker_env.items():
        os.environ[str(key)] = str(value)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _WORKER_GPU_ID = str(gpu_id)
    backend_name = config.model.backend.lower()
    if backend_name == "vllm":
        _configure_vllm_worker_cache(str(gpu_id))
    if backend_name in ("peft", "vllm"):
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "GPU worker {} expected exactly one visible CUDA device, "
                "found {}".format(gpu_id, torch.cuda.device_count())
            )
        torch.cuda.set_device(0)
        backend = create_model_backend(config)
    elif backend_name == "mock":
        backend = create_model_backend(config)
    else:
        raise ValueError(
            "Multi-GPU worker does not support backend {!r}".format(
                backend_name
            )
        )

    system = create_agent_system(config, backend)
    benchmark = JsonlBenchmark(
        config.benchmark, build_scorer(config.benchmark)
    )
    _WORKER_BACKEND = backend
    _WORKER_EVALUATOR = SystemBenchmarkEvaluator(system, benchmark)


def _close_gpu_worker() -> None:
    global _WORKER_BACKEND, _WORKER_EVALUATOR
    if _WORKER_EVALUATOR is not None:
        _WORKER_EVALUATOR.close()
        _WORKER_EVALUATOR = None
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.close()
        _WORKER_BACKEND = None


def _start_controller_monitor(controller_pid: int) -> None:
    """Cover the forkserver case where the controller is not our OS parent."""

    def monitor() -> None:
        while True:
            time.sleep(1.0)
            try:
                os.kill(controller_pid, 0)
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                if exc.errno != errno.EPERM:
                    return

    thread = threading.Thread(
        target=monitor,
        name="agentmerge-controller-watch",
        daemon=True,
    )
    thread.start()


def _worker_status() -> WorkerStatus:
    if _WORKER_GPU_ID is None or _WORKER_BACKEND is None:
        raise RuntimeError("GPU worker has not been initialized")
    return WorkerStatus(
        worker_index=-1,
        gpu_id=_WORKER_GPU_ID,
        process_id=os.getpid(),
        backend=type(_WORKER_BACKEND).__name__,
        vllm_cache_root=(
            os.environ.get("VLLM_CACHE_ROOT")
            if type(_WORKER_BACKEND).__name__ == "VllmCausalLMBackend"
            else None
        ),
    )


def _worker_evaluate(plan: FusionPlan) -> EvaluationResult:
    if _WORKER_EVALUATOR is None or _WORKER_GPU_ID is None:
        raise RuntimeError("GPU worker has not been initialized")
    started = time.perf_counter()
    evaluation = _WORKER_EVALUATOR.evaluate(plan)
    details = dict(evaluation.details)
    runtime = dict(details.get("_runtime", {}))
    runtime.update(
        {
            "gpu_id": _WORKER_GPU_ID,
            "process_id": os.getpid(),
            "evaluation_seconds": time.perf_counter() - started,
        }
    )
    details["_runtime"] = runtime
    return EvaluationResult(metrics=evaluation.metrics, details=details)


def _worker_release_unreferenced_adapters(
    reference_counts: Mapping[str, int]
) -> Mapping[str, Any]:
    if _WORKER_BACKEND is None or _WORKER_GPU_ID is None:
        raise RuntimeError("GPU worker has not been initialized")
    report = dict(
        _WORKER_BACKEND.release_unreferenced_adapters(reference_counts)
    )
    report.update(
        {
            "gpu_id": _WORKER_GPU_ID,
            "process_id": os.getpid(),
        }
    )
    return report


class MultiGpuFusionPlanEvaluator(FusionPlanEvaluator):
    """Load one persistent model replica per GPU and schedule whole plans.

    Every executor owns one process, so a model and its global adapter pool are
    loaded once and reused for all candidates assigned to that physical GPU.
    Candidate assignment is dynamic: a GPU receives new work as soon as its
    previous evaluation finishes.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        if config.runtime.mode != "multi_gpu":
            raise ValueError("MultiGpuFusionPlanEvaluator requires multi_gpu mode")
        if config.benchmark.kind.lower() != "jsonl":
            raise ValueError("Multi-GPU evaluation currently requires JSONL")
        self.config = config
        self.gpu_ids = tuple(config.runtime.gpu_ids)
        benchmark = JsonlBenchmark(
            config.benchmark, build_scorer(config.benchmark)
        )
        self._metric_names = BenchmarkMetricCatalog.from_groups(
            tuple(case.group for case in benchmark.cases)
        ).names
        self._closed = False
        self._context = multiprocessing.get_context(
            config.runtime.start_method
        )
        self._executors = [
            self._create_executor(gpu_id) for gpu_id in self.gpu_ids
        ]
        self._statuses = ()  # type: Tuple[WorkerStatus, ...]
        self._adapter_reference_counts = None  # type: Optional[Dict[str, int]]
        if config.runtime.preload_workers:
            self._statuses = self._preload()

    @property
    def statuses(self) -> Tuple[WorkerStatus, ...]:
        return self._statuses

    @property
    def metric_names(self) -> Tuple[str, ...]:
        return self._metric_names

    def runtime_metadata(self):
        return {
            "evaluator": type(self).__name__,
            "mode": "multi_gpu",
            "controller_pid": os.getpid(),
            "process_group_id": os.getpgrp(),
            "worker_parent_death_signal": "SIGTERM",
            "configured_gpu_ids": list(self.gpu_ids),
            "workers": [
                {
                    "worker_index": status.worker_index,
                    "gpu_id": status.gpu_id,
                    "process_id": status.process_id,
                    "backend": status.backend,
                    "vllm_cache_root": status.vllm_cache_root,
                }
                for status in self._statuses
            ],
            "start_method": self.config.runtime.start_method,
        }

    def _create_executor(self, gpu_id: str) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=self._context,
            initializer=_initialize_gpu_worker,
            initargs=(self.config, gpu_id, os.getpid()),
        )

    def _preload(self) -> Tuple[WorkerStatus, ...]:
        futures = [
            executor.submit(_worker_status)
            for executor in self._executors
        ]
        statuses = []
        timeout = self.config.runtime.evaluation_timeout_seconds
        try:
            for worker_index, (gpu_id, future) in enumerate(
                zip(self.gpu_ids, futures)
            ):
                status = future.result(timeout=timeout)
                statuses.append(
                    WorkerStatus(
                        worker_index=worker_index,
                        gpu_id=status.gpu_id,
                        process_id=status.process_id,
                        backend=status.backend,
                        vllm_cache_root=status.vllm_cache_root,
                    )
                )
        except Exception as exc:
            self.close(wait_for_workers=False)
            raise MultiGpuEvaluationError(
                "Failed to preload model worker for GPU {}: {}".format(
                    gpu_id, exc
                )
            ) from exc
        return tuple(statuses)

    def evaluate(self, plan: FusionPlan) -> EvaluationResult:
        return self.evaluate_many((plan,))[0]

    def evaluate_many(
        self, plans: Sequence[FusionPlan], workers: int = 1
    ) -> Sequence[EvaluationResult]:
        results = [None] * len(plans)
        for index, evaluation in self.evaluate_stream(plans, workers):
            results[index] = evaluation
        if any(result is None for result in results):
            raise RuntimeError("GPU scheduler completed with missing results")
        return results

    def evaluate_stream(
        self, plans: Sequence[FusionPlan], workers: int = 1
    ) -> Iterator[Tuple[int, EvaluationResult]]:
        if self._closed:
            raise RuntimeError("Multi-GPU evaluator is closed")
        if not plans:
            return

        queued = deque(
            _QueuedEvaluation(index, plan)
            for index, plan in enumerate(plans)
        )  # type: Deque[_QueuedEvaluation]
        available = deque(range(len(self._executors)))  # type: Deque[int]
        running = {}  # type: Dict[Future, _RunningEvaluation]
        completed_indices = set()

        self._fill_available_workers(queued, available, running)
        timeout = self.config.runtime.evaluation_timeout_seconds
        while running:
            completed, _ = wait(
                tuple(running),
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                self.close(wait_for_workers=False)
                raise MultiGpuEvaluationError(
                    "No GPU evaluation completed within {} seconds".format(
                        timeout
                    )
                )

            for future in completed:
                active = running.pop(future)
                evaluation = None
                try:
                    evaluation = future.result()
                    details = dict(evaluation.details)
                    runtime = dict(details.get("_runtime", {}))
                    runtime.update(
                        {
                            "worker_index": active.worker_index,
                            "scheduler_attempts": active.task.attempts + 1,
                        }
                    )
                    details["_runtime"] = runtime
                    evaluation = EvaluationResult(
                        metrics=evaluation.metrics, details=details
                    )
                    completed_indices.add(active.task.result_index)
                except Exception as exc:
                    if isinstance(exc, BrokenProcessPool):
                        self._restart_executor(active.worker_index)
                    if active.task.attempts < self.config.runtime.max_retries:
                        active.task.attempts += 1
                        queued.appendleft(active.task)
                    else:
                        self.close(wait_for_workers=False)
                        raise MultiGpuEvaluationError(
                            "Candidate {} failed on GPU {} after {} "
                            "attempt(s): {}".format(
                                active.task.plan.fingerprint()[:12],
                                self.gpu_ids[active.worker_index],
                                active.task.attempts + 1,
                                exc,
                            )
                        ) from exc
                available.append(active.worker_index)
                self._fill_available_workers(queued, available, running)
                if evaluation is not None:
                    yield active.task.result_index, evaluation

            self._fill_available_workers(queued, available, running)

        if len(completed_indices) != len(plans):
            raise RuntimeError("GPU scheduler completed with missing results")

    def _fill_available_workers(
        self,
        queued: Deque[_QueuedEvaluation],
        available: Deque[int],
        running: Dict[Future, _RunningEvaluation],
    ) -> None:
        while queued and available:
            worker_index = available.popleft()
            task = queued.popleft()
            future = self._executors[worker_index].submit(
                _worker_evaluate, task.plan
            )
            running[future] = _RunningEvaluation(
                task=task, worker_index=worker_index
            )

    def _restart_executor(self, worker_index: int) -> None:
        old_executor = self._executors[worker_index]
        old_executor.shutdown(wait=False)
        self._executors[worker_index] = self._create_executor(
            self.gpu_ids[worker_index]
        )
        if self._adapter_reference_counts is not None:
            future = self._executors[worker_index].submit(
                _worker_release_unreferenced_adapters,
                self._adapter_reference_counts,
            )
            try:
                future.result(
                    timeout=self.config.runtime.evaluation_timeout_seconds
                )
            except Exception as exc:
                raise MultiGpuEvaluationError(
                    "Restarted GPU worker {} failed to restore the pruned "
                    "Adapter pool: {}".format(
                        self.gpu_ids[worker_index], exc
                    )
                ) from exc

    def release_unreferenced_adapters(
        self, reference_counts: Mapping[str, int]
    ) -> Mapping[str, Any]:
        """Broadcast a synchronized source-pool release to every GPU worker."""

        if self._closed:
            raise RuntimeError("Multi-GPU evaluator is closed")
        counts = {str(key): int(value) for key, value in reference_counts.items()}
        futures = [
            executor.submit(_worker_release_unreferenced_adapters, counts)
            for executor in self._executors
        ]
        reports = []
        timeout = self.config.runtime.evaluation_timeout_seconds
        try:
            for future in futures:
                reports.append(dict(future.result(timeout=timeout)))
        except Exception as exc:
            self.close(wait_for_workers=False)
            raise MultiGpuEvaluationError(
                "Failed to release zero-reference Adapters on every GPU: {}"
                .format(exc)
            ) from exc

        release_sets = {
            tuple(report.get("released_adapter_keys", ()))
            for report in reports
        }
        resident_sets = {
            tuple(report.get("resident_adapter_keys", ()))
            for report in reports
        }
        if len(release_sets) != 1 or len(resident_sets) != 1:
            self.close(wait_for_workers=False)
            raise MultiGpuEvaluationError(
                "GPU workers produced inconsistent Adapter-pool states"
            )
        self._adapter_reference_counts = counts
        first = reports[0]
        return {
            "supported": all(bool(item.get("supported")) for item in reports),
            "worker_count": len(reports),
            "released_adapter_keys": list(
                first.get("released_adapter_keys", ())
            ),
            "released_coordinate_indices": list(
                first.get("released_coordinate_indices", ())
            ),
            "released_source_count": int(
                first.get("released_source_count", 0)
            ),
            "released_source_bytes": sum(
                int(item.get("released_source_bytes", 0))
                for item in reports
            ),
            "resident_adapter_keys": list(
                first.get("resident_adapter_keys", ())
            ),
            "resident_coordinate_indices": list(
                first.get("resident_coordinate_indices", ())
            ),
            "resident_source_count": int(
                first.get("resident_source_count", 0)
            ),
            "resident_source_bytes": sum(
                int(item.get("resident_source_bytes", 0))
                for item in reports
            ),
            "workers": reports,
        }

    def close(self, wait_for_workers: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        for executor in self._executors:
            _shutdown_executor(executor, wait_for_workers)

    def abort(self) -> None:
        """Stop active workers promptly after a signal or controller failure."""

        self.close(wait_for_workers=False)


def _shutdown_executor(
    executor: ProcessPoolExecutor, wait_for_workers: bool
) -> None:
    if wait_for_workers:
        executor.shutdown(wait=True)
        return

    # Python 3.10 lacks ProcessPoolExecutor.terminate_workers(). Keep this
    # compatibility code isolated until the project moves to Python 3.14+.
    processes = tuple(
        process
        for process in (getattr(executor, "_processes", {}) or {}).values()
        if process is not None
    )
    for process in processes:
        if process.is_alive():
            process.terminate()
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        # ``cancel_futures`` was added after the project's minimum Python 3.8.
        executor.shutdown(wait=False)
    deadline = time.monotonic() + 5.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
