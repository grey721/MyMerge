"""Algorithm-neutral fusion-plan evaluation cache and scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from ..core.domain import EvaluationResult
from ..core.policies import FusionPlan
from .benchmarks import FusionPlanEvaluator


class EventSink(Protocol):
    def on_event(
        self, event_type: str, payload: Mapping[str, Any] = None
    ) -> None:
        ...


class NullEventSink:
    def on_event(
        self, event_type: str, payload: Mapping[str, Any] = None
    ) -> None:
        return None


@dataclass(frozen=True)
class CachedEvaluation:
    evaluation: EvaluationResult
    cached: bool


class EvaluationBudgetExceeded(RuntimeError):
    """Raised before evaluation when a batch would exceed the real-FE budget."""


class CandidateEvaluationService:
    """Evaluate immutable phenotypes once, independent of any chromosome."""

    def __init__(
        self,
        evaluator: FusionPlanEvaluator,
        workers: int = 1,
        observer: Optional[EventSink] = None,
        max_unique_evaluations: Optional[int] = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.evaluator = evaluator
        self.workers = workers
        self.observer = observer or NullEventSink()
        if max_unique_evaluations is not None and max_unique_evaluations < 1:
            raise ValueError("max_unique_evaluations must be positive")
        self.max_unique_evaluations = max_unique_evaluations
        self._cache: Dict[str, EvaluationResult] = {}
        self._real_evaluations = 0

    @property
    def unique_evaluations(self) -> int:
        return self._real_evaluations

    @property
    def remaining_evaluations(self) -> Optional[int]:
        if self.max_unique_evaluations is None:
            return None
        return self.max_unique_evaluations - self._real_evaluations

    def prime(self, plan: FusionPlan, evaluation: EvaluationResult) -> None:
        """Cache an analytically equivalent phenotype without consuming FE.

        This is used only for semantics-preserving representation changes such
        as replicating one Agent-level composition over every layer.
        """

        self._cache.setdefault(plan.fingerprint(), evaluation)

    def required_evaluations(self, plans: Sequence[FusionPlan]) -> int:
        """Return the exact number of new phenotype evaluations in a batch.

        The method is intentionally read-only. Resolution controllers can use
        it immediately before submission to enforce a hard FE reserve without
        spending an evaluation or approximating cache hits from chromosome
        counts.
        """

        return len(self.uncached_fingerprints(plans))

    def uncached_fingerprints(
        self,
        plans: Sequence[FusionPlan],
    ) -> Tuple[str, ...]:
        """Return first-occurrence-ordered phenotype keys absent from cache."""

        seen = set()
        result = []
        for plan in plans:
            key = plan.fingerprint()
            if key not in self._cache and key not in seen:
                seen.add(key)
                result.append(key)
        return tuple(result)

    def release_unreferenced_adapters(
        self, reference_counts: Mapping[str, int]
    ) -> Mapping[str, Any]:
        """Apply one backend resource barrier between search stages."""

        report = dict(
            self.evaluator.release_unreferenced_adapters(reference_counts)
        )
        report["reference_counts"] = {
            adapter_key: int(count)
            for adapter_key, count in sorted(reference_counts.items())
        }
        self.observer.on_event("adapter_pool_release", report)
        return report

    def evaluate(
        self, plans: Sequence[FusionPlan]
    ) -> Tuple[CachedEvaluation, ...]:
        results = [None] * len(plans)
        pending = {}
        for index, plan in enumerate(plans):
            key = plan.fingerprint()
            cached = self._cache.get(key)
            if cached is not None:
                results[index] = CachedEvaluation(cached, True)
                self.observer.on_event(
                    "candidate_cache_hit", {"plan_id": key}
                )
            else:
                pending.setdefault(key, []).append(index)

        representatives = [plans[indices[0]] for indices in pending.values()]
        if representatives:
            required = len(representatives)
            remaining = self.remaining_evaluations
            if remaining is not None and required > remaining:
                raise EvaluationBudgetExceeded(
                    "Evaluation batch requires {} new FE, only {} remain".format(
                        required, remaining
                    )
                )
            submitted_at = {
                key: time.perf_counter() for key in pending
            }
            for key, indices in pending.items():
                self.observer.on_event(
                    "candidate_submitted",
                    {
                        "plan_id": key,
                        "duplicate_count": len(indices) - 1,
                    },
                )
            try:
                keys = tuple(pending)
                stream = self.evaluator.evaluate_stream(
                    representatives, workers=self.workers
                )
                for representative_index, evaluation in stream:
                    key = keys[representative_index]
                    self._cache[key] = evaluation
                    self._real_evaluations += 1
                    for index in pending[key]:
                        results[index] = CachedEvaluation(evaluation, False)
                    runtime = dict(evaluation.details.get("_runtime", {}))
                    self.observer.on_event(
                        "candidate_completed",
                        {
                            "plan_id": key,
                            "elapsed_since_submission_seconds": (
                                time.perf_counter() - submitted_at[key]
                            ),
                            "metrics": dict(evaluation.metrics),
                            "runtime": runtime,
                        },
                    )
            except Exception as exc:
                self.observer.on_event(
                    "evaluation_batch_failed",
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise
        if any(result is None for result in results):
            raise RuntimeError("Evaluation cache returned incomplete results")
        return tuple(results)


__all__ = [
    "CachedEvaluation",
    "CandidateEvaluationService",
    "EvaluationBudgetExceeded",
]
