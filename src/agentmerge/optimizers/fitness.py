"""Encode benchmark metrics as optimizer-ready minimization fitness."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ..core.config import ObjectiveConfig


class FitnessSpace:
    """The single boundary between measured metrics and optimizer fitness.

    Benchmarks own metric production.  Optimizers consume only a minimization
    vector.  Objective definitions remain user-facing experiment policy and
    are deliberately kept here with the encoding that gives them meaning.
    """

    def __init__(self, definitions: Sequence[ObjectiveConfig]) -> None:
        self.definitions = tuple(definitions)
        if not self.definitions:
            raise ValueError("At least one fitness definition is required")

    @property
    def metric_names(self) -> Tuple[str, ...]:
        return tuple(definition.metric for definition in self.definitions)

    def validate_metrics(self, metric_names: Sequence[str]) -> None:
        available = frozenset(str(name) for name in metric_names)
        missing = [
            name for name in self.metric_names if name not in available
        ]
        if missing:
            raise ValueError(
                "Unknown objective metric(s): {}. Available metrics: {}".format(
                    ", ".join(missing), ", ".join(sorted(available))
                )
            )

    def encode(self, metrics: Mapping[str, float]) -> Tuple[float, ...]:
        """Convert raw metrics into the minimization vector optimizers use."""

        missing = [
            definition.metric
            for definition in self.definitions
            if definition.metric not in metrics
        ]
        if missing:
            raise KeyError(
                "Evaluation did not produce objective metrics: {}".format(
                    ", ".join(missing)
                )
            )
        return tuple(
            (
                -float(metrics[definition.metric])
                if definition.direction == "maximize"
                else float(metrics[definition.metric])
            )
            for definition in self.definitions
        )

    def decode(self, fitness: Sequence[float]) -> Mapping[str, float]:
        """Restore user-facing objective values for persistence and display."""

        if len(fitness) != len(self.definitions):
            raise ValueError(
                "Fitness has {} values, expected {}".format(
                    len(fitness), len(self.definitions)
                )
            )
        return {
            definition.metric: (
                -float(value)
                if definition.direction == "maximize"
                else float(value)
            )
            for definition, value in zip(self.definitions, fitness)
        }

    def primary_metrics(self, population: Sequence[Any]) -> Mapping[str, float]:
        """Return configured metrics for the primary-fitness best individual."""

        primary = self.definitions[0]
        eligible = [
            individual
            for individual in population
            if individual.evaluation is not None
            and primary.metric in individual.evaluation.metrics
        ]
        if not eligible:
            return {}
        select = max if primary.direction == "maximize" else min
        best = select(
            eligible,
            key=lambda individual: float(
                individual.evaluation.metrics[primary.metric]
            ),
        )
        return {
            name: float(best.evaluation.metrics[name])
            for name in self.metric_names
            if name in best.evaluation.metrics
        }

    def objective_best_payloads(
        self, population: Sequence[Any]
    ) -> Tuple[Mapping[str, Any], ...]:
        """Build compact, backend-neutral summaries for every objective."""

        summaries = []
        label_width = max(
            len(_short_metric_name(item.metric)) for item in self.definitions
        )
        for objective in self.definitions:
            eligible = [
                individual
                for individual in population
                if individual.evaluation is not None
                and objective.metric in individual.evaluation.metrics
            ]
            if not eligible:
                continue
            select = max if objective.direction == "maximize" else min
            best = select(
                eligible,
                key=lambda individual: float(
                    individual.evaluation.metrics[objective.metric]
                ),
            )
            evaluation = best.evaluation
            metrics = evaluation.metrics
            runtime = evaluation.details.get("_runtime", {})
            runtime = runtime if isinstance(runtime, Mapping) else {}
            payload: Dict[str, Any] = {
                "objective": objective.metric,
                "objective_label": _short_metric_name(objective.metric),
                "objective_width": label_width,
                "direction": objective.direction,
                "candidate_id": best.genome.fingerprint(),
                "objective_value": float(metrics[objective.metric]),
            }
            for name in (
                "worst_group_quality",
                "quality",
                "latency_seconds",
                "total_tokens",
            ):
                if name in metrics:
                    payload[name] = float(metrics[name])
            gpu_id = runtime.get("gpu_id")
            if gpu_id is not None:
                payload["device"] = gpu_id
            summaries.append(payload)
        return tuple(summaries)


def _short_metric_name(metric: str) -> str:
    for prefix in ("domain_quality.", "task_quality."):
        if metric.startswith(prefix):
            return metric[len(prefix) :]
    return metric


__all__ = ["FitnessSpace"]
