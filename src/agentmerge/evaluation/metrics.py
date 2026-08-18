"""Benchmark metric schema and task-aware quality aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


TASK_QUALITY_PREFIX = "task_quality."
DOMAIN_QUALITY_PREFIX = "domain_quality."

# Metrics emitted for every real system benchmark. Keeping the schema here
# lets validation reject an invalid objective before loading any model.
SYSTEM_METRIC_NAMES = (
    "quality",
    "worst_group_quality",
    "latency_seconds",
    "model_latency_seconds",
    "candidate_session_seconds",
    "fusion_materialization_seconds",
    "fusion_release_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "active_adapters",
    "fusion_rank_proxy",
)

SYNTHETIC_METRIC_NAMES = (
    "quality",
    "robustness",
    "worst_group_quality",
    "active_adapters",
    "latency_proxy",
    "fusion_rank_proxy",
)


@dataclass(frozen=True)
class BenchmarkMetricCatalog:
    """Available metrics for one benchmark and its task groups.

    The JSONL ``group`` field is the task identity. Each task is exposed as a
    stable optimization metric while aggregate quality remains available for
    reporting or robust-objective experiments.
    """

    task_names: Tuple[str, ...]
    quality_domains: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted({str(name).strip() for name in self.task_names}))
        if not normalized or any(not name for name in normalized):
            raise ValueError("Benchmark task names must be non-empty")
        object.__setattr__(self, "task_names", normalized)
        normalized_domains = []
        domain_names = set()
        for domain, tasks in self.quality_domains:
            domain_name = str(domain).strip()
            task_group = tuple(dict.fromkeys(str(task).strip() for task in tasks))
            if not domain_name or domain_name in domain_names:
                raise ValueError("Quality domain names must be unique and non-empty")
            if not task_group or any(not task for task in task_group):
                raise ValueError(
                    "Quality domain {!r} must contain tasks".format(domain_name)
                )
            unknown = [task for task in task_group if task not in normalized]
            if unknown:
                raise ValueError(
                    "Quality domain {!r} references unknown task(s): {}".format(
                        domain_name, ", ".join(unknown)
                    )
                )
            domain_names.add(domain_name)
            normalized_domains.append((domain_name, task_group))
        object.__setattr__(
            self, "quality_domains", tuple(sorted(normalized_domains))
        )

    @classmethod
    def from_groups(
        cls,
        groups: Sequence[str],
        quality_domains: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> "BenchmarkMetricCatalog":
        domains = tuple(
            (str(name), tuple(str(task) for task in tasks))
            for name, tasks in (quality_domains or {}).items()
        )
        return cls(tuple(str(group) for group in groups), domains)

    @property
    def names(self) -> Tuple[str, ...]:
        return SYSTEM_METRIC_NAMES + tuple(
            self.task_quality_name(task) for task in self.task_names
        ) + tuple(
            self.domain_quality_name(domain)
            for domain, _ in self.quality_domains
        )

    @staticmethod
    def task_quality_name(task: str) -> str:
        name = str(task).strip()
        if not name:
            raise ValueError("Task name cannot be empty")
        return TASK_QUALITY_PREFIX + name

    @staticmethod
    def domain_quality_name(domain: str) -> str:
        name = str(domain).strip()
        if not name:
            raise ValueError("Domain name cannot be empty")
        return DOMAIN_QUALITY_PREFIX + name

    def aggregate_quality(
        self,
        group_scores: Mapping[str, Sequence[float]],
    ) -> Dict[str, float]:
        """Return micro quality, worst-task quality and every task mean."""

        missing = [task for task in self.task_names if not group_scores.get(task)]
        if missing:
            raise ValueError(
                "Evaluation did not produce scores for task(s): {}".format(
                    ", ".join(missing)
                )
            )
        per_task = {
            task: _mean(group_scores[task]) for task in self.task_names
        }
        all_scores = [
            float(value)
            for task in self.task_names
            for value in group_scores[task]
        ]
        result = {
            "quality": _mean(all_scores),
            "worst_group_quality": min(per_task.values()),
        }
        result.update(
            {
                self.task_quality_name(task): value
                for task, value in per_task.items()
            }
        )
        result.update(
            {
                self.domain_quality_name(domain): _mean(
                    tuple(per_task[task] for task in tasks)
                )
                for domain, tasks in self.quality_domains
            }
        )
        return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty metric sequence")
    return sum(float(value) for value in values) / len(values)


__all__ = [
    "BenchmarkMetricCatalog",
    "DOMAIN_QUALITY_PREFIX",
    "SYNTHETIC_METRIC_NAMES",
    "SYSTEM_METRIC_NAMES",
    "TASK_QUALITY_PREFIX",
]
