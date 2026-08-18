"""Optimizer contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class SearchResult:
    population: Tuple[Any, ...]
    pareto_front: Tuple[Any, ...]
    generations_completed: int
    unique_evaluations: int
    history: Tuple[Mapping[str, object], ...]


class SearchObserver(ABC):
    """Receives complete generation snapshots for checkpointing/reporting."""

    @abstractmethod
    def on_generation(
        self,
        generation: int,
        population: Sequence[Any],
        unique_evaluations: int,
    ) -> None:
        pass

    def on_evaluation(
        self, individual: Any, cached: bool = False
    ) -> None:
        """Persist or publish one completed candidate evaluation."""

    def on_event(
        self, event_type: str, payload: Mapping[str, Any] = None
    ) -> None:
        """Receive a lightweight structured lifecycle event."""


class NullSearchObserver(SearchObserver):
    def on_generation(
        self,
        generation: int,
        population: Sequence[Any],
        unique_evaluations: int,
    ) -> None:
        return None


class StageObserver(SearchObserver):
    """Add stage metadata and global offsets to one stage-local observer."""

    def __init__(
        self,
        observer: SearchObserver,
        stage_index: int,
        stage_count: int,
        stage_name: str,
        generation_offset: int,
        evaluation_offset: int,
    ) -> None:
        self.observer = observer
        self.stage_index = int(stage_index)
        self.stage_count = int(stage_count)
        self.stage_name = str(stage_name)
        self.generation_offset = int(generation_offset)
        self.evaluation_offset = int(evaluation_offset)

    def on_generation(
        self,
        generation: int,
        population: Sequence[Any],
        unique_evaluations: int,
    ) -> None:
        self.observer.on_generation(
            self.generation_offset + generation,
            population,
            self.evaluation_offset + unique_evaluations,
        )

    def on_evaluation(self, individual: Any, cached: bool = False) -> None:
        self.observer.on_evaluation(individual, cached=cached)

    def on_event(
        self, event_type: str, payload: Mapping[str, Any] = None
    ) -> None:
        data = dict(payload or {})
        data.update(
            {
                "stage_index": self.stage_index,
                "stage_count": self.stage_count,
                "stage": self.stage_name,
            }
        )
        if "generation" in data:
            local_generation = int(data["generation"])
            data["stage_generation"] = local_generation
            data["generation"] = self.generation_offset + local_generation
        if event_type == "generation_completed" and "unique_evaluations" in data:
            data["unique_evaluations"] = (
                self.evaluation_offset + int(data["unique_evaluations"])
            )
        self.observer.on_event(event_type, data)


class Optimizer(ABC):
    @abstractmethod
    def optimize(self) -> SearchResult:
        pass


class AskTellOptimizer(Optimizer):
    """Optimizer whose proposal logic is independent of evaluation scheduling."""

    @property
    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def ask(self) -> Sequence[Any]:
        """Propose the next candidate batch."""

    @abstractmethod
    def evaluate_candidates(
        self, candidates: Sequence[Any]
    ) -> None:
        """Delegate candidate execution to a reusable scheduler/service."""

    @abstractmethod
    def tell(self, candidates: Sequence[Any]) -> None:
        """Update optimizer state from evaluated candidates."""

    @abstractmethod
    def result(self) -> SearchResult:
        """Build the final immutable result after completion."""

    def optimize(self) -> SearchResult:
        while not self.done:
            candidates = tuple(self.ask())
            if not candidates:
                raise RuntimeError(
                    "Ask-tell optimizer returned no candidates before completion"
                )
            self.evaluate_candidates(candidates)
            self.tell(candidates)
        return self.result()
