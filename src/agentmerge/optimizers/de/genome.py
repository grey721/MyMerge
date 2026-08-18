"""Evaluated individual state used by decomposition-based DE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ...core.domain import EvaluationResult
from ..contracts import SearchGenome


@dataclass
class DEIndividual:
    genome: SearchGenome
    fitness: Optional[Tuple[float, ...]] = None
    evaluation: Optional[EvaluationResult] = None
    rank: int = 0
    crowding_distance: float = 0.0
    target_index: Optional[int] = None
    replacement_indices: Tuple[int, ...] = ()

    @property
    def evaluated(self) -> bool:
        return self.fitness is not None and self.evaluation is not None

    def clone(self) -> "DEIndividual":
        return DEIndividual(
            genome=self.genome.clone(),
            fitness=self.fitness,
            evaluation=self.evaluation,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            target_index=self.target_index,
            replacement_indices=self.replacement_indices,
        )


__all__ = ["DEIndividual"]
