"""Population-update policy separated from DE reproduction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from .decomposition import ScalarDecomposition
from .genome import DEIndividual


class PopulationUpdateStrategy(ABC):
    """A replaceable multi-objective survival boundary.

    The first implementation is MOEA/D scalar replacement. A Pareto/GDE3
    implementation can later satisfy this boundary without changing DE
    variation, evaluation, search-space encoding, or runtime translation.
    """

    @abstractmethod
    def update(
        self,
        population: Sequence[DEIndividual],
        offspring: Sequence[DEIndividual],
        reference_directions: Sequence[Sequence[float]],
        ideal_point: Sequence[float],
    ) -> List[DEIndividual]:
        pass


class DecompositionUpdate(PopulationUpdateStrategy):
    def __init__(
        self,
        decomposition: ScalarDecomposition,
        max_replacements: int,
    ) -> None:
        self.decomposition = decomposition
        self.max_replacements = max_replacements

    def update(
        self,
        population: Sequence[DEIndividual],
        offspring: Sequence[DEIndividual],
        reference_directions: Sequence[Sequence[float]],
        ideal_point: Sequence[float],
    ) -> List[DEIndividual]:
        updated = [individual.clone() for individual in population]
        for trial in offspring:
            if trial.fitness is None:
                raise ValueError("Cannot select with unevaluated DE offspring")
            replacement_count = 0
            for index in trial.replacement_indices:
                incumbent = updated[index]
                if incumbent.fitness is None:
                    raise ValueError(
                        "DE population contains an unevaluated individual"
                    )
                weights = reference_directions[index]
                old_value = self.decomposition.value(
                    incumbent.fitness, weights, ideal_point
                )
                new_value = self.decomposition.value(
                    trial.fitness, weights, ideal_point
                )
                if new_value <= old_value:
                    updated[index] = trial.clone()
                    replacement_count += 1
                    if replacement_count >= self.max_replacements:
                        break
        return updated


__all__ = ["DecompositionUpdate", "PopulationUpdateStrategy"]
