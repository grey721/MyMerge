"""Canonical DE/current/1/bin with bounded polynomial mutation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import List, Optional

from ..contracts import EvolutionSearchSpace, SearchGenome


class DEVariation(ABC):
    @abstractmethod
    def create(
        self,
        target: SearchGenome,
        donor_left: SearchGenome,
        donor_right: SearchGenome,
        rng: random.Random,
    ) -> SearchGenome:
        pass


class DifferentialVariation(DEVariation):
    def __init__(
        self,
        search_space: EvolutionSearchSpace,
        differential_weight: float,
        crossover_probability: float,
        mutation_probability: Optional[float],
        mutation_distribution_index: float,
    ) -> None:
        self.search_space = search_space
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.mutation_distribution_index = mutation_distribution_index

    def create(
        self,
        target: SearchGenome,
        donor_left: SearchGenome,
        donor_right: SearchGenome,
        rng: random.Random,
    ) -> SearchGenome:
        target_values = self.search_space.encode(target)
        left_values = self.search_space.encode(donor_left)
        right_values = self.search_space.encode(donor_right)
        bounds = self.search_space.bounds
        forced_index = rng.randrange(len(target_values))
        trial: List[float] = []
        for index, target_value in enumerate(target_values):
            use_donor = (
                index == forced_index
                or rng.random() < self.crossover_probability
            )
            value = (
                target_value
                + self.differential_weight
                * (left_values[index] - right_values[index])
                if use_donor
                else target_value
            )
            lower, upper = bounds[index]
            trial.append(min(upper, max(lower, value)))

        probability = (
            self.mutation_probability
            if self.mutation_probability is not None
            else 1.0 / len(trial)
        )
        for index, (lower, upper) in enumerate(bounds):
            if rng.random() < probability and upper > lower:
                trial[index] = self._polynomial_mutation(
                    trial[index], lower, upper, rng
                )
        return self.search_space.decode(trial, rng)

    def _polynomial_mutation(
        self,
        value: float,
        lower: float,
        upper: float,
        rng: random.Random,
    ) -> float:
        delta_left = (value - lower) / (upper - lower)
        delta_right = (upper - value) / (upper - lower)
        random_value = rng.random()
        power = 1.0 / (self.mutation_distribution_index + 1.0)
        if random_value <= 0.5:
            base = 2.0 * random_value + (1.0 - 2.0 * random_value) * (
                1.0 - delta_left
            ) ** (self.mutation_distribution_index + 1.0)
            delta = base**power - 1.0
        else:
            base = 2.0 * (1.0 - random_value) + 2.0 * (
                random_value - 0.5
            ) * (1.0 - delta_right) ** (
                self.mutation_distribution_index + 1.0
            )
            delta = 1.0 - base**power
        return min(upper, max(lower, value + delta * (upper - lower)))


__all__ = ["DEVariation", "DifferentialVariation"]
