"""NSGA-II crossover and mutation for coefficient search spaces."""

from __future__ import annotations

import random
from typing import List

from ..contracts import EvolutionSearchSpace
from ..contracts import SearchGenome


class NSGA2Variation:
    """Real-vector crossover and mutation owned by the NSGA-II package."""

    def __init__(
        self,
        search_space: EvolutionSearchSpace,
        crossover_probability: float,
        mutation_probability: float,
        sparsity_mutation_probability: float,
        mutation_sigma: float,
    ) -> None:
        self.search_space = search_space
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.sparsity_mutation_probability = sparsity_mutation_probability
        self.mutation_sigma = mutation_sigma

    def crossover(
        self,
        left: SearchGenome,
        right: SearchGenome,
        rng: random.Random,
    ) -> SearchGenome:
        if rng.random() > self.crossover_probability:
            return left.clone()
        left_values = self.search_space.encode(left)
        right_values = self.search_space.encode(right)
        child = list(left_values)
        for start, end in self.search_space.coefficient_groups:
            alpha = rng.random()
            for index in range(start, end):
                child[index] = (
                    alpha * left_values[index]
                    + (1.0 - alpha) * right_values[index]
                )
        return self.search_space.decode(child, rng)

    def mutate(
        self, genome: SearchGenome, rng: random.Random
    ) -> SearchGenome:
        values = list(self.search_space.encode(genome))
        bounds = self.search_space.bounds
        for start, end in self.search_space.coefficient_groups:
            for index in range(start, end):
                if rng.random() < self.mutation_probability:
                    lower, upper = bounds[index]
                    values[index] = min(
                        upper,
                        max(
                            lower,
                            values[index] + rng.gauss(0.0, self.mutation_sigma),
                        ),
                    )
            if rng.random() < self.sparsity_mutation_probability:
                self._mutate_sparsity(values, start, end, rng)
        return self.search_space.decode(values, rng)

    @staticmethod
    def _mutate_sparsity(
        values: List[float], start: int, end: int, rng: random.Random
    ) -> None:
        active = [index for index in range(start, end) if values[index] > 0.0]
        inactive = [index for index in range(start, end) if values[index] <= 0.0]
        if active and inactive and rng.random() < 0.5:
            values[rng.choice(active)] = 0.0
        elif inactive:
            values[rng.choice(inactive)] = rng.uniform(0.05, 0.5)


__all__ = ["NSGA2Variation"]
