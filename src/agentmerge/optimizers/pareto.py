"""Reusable Pareto-ranking primitives for multi-objective optimizers."""

from __future__ import annotations

import math
from typing import List, Optional, Protocol, Sequence, Tuple


class ParetoIndividual(Protocol):
    fitness: Optional[Tuple[float, ...]]
    rank: int
    crowding_distance: float


def dominates(left: ParetoIndividual, right: ParetoIndividual) -> bool:
    if left.fitness is None or right.fitness is None:
        raise ValueError("Cannot compare unevaluated individuals")
    no_worse = all(
        left_value <= right_value
        for left_value, right_value in zip(left.fitness, right.fitness)
    )
    strictly_better = any(
        left_value < right_value
        for left_value, right_value in zip(left.fitness, right.fitness)
    )
    return no_worse and strictly_better


def fast_non_dominated_sort(
    population: Sequence[ParetoIndividual],
) -> List[List[ParetoIndividual]]:
    domination_counts = [0] * len(population)
    dominated = [[] for _ in population]  # type: List[List[int]]
    first_front: List[int] = []

    for left_index, left in enumerate(population):
        for right_index, right in enumerate(population):
            if left_index == right_index:
                continue
            if dominates(left, right):
                dominated[left_index].append(right_index)
            elif dominates(right, left):
                domination_counts[left_index] += 1
        if domination_counts[left_index] == 0:
            first_front.append(left_index)

    fronts: List[List[int]] = [first_front]
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: List[int] = []
        for left_index in fronts[front_index]:
            for right_index in dominated[left_index]:
                domination_counts[right_index] -= 1
                if domination_counts[right_index] == 0:
                    next_front.append(right_index)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return [
        [population[index] for index in front]
        for front in fronts
        if front
    ]


def assign_crowding_distance(
    front: Sequence[ParetoIndividual],
) -> None:
    if not front:
        return
    for individual in front:
        individual.crowding_distance = 0.0
    if len(front) <= 2:
        for individual in front:
            individual.crowding_distance = math.inf
        return

    objective_count = len(front[0].fitness or ())
    for objective_index in range(objective_count):
        ordered = sorted(
            front,
            key=lambda individual: individual.fitness[objective_index],
        )
        ordered[0].crowding_distance = math.inf
        ordered[-1].crowding_distance = math.inf
        minimum = ordered[0].fitness[objective_index]
        maximum = ordered[-1].fitness[objective_index]
        if maximum == minimum:
            continue
        scale = maximum - minimum
        for index in range(1, len(ordered) - 1):
            if math.isinf(ordered[index].crowding_distance):
                continue
            previous_value = ordered[index - 1].fitness[objective_index]
            next_value = ordered[index + 1].fitness[objective_index]
            ordered[index].crowding_distance += (
                next_value - previous_value
            ) / scale


__all__ = [
    "assign_crowding_distance",
    "dominates",
    "fast_non_dominated_sort",
]
