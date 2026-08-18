"""Replaceable scalar decompositions and reference-direction utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import List, Sequence, Tuple


Vector = Tuple[float, ...]


class ScalarDecomposition(ABC):
    @abstractmethod
    def value(
        self,
        fitness: Sequence[float],
        weights: Sequence[float],
        ideal_point: Sequence[float],
    ) -> float:
        pass


class TchebycheffDecomposition(ScalarDecomposition):
    def value(self, fitness, weights, ideal_point) -> float:
        return max(
            max(float(weight), 1e-6) * abs(float(value) - float(ideal))
            for value, weight, ideal in zip(fitness, weights, ideal_point)
        )


class WeightedSumDecomposition(ScalarDecomposition):
    def value(self, fitness, weights, ideal_point) -> float:
        del ideal_point
        return sum(
            float(weight) * float(value)
            for value, weight in zip(fitness, weights)
        )


class PBIDecomposition(ScalarDecomposition):
    def __init__(self, theta: float = 5.0) -> None:
        if theta <= 0.0:
            raise ValueError("PBI theta must be positive")
        self.theta = float(theta)

    def value(self, fitness, weights, ideal_point) -> float:
        norm = math.sqrt(sum(float(weight) ** 2 for weight in weights))
        if norm <= 0.0:
            raise ValueError("PBI requires a non-zero reference direction")
        unit = [float(weight) / norm for weight in weights]
        delta = [
            float(value) - float(ideal)
            for value, ideal in zip(fitness, ideal_point)
        ]
        d1 = abs(sum(value * direction for value, direction in zip(delta, unit)))
        projection = [d1 * direction for direction in unit]
        d2 = math.sqrt(
            sum((value - projected) ** 2 for value, projected in zip(delta, projection))
        )
        return d1 + self.theta * d2


def create_decomposition(name: str, *, pbi_theta: float = 5.0) -> ScalarDecomposition:
    normalized = name.strip().lower()
    if normalized == "tchebycheff":
        return TchebycheffDecomposition()
    if normalized == "weighted_sum":
        return WeightedSumDecomposition()
    if normalized == "pbi":
        return PBIDecomposition(pbi_theta)
    raise ValueError("Unknown decomposition {!r}".format(name))


def uniform_reference_directions(count: int, objectives: int) -> Tuple[Vector, ...]:
    """Return exactly ``count`` deterministic simplex directions.

    Das--Dennis lattice points are generated first. If a lattice contains
    more points than requested, greedy maximin subsampling retains the
    objective extremes and then fills the largest angular gaps.
    """

    if count < 1 or objectives < 1:
        raise ValueError("Reference direction dimensions must be positive")
    if objectives == 1:
        return tuple((1.0,) for _ in range(count))
    divisions = 1
    while math.comb(divisions + objectives - 1, objectives - 1) < count:
        divisions += 1
    candidates = tuple(
        tuple(value / float(divisions) for value in composition)
        for composition in _compositions(divisions, objectives)
    )
    if len(candidates) == count:
        return candidates

    selected: List[Vector] = []
    for objective in range(objectives):
        extreme = tuple(1.0 if index == objective else 0.0 for index in range(objectives))
        if extreme in candidates and extreme not in selected and len(selected) < count:
            selected.append(extreme)
    remaining = [item for item in candidates if item not in selected]
    while len(selected) < count:
        choice = max(
            remaining,
            key=lambda item: min(_squared_distance(item, other) for other in selected),
        )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)


def reference_neighborhoods(
    directions: Sequence[Sequence[float]], size: int
) -> Tuple[Tuple[int, ...], ...]:
    if not 1 <= size <= len(directions):
        raise ValueError("Invalid reference neighborhood size")
    return tuple(
        tuple(
            sorted(
                range(len(directions)),
                key=lambda right: (_squared_distance(left, directions[right]), right),
            )[:size]
        )
        for left in directions
    )


def _compositions(total: int, length: int):
    if length == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for tail in _compositions(total - value, length - 1):
            yield (value,) + tail


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


__all__ = [
    "PBIDecomposition",
    "ScalarDecomposition",
    "TchebycheffDecomposition",
    "WeightedSumDecomposition",
    "create_decomposition",
    "reference_neighborhoods",
    "uniform_reference_directions",
]
