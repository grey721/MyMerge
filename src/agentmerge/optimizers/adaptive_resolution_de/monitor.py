"""Successful-DE direction coherence and marginal utility monitoring.

This monitor does not construct a CMA-ES evolution path and never proposes or
evaluates a probe. It only reuses already evaluated DE trial displacements.
Reference-vector sectors keep different Pareto preferences from cancelling
one another, following the regional-direction motivation of Qin et al. (2021)
without adopting their additional directed-sampling evaluations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from statistics import median
from typing import Deque, Dict, Hashable, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class DirectionalCheckpoint:
    valid_direction_sectors: int
    total_sectors: int
    valid_sector_fraction: float
    directional_coherence: Optional[float]
    marginal_utility: float
    utility_ema: float
    utility_peak: float
    utility_ratio: float


class DirectionalMonitor:
    """Aggregate existing successful DE directions by Pareto sector."""

    def __init__(
        self,
        reference_directions: Sequence[Sequence[float]],
        *,
        num_sectors: int,
        sector_seed: int,
        direction_window: int,
        min_directions_per_sector: int,
        direction_norm_eps: float,
        utility_ema_beta: float,
        utility_eps: float,
        structural_weights: Sequence[float],
    ) -> None:
        if not reference_directions:
            raise ValueError("DirectionalMonitor requires reference directions")
        self.sector_ids = _cluster_reference_directions(
            reference_directions,
            min(int(num_sectors), len(reference_directions)),
            int(sector_seed),
        )
        self.total_sectors = max(self.sector_ids) + 1
        self.direction_window = int(direction_window)
        self.min_directions_per_sector = int(min_directions_per_sector)
        self.direction_norm_eps = float(direction_norm_eps)
        self.utility_ema_beta = float(utility_ema_beta)
        self.utility_eps = float(utility_eps)
        self._structural_weights = tuple(float(item) for item in structural_weights)
        self.reset_for_new_stage(structural_weights)

    @property
    def direction_counts(self) -> Tuple[int, ...]:
        return tuple(len(self._directions[index]) for index in range(self.total_sectors))

    def observe(
        self,
        subproblem_idx: int,
        parent_vector: Sequence[float],
        trial_vector: Sequence[float],
        scalar_before: float,
        scalar_trial: float,
        *,
        sample_id: Optional[Hashable] = None,
    ) -> bool:
        """Record one real evaluated offspring at most once.

        Scalar values must be computed with the same ideal-point snapshot.
        The scalarization is minimization-oriented because FitnessSpace has
        already converted maximization objectives to minimization fitness.
        """

        if sample_id is not None:
            if sample_id in self._seen_samples:
                return False
            self._seen_samples.add(sample_id)
        if not 0 <= subproblem_idx < len(self.sector_ids):
            raise IndexError("subproblem_idx is outside reference directions")
        if len(parent_vector) != len(self._structural_weights) or len(
            trial_vector
        ) != len(self._structural_weights):
            raise ValueError("DE direction has the wrong decision dimension")

        sector = self.sector_ids[subproblem_idx]
        denominator = abs(float(scalar_before)) + self.utility_eps
        gain = max(
            0.0,
            (float(scalar_before) - float(scalar_trial)) / denominator,
        )
        self._interval_gain[sector] += gain
        self._interval_fe[sector] += 1
        if gain <= 0.0:
            return True

        direction = [
            (float(trial) - float(parent)) * weight
            for parent, trial, weight in zip(
                parent_vector, trial_vector, self._structural_weights
            )
        ]
        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= self.direction_norm_eps:
            return True
        self._directions[sector].append(
            tuple(value / norm for value in direction)
        )
        return True

    def checkpoint(self) -> DirectionalCheckpoint:
        coherences = []
        for sector in range(self.total_sectors):
            directions = self._directions[sector]
            if len(directions) >= self.min_directions_per_sector:
                coherences.append(_mean_pairwise_cosine(directions))
        valid_count = len(coherences)
        coherence = median(coherences) if coherences else None

        utilities = [
            self._interval_gain[sector] / self._interval_fe[sector]
            for sector in range(self.total_sectors)
            if self._interval_fe[sector] > 0
        ]
        marginal = float(median(utilities)) if utilities else 0.0
        if self._utility_ema is None:
            self._utility_ema = marginal
        else:
            self._utility_ema = (
                self.utility_ema_beta * self._utility_ema
                + (1.0 - self.utility_ema_beta) * marginal
            )
        self._utility_peak = max(self._utility_peak, self._utility_ema)
        utility_ratio = self._utility_ema / (
            self._utility_peak + self.utility_eps
        )
        if self._utility_peak <= self.utility_eps:
            utility_ratio = 0.0

        self._interval_gain = [0.0] * self.total_sectors
        self._interval_fe = [0] * self.total_sectors
        return DirectionalCheckpoint(
            valid_direction_sectors=valid_count,
            total_sectors=self.total_sectors,
            valid_sector_fraction=valid_count / float(self.total_sectors),
            directional_coherence=coherence,
            marginal_utility=marginal,
            utility_ema=float(self._utility_ema),
            utility_peak=float(self._utility_peak),
            utility_ratio=float(utility_ratio),
        )

    def reset_for_new_stage(
        self,
        structural_weights: Sequence[float],
    ) -> None:
        self._structural_weights = tuple(float(item) for item in structural_weights)
        if not self._structural_weights:
            raise ValueError("Structural direction weights cannot be empty")
        self._directions = {
            sector: deque(maxlen=self.direction_window)
            for sector in range(self.total_sectors)
        }  # type: Dict[int, Deque[Tuple[float, ...]]]
        self._interval_gain = [0.0] * self.total_sectors
        self._interval_fe = [0] * self.total_sectors
        self._utility_ema = None  # type: Optional[float]
        self._utility_peak = 0.0
        self._seen_samples = set()  # type: Set[Hashable]


def _mean_pairwise_cosine(
    directions: Sequence[Sequence[float]],
) -> float:
    """Compute pairwise cosine in O(nD) from the norm of the vector sum."""

    count = len(directions)
    if count < 2:
        raise ValueError("Pairwise coherence requires at least two directions")
    width = len(directions[0])
    summed = [0.0] * width
    for direction in directions:
        if len(direction) != width:
            raise ValueError("Direction dimensions differ")
        for index, value in enumerate(direction):
            summed[index] += float(value)
    squared_norm = sum(value * value for value in summed)
    # C <= 1. Random independent directions have expectation near zero;
    # negative values indicate net cancellation. For n > 2 the attainable
    # lower bound is not generally -1.
    return min(1.0, (squared_norm - count) / (count * (count - 1)))


def _cluster_reference_directions(
    directions: Sequence[Sequence[float]],
    count: int,
    seed: int,
) -> Tuple[int, ...]:
    if count < 1:
        raise ValueError("num_sectors must be positive")
    normalized = tuple(_unit(item) for item in directions)
    rng = random.Random(seed)
    first = rng.randrange(len(normalized))
    centroid_indices = [first]
    while len(centroid_indices) < count:
        remaining = [
            index
            for index in range(len(normalized))
            if index not in centroid_indices
        ]
        choice = min(
            remaining,
            key=lambda index: (
                max(
                    _dot(normalized[index], normalized[other])
                    for other in centroid_indices
                ),
                index,
            ),
        )
        centroid_indices.append(choice)
    centroids = [normalized[index] for index in centroid_indices]
    assignments = [0] * len(normalized)
    for _ in range(32):
        updated = [
            max(
                range(count),
                key=lambda sector: (
                    _dot(direction, centroids[sector]), -sector
                ),
            )
            for direction in normalized
        ]
        if updated == assignments:
            break
        assignments = updated
        for sector in range(count):
            members = [
                normalized[index]
                for index, assigned in enumerate(assignments)
                if assigned == sector
            ]
            if members:
                centroids[sector] = _unit(
                    tuple(
                        sum(item[column] for item in members)
                        for column in range(len(members[0]))
                    )
                )
    return tuple(assignments)


def _unit(values: Sequence[float]) -> Tuple[float, ...]:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 0.0:
        raise ValueError("Reference directions must be non-zero")
    return tuple(float(value) / norm for value in values)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


__all__ = [
    "DirectionalCheckpoint",
    "DirectionalMonitor",
]
