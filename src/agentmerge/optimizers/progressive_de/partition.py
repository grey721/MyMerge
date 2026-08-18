"""Nested contiguous layer partitions for progressive freedom release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from ...core.policies import FusionGranularity, ModuleTarget


LayerInterval = Tuple[int, int]


@dataclass(frozen=True)
class RecursiveLayerPartition(FusionGranularity):
    """A contiguous layer partition produced by recursive bisection."""

    num_layers: int
    intervals: Tuple[LayerInterval, ...]

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not self.intervals:
            raise ValueError("A layer partition requires at least one interval")
        cursor = 0
        for start, end in self.intervals:
            if start != cursor or end <= start:
                raise ValueError("Layer intervals must be contiguous and non-empty")
            cursor = end
        if cursor != self.num_layers:
            raise ValueError("Layer intervals must cover every model layer")

    @property
    def scope_names(self) -> Tuple[str, ...]:
        return tuple(
            "layers:{}-{}".format(start, end - 1)
            for start, end in self.intervals
        )

    def scope_for(self, target: ModuleTarget) -> str:
        layer = target.layer_index
        if layer is None:
            raise ValueError(
                "Recursive layer fusion requires a layer index for module {!r}".format(
                    target.module_path
                )
            )
        for scope, (start, end) in zip(self.scope_names, self.intervals):
            if start <= layer < end:
                return scope
        raise ValueError(
            "Layer index {} outside configured range 0..{}".format(
                layer, self.num_layers - 1
            )
        )

    def to_dict(self):
        return {
            "type": "recursive_layer_partition",
            "num_layers": self.num_layers,
            "intervals": [list(interval) for interval in self.intervals],
        }


def build_refinement_partitions(
    num_layers: int,
    group_counts: Sequence[int],
) -> Tuple[RecursiveLayerPartition, ...]:
    """Build nested partitions by repeatedly bisecting the largest interval."""

    counts = tuple(int(value) for value in group_counts)
    if not counts or counts[0] != 1:
        raise ValueError("refinement_groups must start with 1")
    if counts[-1] != num_layers:
        raise ValueError("refinement_groups must finish with num_layers")
    if any(left >= right for left, right in zip(counts, counts[1:])):
        raise ValueError("refinement_groups must be strictly increasing")
    if any(value < 1 or value > num_layers for value in counts):
        raise ValueError("refinement_groups must be within 1..num_layers")

    intervals = [(0, int(num_layers))]
    result = []
    for target_count in counts:
        while len(intervals) < target_count:
            splittable = [
                (end - start, -index, index)
                for index, (start, end) in enumerate(intervals)
                if end - start > 1
            ]
            if not splittable:
                raise ValueError("Cannot refine layer partition any further")
            _, _, index = max(splittable)
            start, end = intervals[index]
            middle = start + (end - start) // 2
            intervals[index : index + 1] = [
                (start, middle),
                (middle, end),
            ]
        result.append(
            RecursiveLayerPartition(num_layers, tuple(intervals))
        )
    return tuple(result)


__all__ = [
    "LayerInterval",
    "RecursiveLayerPartition",
    "build_refinement_partitions",
]
