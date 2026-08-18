"""Nested LoRA-fusion resolution spaces and lossless state promotion.

The hierarchy forms nested search spaces ``X_0 subset ... subset X_S``.
Coarse spaces are easier for black-box search, while the final singleton-layer
space has full expressiveness. Promotion copies each parent coefficient into
all child groups, embedding the exact same phenotype in the finer space. It
therefore changes neither objectives nor the function-evaluation counter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..contracts import EvolutionSearchSpace, SearchGenome
from ..de.genome import DEIndividual
from ..progressive_de.partition import RecursiveLayerPartition
from ..progressive_de.transition import project_to_refined_space
from ..representation import ScopedFusionSearchSpace


@dataclass(frozen=True)
class LayerGroup:
    group_id: int
    start_layer: int
    end_layer: int
    parent_group_id: Optional[int]

    @property
    def size(self) -> int:
        return self.end_layer - self.start_layer

    @property
    def interval(self) -> Tuple[int, int]:
        return (self.start_layer, self.end_layer)


def build_nested_layer_groups(
    num_layers: int,
    first_split: int = 4,
) -> Tuple[Tuple[LayerGroup, ...], ...]:
    """Build root -> up-to-``first_split`` -> repeated-bisection stages."""

    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if first_split < 2:
        raise ValueError("first_split must be at least 2")
    stages = ((LayerGroup(0, 0, int(num_layers), None),),)
    if num_layers == 1:
        return stages

    first_intervals = _even_intervals(0, num_layers, min(first_split, num_layers))
    current = tuple(
        LayerGroup(index, start, end, 0)
        for index, (start, end) in enumerate(first_intervals)
    )
    stages += (current,)
    while any(group.size > 1 for group in current):
        children = []
        for parent in current:
            intervals = (
                _even_intervals(parent.start_layer, parent.end_layer, 2)
                if parent.size > 1
                else (parent.interval,)
            )
            for start, end in intervals:
                children.append(
                    LayerGroup(
                        len(children), start, end, parent.group_id
                    )
                )
        current = tuple(children)
        stages += (current,)
    return stages


def _even_intervals(
    start: int,
    end: int,
    count: int,
) -> Tuple[Tuple[int, int], ...]:
    length = end - start
    if not 1 <= count <= length:
        raise ValueError("Cannot split interval into the requested group count")
    base, remainder = divmod(length, count)
    intervals = []
    cursor = start
    for index in range(count):
        width = base + (1 if index < remainder else 0)
        intervals.append((cursor, cursor + width))
        cursor += width
    return tuple(intervals)


class ResolutionManager:
    """Own resolution geometry, compact vectors, and zero-FE promotion."""

    def __init__(
        self,
        num_layers: int,
        agent_adapter_counts: Mapping[str, int],
        first_split: int = 4,
    ) -> None:
        if not agent_adapter_counts:
            raise ValueError("At least one Agent is required")
        if any(int(value) < 1 for value in agent_adapter_counts.values()):
            raise ValueError("Every Agent must expose at least one Adapter")
        self.num_layers = int(num_layers)
        self.agent_adapter_counts = {
            str(name): int(count)
            for name, count in agent_adapter_counts.items()
        }
        self.stages = build_nested_layer_groups(num_layers, first_split)
        self._stage = 0

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def partition(self) -> Tuple[LayerGroup, ...]:
        return self.stages[self._stage]

    @property
    def group_count(self) -> int:
        return len(self.partition)

    @property
    def is_final(self) -> bool:
        return self._stage == self.stage_count - 1

    @property
    def dimension(self) -> int:
        return self.dimension_for(self._stage)

    def dimension_for(self, stage: int) -> int:
        return len(self.stages[stage]) * sum(self.agent_adapter_counts.values())

    def recursive_partition(self, stage: int) -> RecursiveLayerPartition:
        groups = self.stages[stage]
        return RecursiveLayerPartition(
            self.num_layers,
            tuple(group.interval for group in groups),
        )

    def promote_vector(self, values: Sequence[float]) -> Tuple[float, ...]:
        if self.is_final:
            raise RuntimeError("Full layer-wise resolution cannot be promoted")
        source = self.stages[self._stage]
        target = self.stages[self._stage + 1]
        expected = self.dimension_for(self._stage)
        if len(values) != expected:
            raise ValueError(
                "Source vector has {}, expected {}".format(len(values), expected)
            )
        source_offsets = self._coordinate_offsets(source)
        promoted = []
        for agent_name, adapter_count in self.agent_adapter_counts.items():
            for child in target:
                parent_id = child.parent_group_id
                if parent_id is None:
                    raise ValueError("Refined group is missing its parent")
                base = source_offsets[(agent_name, parent_id)]
                promoted.extend(values[base : base + adapter_count])
        return tuple(float(value) for value in promoted)

    def decode_to_full_layer_coefficients(
        self,
        values: Sequence[float],
        stage: Optional[int] = None,
    ) -> Mapping[str, Tuple[Tuple[float, ...], ...]]:
        stage_index = self._stage if stage is None else int(stage)
        groups = self.stages[stage_index]
        expected = self.dimension_for(stage_index)
        if len(values) != expected:
            raise ValueError(
                "Decision vector has {}, expected {}".format(len(values), expected)
            )
        offsets = self._coordinate_offsets(groups)
        decoded: Dict[str, Tuple[Tuple[float, ...], ...]] = {}
        for agent_name, adapter_count in self.agent_adapter_counts.items():
            layers = []
            for layer in range(self.num_layers):
                group = next(
                    item
                    for item in groups
                    if item.start_layer <= layer < item.end_layer
                )
                offset = offsets[(agent_name, group.group_id)]
                layers.append(
                    tuple(
                        float(value)
                        for value in values[offset : offset + adapter_count]
                    )
                )
            decoded[agent_name] = tuple(layers)
        return decoded

    def structural_direction(
        self,
        displacement: Sequence[float],
        bounds: Sequence[Tuple[float, float]],
    ) -> Tuple[float, ...]:
        """Map compact displacement norm to full-layer coefficient geometry."""

        if len(displacement) != self.dimension or len(bounds) != self.dimension:
            raise ValueError("Direction and bounds must match current dimension")
        weights = self.structural_weights(bounds)
        return tuple(
            float(value) * weight
            for value, weight in zip(displacement, weights)
        )

    def structural_weights(
        self,
        bounds: Sequence[Tuple[float, float]],
    ) -> Tuple[float, ...]:
        if len(bounds) != self.dimension:
            raise ValueError("Bounds must match current decision dimension")
        weights = []
        index = 0
        for _, adapter_count in self.agent_adapter_counts.items():
            for group in self.partition:
                for _ in range(adapter_count):
                    lower, upper = bounds[index]
                    width = float(upper) - float(lower)
                    if width <= 0.0:
                        raise ValueError("Direction normalization requires open bounds")
                    weights.append(math.sqrt(group.size) / width)
                    index += 1
        return tuple(weights)

    def promote_genome(
        self,
        genome: SearchGenome,
        source_space: EvolutionSearchSpace,
        target_space: ScopedFusionSearchSpace,
    ) -> SearchGenome:
        return project_to_refined_space(genome, source_space, target_space)

    def promote_individuals(
        self,
        individuals: Sequence[DEIndividual],
        source_space: EvolutionSearchSpace,
        target_space: ScopedFusionSearchSpace,
    ) -> Tuple[DEIndividual, ...]:
        promoted = []
        for index, individual in enumerate(individuals):
            if not individual.evaluated:
                raise ValueError(
                    "Cannot promote unevaluated individual {}".format(index)
                )
            promoted.append(
                DEIndividual(
                    genome=self.promote_genome(
                        individual.genome, source_space, target_space
                    ),
                    fitness=individual.fitness,
                    evaluation=individual.evaluation,
                    rank=individual.rank,
                    crowding_distance=individual.crowding_distance,
                )
            )
        return tuple(promoted)

    def sibling_coordinate_groups(self) -> Tuple[Tuple[int, ...], ...]:
        """Return newly replicated coordinate sets after the next promotion."""

        if self.is_final:
            return ()
        children = self.stages[self._stage + 1]
        target_offsets = self._coordinate_offsets(children)
        by_parent: Dict[int, list] = {}
        for child in children:
            assert child.parent_group_id is not None
            by_parent.setdefault(child.parent_group_id, []).append(child)
        result = []
        for agent_name, adapter_count in self.agent_adapter_counts.items():
            for child_groups in by_parent.values():
                if len(child_groups) < 2:
                    continue
                for adapter_index in range(adapter_count):
                    result.append(
                        tuple(
                            target_offsets[(agent_name, child.group_id)]
                            + adapter_index
                            for child in child_groups
                        )
                    )
        return tuple(result)

    def advance(self) -> None:
        if self.is_final:
            raise RuntimeError("Cannot advance beyond full layer resolution")
        self._stage += 1

    def _coordinate_offsets(
        self,
        groups: Sequence[LayerGroup],
    ) -> Mapping[Tuple[str, int], int]:
        offsets = {}
        cursor = 0
        for agent_name, adapter_count in self.agent_adapter_counts.items():
            for group in groups:
                offsets[(agent_name, group.group_id)] = cursor
                cursor += adapter_count
        return offsets


class SymmetryReleaseDiagnostic:
    """Observe whether normal DE variation separates replicated children."""

    def __init__(
        self,
        coordinate_groups: Sequence[Sequence[int]],
        max_batches: int,
        tolerance: float = 1e-12,
    ) -> None:
        self.coordinate_groups = tuple(tuple(group) for group in coordinate_groups)
        self.max_batches = int(max_batches)
        self.tolerance = float(tolerance)
        self.batches_observed = 0
        self.diverged = not self.coordinate_groups

    @property
    def complete(self) -> bool:
        return self.diverged or self.batches_observed >= self.max_batches

    def observe(self, vectors: Sequence[Sequence[float]]) -> bool:
        if self.complete:
            return self.diverged
        self.batches_observed += 1
        for vector in vectors:
            for group in self.coordinate_groups:
                values = [float(vector[index]) for index in group]
                if max(values) - min(values) > self.tolerance:
                    self.diverged = True
                    return True
        return False


__all__ = [
    "LayerGroup",
    "ResolutionManager",
    "SymmetryReleaseDiagnostic",
    "build_nested_layer_groups",
]
