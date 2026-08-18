"""Fusion coefficient representations, codecs, constraints, and repair."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .genome import FusionGenome, ScopedFusionGenome
from ..core.policies import (
    FusionGranularity,
    LayerGranularity,
    LayerGroupGranularity,
    LayerModuleGranularity,
    ModuleTypeGranularity,
)
from .contracts import EvolutionSearchSpace
from .contracts import SearchGenome


@dataclass(frozen=True)
class FusionSearchSpace(EvolutionSearchSpace):
    """Original Agent x Adapter sparse-simplex representation.

    The class now also implements the generic bounded-vector and policy-decoder
    contracts, so optimizers and runtimes no longer inspect ``FusionGenome``.
    """

    agent_names: Sequence[str]
    adapter_names: Sequence[str]
    min_active: int = 1
    max_active: int = 3
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        _validate_axes(
            self.agent_names,
            self.adapter_names,
            self.min_active,
            self.max_active,
        )

    @property
    def dimension(self) -> int:
        return len(self.agent_names) * len(self.adapter_names)

    @property
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((0.0, 1.0) for _ in range(self.dimension))

    @property
    def coefficient_groups(self) -> Tuple[Tuple[int, int], ...]:
        width = len(self.adapter_names)
        return tuple(
            (index * width, (index + 1) * width)
            for index in range(len(self.agent_names))
        )

    def sample(self, rng: random.Random) -> FusionGenome:
        return FusionGenome(
            {
                agent_name: self._sample_vector(rng)
                for agent_name in self.agent_names
            }
        )

    def uniform(self) -> FusionGenome:
        vector = _uniform_vector(len(self.adapter_names), self.max_active)
        return FusionGenome(
            {agent_name: list(vector) for agent_name in self.agent_names}
        )

    def base(self) -> FusionGenome:
        vector = [0.0] * len(self.adapter_names)
        return FusionGenome(
            {agent_name: list(vector) for agent_name in self.agent_names}
        )

    def one_hot(self, adapter_index: int) -> FusionGenome:
        vector = _one_hot_vector(len(self.adapter_names), adapter_index)
        return FusionGenome(
            {agent_name: list(vector) for agent_name in self.agent_names}
        )

    def role_one_hot(self, adapter_index: int, selected_agent: str) -> FusionGenome:
        if selected_agent not in self.agent_names:
            raise KeyError("Unknown Agent {!r}".format(selected_agent))
        one_hot = _one_hot_vector(len(self.adapter_names), adapter_index)
        zeros = [0.0] * len(self.adapter_names)
        return FusionGenome(
            {
                agent_name: list(one_hot if agent_name == selected_agent else zeros)
                for agent_name in self.agent_names
            }
        )

    def initial_genomes(self) -> Sequence[FusionGenome]:
        genomes = []
        if self.min_active == 0:
            genomes.append(self.base())
        if self.max_active >= 1:
            genomes.extend(
                self.one_hot(index) for index in range(len(self.adapter_names))
            )
        genomes.append(self.uniform())
        return tuple(genomes)

    def repair(
        self, genome: SearchGenome, rng: random.Random = None
    ) -> FusionGenome:
        if not isinstance(genome, FusionGenome):
            raise TypeError("FusionSearchSpace requires FusionGenome")
        local_rng = rng or random.Random(0)
        repaired = genome.clone()
        for agent_name in self.agent_names:
            raw = repaired.agent_weights.get(agent_name)
            if raw is None or len(raw) != len(self.adapter_names):
                raise ValueError(
                    "Invalid weight vector for agent {!r}".format(agent_name)
                )
            repaired.agent_weights[agent_name] = self.repair_vector(raw, local_rng)
        extra_agents = set(repaired.agent_weights) - set(self.agent_names)
        if extra_agents:
            raise ValueError(
                "Genome contains unknown agents: {}".format(
                    ", ".join(sorted(extra_agents))
                )
            )
        return repaired

    def repair_vector(
        self, values: Sequence[float], rng: random.Random
    ) -> List[float]:
        return _repair_simplex_vector(
            values,
            rng,
            min_active=self.min_active,
            max_active=self.max_active,
            epsilon=self.epsilon,
        )

    def encode(self, genome: SearchGenome) -> Tuple[float, ...]:
        if not isinstance(genome, FusionGenome):
            raise TypeError("FusionSearchSpace requires FusionGenome")
        return tuple(
            value
            for agent_name in self.agent_names
            for value in genome.agent_weights[agent_name]
        )

    def decode(
        self, values: Sequence[float], rng: random.Random = None
    ) -> FusionGenome:
        if len(values) != self.dimension:
            raise ValueError(
                "Encoded vector has {}, expected {}".format(
                    len(values), self.dimension
                )
            )
        width = len(self.adapter_names)
        genome = FusionGenome(
            {
                agent_name: list(values[index * width : (index + 1) * width])
                for index, agent_name in enumerate(self.agent_names)
            }
        )
        return self.repair(genome, rng)

    def seed_from_spec(self, specification: Mapping[str, object]) -> FusionGenome:
        unknown_agents = set(specification) - set(self.agent_names)
        if unknown_agents:
            raise ValueError(
                "Seed contains unknown Agents: {}".format(
                    ", ".join(sorted(unknown_agents))
                )
            )
        weights = {}
        for agent_name in self.agent_names:
            raw = specification.get(agent_name, {})
            if not isinstance(raw, Mapping):
                raise ValueError(
                    "Seed Agent {!r} must map adapter names to weights".format(
                        agent_name
                    )
                )
            weights[agent_name] = _validate_seed_vector(
                _named_adapter_vector(raw, self.adapter_names, agent_name),
                self.min_active,
                self.max_active,
                "Agent {!r}".format(agent_name),
            )
        return FusionGenome(weights)

    def _sample_vector(self, rng: random.Random) -> List[float]:
        return _sample_simplex_vector(
            len(self.adapter_names), self.min_active, self.max_active, rng
        )


@dataclass(frozen=True)
class ScopedFusionSearchSpace(EvolutionSearchSpace):
    """Agent x Scope x Adapter representation for arbitrary granularity."""

    agent_names: Sequence[str]
    adapter_names: Sequence[str]
    granularity: FusionGranularity
    min_active: int = 1
    max_active: int = 3
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        _validate_axes(
            self.agent_names,
            self.adapter_names,
            self.min_active,
            self.max_active,
        )
        if not self.granularity.scope_names:
            raise ValueError("granularity must expose at least one scope")

    @property
    def scope_names(self) -> Tuple[str, ...]:
        return self.granularity.scope_names

    @property
    def dimension(self) -> int:
        return (
            len(self.agent_names)
            * len(self.scope_names)
            * len(self.adapter_names)
        )

    @property
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((0.0, 1.0) for _ in range(self.dimension))

    @property
    def coefficient_groups(self) -> Tuple[Tuple[int, int], ...]:
        width = len(self.adapter_names)
        count = len(self.agent_names) * len(self.scope_names)
        return tuple(
            (index * width, (index + 1) * width) for index in range(count)
        )

    def sample(self, rng: random.Random) -> ScopedFusionGenome:
        return ScopedFusionGenome(
            {
                agent_name: {
                    scope_name: _sample_simplex_vector(
                        len(self.adapter_names),
                        self.min_active,
                        self.max_active,
                        rng,
                    )
                    for scope_name in self.scope_names
                }
                for agent_name in self.agent_names
            }
        )

    def uniform(self) -> ScopedFusionGenome:
        vector = _uniform_vector(len(self.adapter_names), self.max_active)
        return self._broadcast(vector)

    def base(self) -> ScopedFusionGenome:
        return self._broadcast([0.0] * len(self.adapter_names))

    def one_hot(self, adapter_index: int) -> ScopedFusionGenome:
        return self._broadcast(
            _one_hot_vector(len(self.adapter_names), adapter_index)
        )

    def role_one_hot(
        self, adapter_index: int, selected_agent: str
    ) -> ScopedFusionGenome:
        if selected_agent not in self.agent_names:
            raise KeyError("Unknown Agent {!r}".format(selected_agent))
        one_hot = _one_hot_vector(len(self.adapter_names), adapter_index)
        zeros = [0.0] * len(self.adapter_names)
        return ScopedFusionGenome(
            {
                agent_name: {
                    scope_name: list(
                        one_hot if agent_name == selected_agent else zeros
                    )
                    for scope_name in self.scope_names
                }
                for agent_name in self.agent_names
            }
        )

    def initial_genomes(self) -> Sequence[ScopedFusionGenome]:
        genomes = []
        if self.min_active == 0:
            genomes.append(self.base())
        if self.max_active >= 1:
            genomes.extend(
                self.one_hot(index) for index in range(len(self.adapter_names))
            )
        genomes.append(self.uniform())
        return tuple(genomes)

    def repair(
        self, genome: SearchGenome, rng: random.Random = None
    ) -> ScopedFusionGenome:
        if not isinstance(genome, ScopedFusionGenome):
            raise TypeError("ScopedFusionSearchSpace requires ScopedFusionGenome")
        local_rng = rng or random.Random(0)
        repaired = genome.clone()
        extra_agents = set(repaired.agent_scope_weights) - set(self.agent_names)
        if extra_agents:
            raise ValueError(
                "Genome contains unknown agents: {}".format(
                    ", ".join(sorted(extra_agents))
                )
            )
        for agent_name in self.agent_names:
            scopes = repaired.agent_scope_weights.get(agent_name)
            if scopes is None:
                raise ValueError("Missing Agent {!r}".format(agent_name))
            if set(scopes) != set(self.scope_names):
                raise ValueError(
                    "Agent {!r} scope names do not match granularity".format(
                        agent_name
                    )
                )
            repaired.agent_scope_weights[agent_name] = {
                scope_name: _repair_simplex_vector(
                    scopes[scope_name],
                    local_rng,
                    self.min_active,
                    self.max_active,
                    self.epsilon,
                )
                for scope_name in self.scope_names
            }
        return repaired

    def encode(self, genome: SearchGenome) -> Tuple[float, ...]:
        if not isinstance(genome, ScopedFusionGenome):
            raise TypeError("ScopedFusionSearchSpace requires ScopedFusionGenome")
        return tuple(
            value
            for agent_name in self.agent_names
            for scope_name in self.scope_names
            for value in genome.agent_scope_weights[agent_name][scope_name]
        )

    def decode(
        self, values: Sequence[float], rng: random.Random = None
    ) -> ScopedFusionGenome:
        if len(values) != self.dimension:
            raise ValueError(
                "Encoded vector has {}, expected {}".format(
                    len(values), self.dimension
                )
            )
        width = len(self.adapter_names)
        offset = 0
        agent_scopes = {}
        for agent_name in self.agent_names:
            scopes = {}
            for scope_name in self.scope_names:
                scopes[scope_name] = list(values[offset : offset + width])
                offset += width
            agent_scopes[agent_name] = scopes
        return self.repair(ScopedFusionGenome(agent_scopes), rng)

    def seed_from_spec(
        self, specification: Mapping[str, object]
    ) -> ScopedFusionGenome:
        unknown_agents = set(specification) - set(self.agent_names)
        if unknown_agents:
            raise ValueError(
                "Seed contains unknown Agents: {}".format(
                    ", ".join(sorted(unknown_agents))
                )
            )
        agent_scopes = {}
        for agent_name in self.agent_names:
            raw = specification.get(agent_name, {})
            if not isinstance(raw, Mapping):
                raise ValueError("Scoped seed Agent must be a mapping")
            is_scoped = bool(raw) and all(
                isinstance(value, Mapping) for value in raw.values()
            )
            if is_scoped:
                unknown_scopes = set(raw) - set(self.scope_names)
                if unknown_scopes:
                    raise ValueError(
                        "Seed contains unknown scopes: {}".format(
                            ", ".join(sorted(unknown_scopes))
                        )
                    )
                scopes = {
                    scope_name: _validate_seed_vector(
                        _named_adapter_vector(
                            raw.get(scope_name, {}),
                            self.adapter_names,
                            agent_name,
                        ),
                        self.min_active,
                        self.max_active,
                        "Agent {!r} scope {!r}".format(
                            agent_name, scope_name
                        ),
                    )
                    for scope_name in self.scope_names
                }
            else:
                vector = _validate_seed_vector(
                    _named_adapter_vector(
                        raw, self.adapter_names, agent_name
                    ),
                    self.min_active,
                    self.max_active,
                    "Agent {!r}".format(agent_name),
                )
                scopes = {
                    scope_name: list(vector) for scope_name in self.scope_names
                }
            agent_scopes[agent_name] = scopes
        return ScopedFusionGenome(agent_scopes)

    def _broadcast(self, vector: Sequence[float]) -> ScopedFusionGenome:
        return ScopedFusionGenome(
            {
                agent_name: {
                    scope_name: list(vector) for scope_name in self.scope_names
                }
                for agent_name in self.agent_names
            }
        )

    def broadcast_agent_weights(
        self, agent_weights: Mapping[str, Sequence[float]]
    ) -> ScopedFusionGenome:
        """Lift one global coefficient vector per Agent into every scope."""

        unknown_agents = set(agent_weights) - set(self.agent_names)
        if unknown_agents:
            raise ValueError(
                "Global seed contains unknown Agents: {}".format(
                    ", ".join(sorted(unknown_agents))
                )
            )
        genome = ScopedFusionGenome(
            {
                agent_name: {
                    scope_name: list(agent_weights[agent_name])
                    for scope_name in self.scope_names
                }
                for agent_name in self.agent_names
            }
        )
        return self.repair(genome)


class AgentSpecificScopedFusionSearchSpace(EvolutionSearchSpace):
    """Ragged Agent-specific Adapter coordinates over a global runtime axis.

    Evolution sees only the selected coordinates of each Agent. Structured
    genomes remain expanded onto the global Adapter axis so persistence and
    runtime translation stay backend-neutral.  The layout is constructed from
    arbitrary selected-index sets; it does not know whether they came from
    top-k screening, a threshold, a manual allow-list, or another selector.
    """

    def __init__(
        self,
        template: ScopedFusionSearchSpace,
        selected_adapter_indices: Mapping[str, Sequence[int]],
    ) -> None:
        self.agent_names = tuple(template.agent_names)
        self.adapter_names = tuple(template.adapter_names)
        self.granularity = template.granularity
        self.min_active = int(template.min_active)
        self.max_active = int(template.max_active)
        self.epsilon = float(template.epsilon)
        if set(selected_adapter_indices) != set(self.agent_names):
            raise ValueError(
                "Agent-specific Adapter layout must define every Agent exactly once"
            )
        width = len(self.adapter_names)
        normalized = {}
        for agent_name in self.agent_names:
            indices = tuple(int(index) for index in selected_adapter_indices[agent_name])
            if len(indices) != len(set(indices)):
                raise ValueError(
                    "Agent {!r} Adapter layout contains duplicates".format(agent_name)
                )
            if any(index < 0 or index >= width for index in indices):
                raise ValueError(
                    "Agent {!r} Adapter layout contains an out-of-range index".format(
                        agent_name
                    )
                )
            if self.min_active > len(indices):
                raise ValueError(
                    "Agent {!r} retains fewer Adapters than min_active".format(
                        agent_name
                    )
                )
            normalized[agent_name] = indices
        if not any(normalized.values()):
            raise ValueError("Compressed scoped search space has no coordinates")
        self._selected_adapter_indices = normalized

    @classmethod
    def from_masks(
        cls,
        template: ScopedFusionSearchSpace,
        masks: Mapping[str, Sequence[bool]],
    ) -> "AgentSpecificScopedFusionSearchSpace":
        expected_width = len(template.adapter_names)
        selected = {}
        if set(masks) != set(template.agent_names):
            raise ValueError("Pool masks must define every Agent exactly once")
        for agent_name in template.agent_names:
            mask = tuple(bool(value) for value in masks[agent_name])
            if len(mask) != expected_width:
                raise ValueError(
                    "Agent {!r} pool mask width does not match Adapter axis".format(
                        agent_name
                    )
                )
            selected[agent_name] = tuple(
                index for index, keep in enumerate(mask) if keep
            )
        return cls(template, selected)

    @property
    def scope_names(self) -> Tuple[str, ...]:
        return tuple(self.granularity.scope_names)

    def selected_adapter_indices(self, agent_name: str) -> Tuple[int, ...]:
        try:
            return self._selected_adapter_indices[agent_name]
        except KeyError as exc:
            raise KeyError("Unknown Agent {!r}".format(agent_name)) from exc

    def selected_adapter_names(self, agent_name: str) -> Tuple[str, ...]:
        return tuple(
            self.adapter_names[index]
            for index in self.selected_adapter_indices(agent_name)
        )

    @property
    def dimension(self) -> int:
        return len(self.scope_names) * sum(
            len(self.selected_adapter_indices(agent_name))
            for agent_name in self.agent_names
        )

    @property
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((0.0, 1.0) for _ in range(self.dimension))

    @property
    def coefficient_groups(self) -> Tuple[Tuple[int, int], ...]:
        groups = []
        offset = 0
        for agent_name in self.agent_names:
            local_width = len(self.selected_adapter_indices(agent_name))
            for _ in self.scope_names:
                groups.append((offset, offset + local_width))
                offset += local_width
        return tuple(groups)

    def sample(self, rng: random.Random) -> ScopedFusionGenome:
        rows = {}
        for agent_name in self.agent_names:
            indices = self.selected_adapter_indices(agent_name)
            rows[agent_name] = {
                scope_name: self._expand(
                    indices,
                    self._sample_local(len(indices), rng),
                )
                for scope_name in self.scope_names
            }
        return ScopedFusionGenome(rows)

    def uniform(self) -> ScopedFusionGenome:
        rows = {}
        for agent_name in self.agent_names:
            indices = self.selected_adapter_indices(agent_name)
            local = (
                _uniform_vector(len(indices), min(self.max_active, len(indices)))
                if indices
                else []
            )
            rows[agent_name] = {
                scope_name: self._expand(indices, local)
                for scope_name in self.scope_names
            }
        return ScopedFusionGenome(rows)

    def base(self) -> ScopedFusionGenome:
        return ScopedFusionGenome(
            {
                agent_name: {
                    scope_name: [0.0] * len(self.adapter_names)
                    for scope_name in self.scope_names
                }
                for agent_name in self.agent_names
            }
        )

    def one_hot(self, adapter_index: int) -> ScopedFusionGenome:
        if adapter_index < 0 or adapter_index >= len(self.adapter_names):
            raise IndexError("Adapter index is out of range")
        rows = {}
        for agent_name in self.agent_names:
            indices = self.selected_adapter_indices(agent_name)
            vector = [0.0] * len(self.adapter_names)
            if adapter_index in indices:
                vector[adapter_index] = 1.0
            rows[agent_name] = {
                scope_name: list(vector) for scope_name in self.scope_names
            }
        return ScopedFusionGenome(rows)

    def initial_genomes(self) -> Sequence[ScopedFusionGenome]:
        genomes = []
        if self.min_active == 0:
            genomes.append(self.base())
        if self.max_active >= 1:
            genomes.extend(
                self.one_hot(index) for index in range(len(self.adapter_names))
            )
        genomes.append(self.uniform())
        unique = {}
        for genome in genomes:
            unique.setdefault(genome.fingerprint(), genome)
        return tuple(unique.values())

    def repair(
        self, genome: SearchGenome, rng: random.Random = None
    ) -> ScopedFusionGenome:
        if not isinstance(genome, ScopedFusionGenome):
            raise TypeError(
                "AgentSpecificScopedFusionSearchSpace requires ScopedFusionGenome"
            )
        local_rng = rng or random.Random(0)
        if set(genome.agent_scope_weights) != set(self.agent_names):
            raise ValueError("Genome Agent axis does not match compressed layout")
        repaired = {}
        for agent_name in self.agent_names:
            scopes = genome.agent_scope_weights[agent_name]
            if set(scopes) != set(self.scope_names):
                raise ValueError(
                    "Agent {!r} scope axis does not match granularity".format(
                        agent_name
                    )
                )
            indices = self.selected_adapter_indices(agent_name)
            repaired_scopes = {}
            for scope_name in self.scope_names:
                row = scopes[scope_name]
                if len(row) != len(self.adapter_names):
                    raise ValueError("Expanded Adapter row width mismatch")
                local = [row[index] for index in indices]
                repaired_local = self._repair_local(local, local_rng)
                repaired_scopes[scope_name] = self._expand(
                    indices, repaired_local
                )
            repaired[agent_name] = repaired_scopes
        return ScopedFusionGenome(repaired)

    def encode(self, genome: SearchGenome) -> Tuple[float, ...]:
        repaired = self.repair(genome)
        if repaired.fingerprint() != genome.fingerprint():
            raise ValueError(
                "Genome is outside the Agent-specific compressed search space"
            )
        return tuple(
            repaired.agent_scope_weights[agent_name][scope_name][index]
            for agent_name in self.agent_names
            for scope_name in self.scope_names
            for index in self.selected_adapter_indices(agent_name)
        )

    def decode(
        self, values: Sequence[float], rng: random.Random = None
    ) -> ScopedFusionGenome:
        if len(values) != self.dimension:
            raise ValueError(
                "Compressed vector has {}, expected {}".format(
                    len(values), self.dimension
                )
            )
        offset = 0
        rows = {}
        for agent_name in self.agent_names:
            indices = self.selected_adapter_indices(agent_name)
            width = len(indices)
            scopes = {}
            for scope_name in self.scope_names:
                local = list(values[offset : offset + width])
                offset += width
                scopes[scope_name] = self._expand(indices, local)
            rows[agent_name] = scopes
        return self.repair(ScopedFusionGenome(rows), rng)

    def seed_from_spec(
        self, specification: Mapping[str, object]
    ) -> ScopedFusionGenome:
        template = ScopedFusionSearchSpace(
            self.agent_names,
            self.adapter_names,
            self.granularity,
            self.min_active,
            self.max_active,
            self.epsilon,
        )
        return self.repair(template.seed_from_spec(specification))

    def broadcast_agent_weights(
        self, agent_weights: Mapping[str, Sequence[float]]
    ) -> ScopedFusionGenome:
        if set(agent_weights) != set(self.agent_names):
            raise ValueError("Global seed Agent axis does not match compressed layout")
        rows = {}
        for agent_name in self.agent_names:
            vector = list(agent_weights[agent_name])
            if len(vector) != len(self.adapter_names):
                raise ValueError("Global seed Adapter axis width mismatch")
            rows[agent_name] = {
                scope_name: list(vector) for scope_name in self.scope_names
            }
        return self.repair(ScopedFusionGenome(rows))

    def _sample_local(self, width: int, rng: random.Random) -> List[float]:
        if width == 0:
            return []
        return _sample_simplex_vector(
            width,
            self.min_active,
            min(self.max_active, width),
            rng,
        )

    def _repair_local(
        self, values: Sequence[float], rng: random.Random
    ) -> List[float]:
        if not values:
            return []
        return _repair_simplex_vector(
            values,
            rng,
            min_active=self.min_active,
            max_active=min(self.max_active, len(values)),
            epsilon=self.epsilon,
        )

    def _expand(
        self, indices: Sequence[int], local_values: Sequence[float]
    ) -> List[float]:
        if len(indices) != len(local_values):
            raise ValueError("Local Adapter vector width mismatch")
        expanded = [0.0] * len(self.adapter_names)
        for index, value in zip(indices, local_values):
            expanded[index] = float(value)
        return expanded


SearchSpaceBuilder = Callable[[Any, int, Any], EvolutionSearchSpace]


class FusionSearchSpaceRegistry:
    """Register bounded coefficient representations for vector optimizers."""

    def __init__(self) -> None:
        self._builders = {}  # type: Dict[str, SearchSpaceBuilder]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._builders))

    def register(
        self,
        name: str,
        builder: SearchSpaceBuilder,
        *,
        replace: bool = False,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Search-space name cannot be empty")
        if normalized in self._builders and not replace:
            raise ValueError(
                "Search space {!r} is already registered".format(normalized)
            )
        self._builders[normalized] = builder

    def create(
        self,
        name: str,
        config: Any,
        stage_index: int = 0,
        settings: Any = None,
    ) -> EvolutionSearchSpace:
        normalized = name.strip().lower()
        try:
            builder = self._builders[normalized]
        except KeyError:
            raise ValueError(
                "Unknown coefficient representation {!r}; registered: {}".format(
                    name, ", ".join(self.names) or "<none>"
                )
            )
        return builder(config, stage_index, settings)


def create_default_search_space_registry() -> FusionSearchSpaceRegistry:
    registry = FusionSearchSpaceRegistry()
    registry.register("global", _build_global_space)
    registry.register("adapter", _build_global_space)
    registry.register(
        "layer",
        lambda config, stage_index, settings: _build_scoped_space(
            config, "layer", stage_index, settings
        ),
    )
    registry.register(
        "layer_group",
        lambda config, stage_index, settings: _build_scoped_space(
            config, "layer_group", stage_index, settings
        ),
    )
    registry.register(
        "module_type",
        lambda config, stage_index, settings: _build_scoped_space(
            config, "module_type", stage_index, settings
        ),
    )
    registry.register(
        "layer_module",
        lambda config, stage_index, settings: _build_scoped_space(
            config, "layer_module", stage_index, settings
        ),
    )
    return registry


def _build_global_space(
    config: Any, stage_index: int = 0, settings: Any = None
) -> FusionSearchSpace:
    settings = _resolve_space_settings(config, stage_index, settings)
    return FusionSearchSpace(
        agent_names=config.agent_names,
        adapter_names=config.adapter_names,
        min_active=settings.min_active_adapters,
        max_active=settings.max_active_adapters,
    )


def _build_scoped_space(
    config: Any, kind: str, stage_index: int = 0, settings: Any = None
) -> ScopedFusionSearchSpace:
    settings = _resolve_space_settings(config, stage_index, settings)
    options = dict(settings.representation_options)
    num_layers = None
    module_types = None
    if kind in ("layer", "layer_group", "layer_module"):
        num_layers = _resolve_num_layers(config, options)
    if kind in ("module_type", "layer_module"):
        configured = options.get("module_types")
        if configured is None:
            configured = _common_target_modules(config.adapters)
        module_types = tuple(str(item) for item in configured)

    if kind == "layer":
        granularity = LayerGranularity(num_layers)
    elif kind == "layer_group":
        group_size = options.get("group_size")
        if group_size is None:
            group_count = int(options.get("group_count", 6))
            if group_count < 1:
                raise ValueError("group_count must be positive")
            group_size = (num_layers + group_count - 1) // group_count
        granularity = LayerGroupGranularity(num_layers, int(group_size))
    elif kind == "module_type":
        granularity = ModuleTypeGranularity(module_types)
    elif kind == "layer_module":
        granularity = LayerModuleGranularity(num_layers, module_types)
    else:  # pragma: no cover - registry controls this input
        raise ValueError("Unknown scoped representation {!r}".format(kind))

    return ScopedFusionSearchSpace(
        agent_names=config.agent_names,
        adapter_names=config.adapter_names,
        granularity=granularity,
        min_active=settings.min_active_adapters,
        max_active=settings.max_active_adapters,
    )


@dataclass(frozen=True)
class _DefaultSpaceSettings:
    """Optimizer-neutral defaults for direct registry use.

    Normal experiment assembly always supplies the owning optimizer's resolved
    settings.  Keeping this small fallback here lets representation tests and
    third-party callers construct a space without making the shared module
    import NSGA-II or DE configuration classes.
    """

    min_active_adapters: int
    max_active_adapters: int
    representation_options: Mapping[str, Any]


def _resolve_space_settings(
    config: Any, stage_index: int, settings: Any
) -> Any:
    if settings is not None:
        return settings
    if not 0 <= stage_index < config.search.stage_count:
        raise ValueError("Search-space stage index is out of range")
    return _DefaultSpaceSettings(
        min_active_adapters=0,
        max_active_adapters=len(config.adapters),
        representation_options=dict(config.search.granularity_options),
    )


def _resolve_num_layers(config: Any, options: Mapping[str, Any]) -> int:
    explicit = options.get("num_layers")
    if explicit is not None:
        value = int(explicit)
        if value < 1:
            raise ValueError("representation_options.num_layers must be positive")
        return value
    model_path = Path(config.model.name_or_path)
    config_path = model_path / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if model_config.get(key) is not None:
                return int(model_config[key])
    raise ValueError(
        "Layer granularity requires representation_options.num_layers when "
        "the model's local config.json is unavailable"
    )


def _common_target_modules(adapters: Sequence[Any]) -> Tuple[str, ...]:
    if not adapters:
        raise ValueError("Cannot infer module_types from an empty Adapter pool")
    first = tuple(adapters[0].target_modules)
    if not first:
        raise ValueError(
            "module_type granularity requires representation_options.module_types "
            "or Adapter target_modules metadata"
        )
    first_set = set(first)
    for adapter in adapters[1:]:
        if set(adapter.target_modules) != first_set:
            raise ValueError(
                "Cannot infer one module_type axis from heterogeneous target_modules"
            )
    return first


def _validate_axes(
    agent_names: Sequence[str],
    adapter_names: Sequence[str],
    min_active: int,
    max_active: int,
) -> None:
    if not agent_names:
        raise ValueError("agent_names cannot be empty")
    if not adapter_names:
        raise ValueError("adapter_names cannot be empty")
    if min_active < 0:
        raise ValueError("min_active cannot be negative")
    if max_active < min_active:
        raise ValueError("max_active must be >= min_active")
    if max_active > len(adapter_names):
        raise ValueError("max_active cannot exceed adapter count")


def _sample_simplex_vector(
    width: int, min_active: int, max_active: int, rng: random.Random
) -> List[float]:
    active_count = rng.randint(min_active, max_active)
    vector = [0.0] * width
    if active_count == 0:
        return vector
    active = rng.sample(range(width), active_count)
    raw = [rng.gammavariate(1.0, 1.0) for _ in active]
    total = sum(raw)
    for index, value in zip(active, raw):
        vector[index] = value / total
    return vector


def _repair_simplex_vector(
    values: Sequence[float],
    rng: random.Random,
    min_active: int,
    max_active: int,
    epsilon: float,
) -> List[float]:
    vector = [max(0.0, float(value)) for value in values]
    vector = [value if value >= epsilon else 0.0 for value in vector]
    active = [index for index, value in enumerate(vector) if value > 0.0]
    if len(active) > max_active:
        keep = set(
            sorted(active, key=lambda index: vector[index], reverse=True)[
                :max_active
            ]
        )
        vector = [
            value if index in keep else 0.0
            for index, value in enumerate(vector)
        ]
        active = sorted(keep)
    inactive = [index for index in range(len(vector)) if index not in set(active)]
    while len(active) < min_active:
        if not inactive:
            break
        selected = rng.choice(inactive)
        inactive.remove(selected)
        active.append(selected)
        vector[selected] = rng.uniform(0.05, 1.0)
    total = sum(vector)
    if total <= epsilon:
        if min_active == 0:
            return [0.0] * len(vector)
        selected = rng.randrange(len(vector))
        vector[selected] = 1.0
        total = 1.0
    return [value / total for value in vector]


def _uniform_vector(width: int, active: int) -> List[float]:
    if active == 0:
        return [0.0] * width
    return [1.0 / active if index < active else 0.0 for index in range(width)]


def _one_hot_vector(width: int, adapter_index: int) -> List[float]:
    if not 0 <= adapter_index < width:
        raise IndexError("adapter_index out of range")
    vector = [0.0] * width
    vector[adapter_index] = 1.0
    return vector


def _named_adapter_vector(
    raw: Mapping[str, object],
    adapter_names: Sequence[str],
    agent_name: str,
) -> List[float]:
    unknown = set(raw) - set(adapter_names)
    if unknown:
        raise ValueError(
            "Seed Agent {!r} contains unknown adapters: {}".format(
                agent_name, ", ".join(sorted(unknown))
            )
        )
    vector = [float(raw.get(name, 0.0)) for name in adapter_names]
    if any(value < 0.0 for value in vector):
        raise ValueError("Seed weights cannot be negative")
    return vector


def _validate_seed_vector(
    vector: Sequence[float],
    min_active: int,
    max_active: int,
    context: str,
) -> List[float]:
    active = sum(value > 0.0 for value in vector)
    if not min_active <= active <= max_active:
        raise ValueError(
            "{} activates {}, expected {}..{}".format(
                context, active, min_active, max_active
            )
        )
    total = sum(vector)
    if total > 0.0:
        return [float(value) / total for value in vector]
    return [0.0 for _ in vector]


__all__ = [
    "AgentSpecificScopedFusionSearchSpace",
    "FusionSearchSpace",
    "FusionSearchSpaceRegistry",
    "ScopedFusionSearchSpace",
    "create_default_search_space_registry",
]
