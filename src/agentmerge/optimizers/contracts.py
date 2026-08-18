"""Algorithm-neutral genome, bounded search-space, and translation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any, Mapping, Protocol, Sequence, Tuple, runtime_checkable

from ..core.policies import FusionPlan


@runtime_checkable
class SearchGenome(Protocol):
    """Opaque genome contract shared by compatible optimizer packages."""

    def clone(self) -> "SearchGenome":
        ...

    def fingerprint(self, precision: int = 10) -> str:
        ...

    def to_dict(self) -> Mapping[str, Any]:
        ...


@runtime_checkable
class FusionPlanTranslator(Protocol):
    """Translate an opaque chromosome into an evaluator-facing phenotype.

    Search operators never need to know whether coefficients are global,
    layer-scoped, or module-scoped.  A pipeline can therefore replace this
    translator when it changes fusion granularity between stages.
    """

    def plan_for(self, genome: SearchGenome) -> FusionPlan:
        ...


class EvolutionSearchSpace(ABC):
    """Encode structured genomes as bounded real vectors.

    Population algorithms may use the structured methods directly. Vector-based
    ES and DE implementations can use ``encode``/``decode`` and ``bounds`` while
    sharing the same repair and runtime decoding semantics.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        pass

    @property
    @abstractmethod
    def coefficient_groups(self) -> Tuple[Tuple[int, int], ...]:
        """Half-open vector ranges that form independently repaired groups."""

    @abstractmethod
    def sample(self, rng: random.Random) -> SearchGenome:
        pass

    @abstractmethod
    def repair(
        self, genome: SearchGenome, rng: random.Random = None
    ) -> SearchGenome:
        pass

    @abstractmethod
    def encode(self, genome: SearchGenome) -> Tuple[float, ...]:
        pass

    @abstractmethod
    def decode(
        self, values: Sequence[float], rng: random.Random = None
    ) -> SearchGenome:
        pass

    @abstractmethod
    def initial_genomes(self) -> Sequence[SearchGenome]:
        """Return representation-specific deterministic baselines."""

    @abstractmethod
    def seed_from_spec(self, specification: Mapping[str, object]) -> SearchGenome:
        """Parse one configuration seed without optimizer-specific knowledge."""

__all__ = ["EvolutionSearchSpace", "FusionPlanTranslator", "SearchGenome"]
