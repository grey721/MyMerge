"""Assembly contract for one decomposition-based DE stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import DEConfig
from .survival import PopulationUpdateStrategy
from .variation import DEVariation
from ..contracts import (
    EvolutionSearchSpace,
    FusionPlanTranslator,
    SearchGenome,
)


@dataclass(frozen=True)
class DEComponents:
    name: str
    settings: DEConfig
    search_space: EvolutionSearchSpace
    translator: FusionPlanTranslator
    variation: DEVariation
    update_strategy: PopulationUpdateStrategy
    initial_genomes: Tuple[SearchGenome, ...] = ()


__all__ = ["DEComponents"]
