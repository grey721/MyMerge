"""Reusable assembly contract exported by the single-stage NSGA-II package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import NSGA2Config
from ..contracts import EvolutionSearchSpace, FusionPlanTranslator, SearchGenome
from .variation import NSGA2Variation


@dataclass(frozen=True)
class NSGA2Components:
    """One fully assembled NSGA-II stage.

    Two-stage NSGA-II reuses this public package boundary instead of reaching
    into selection, variation, chromosome, or representation internals.
    """

    name: str
    settings: NSGA2Config
    search_space: EvolutionSearchSpace
    translator: FusionPlanTranslator
    variation: NSGA2Variation
    initial_genomes: Tuple[SearchGenome, ...] = ()


__all__ = ["NSGA2Components"]
