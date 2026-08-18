"""Shared fusion genomes and the evaluated NSGA-II individual."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.domain import AdapterMixture, EvaluationResult
from .contracts import SearchGenome


@dataclass
class FusionGenome:
    """One adapter coefficient vector per agent."""

    agent_weights: Dict[str, List[float]]

    def clone(self) -> "FusionGenome":
        return FusionGenome(
            agent_weights={
                agent_name: list(weights)
                for agent_name, weights in self.agent_weights.items()
            }
        )

    def mixture_for(
        self, agent_name: str, adapter_names: Sequence[str]
    ) -> AdapterMixture:
        if agent_name not in self.agent_weights:
            raise KeyError("No weights configured for agent {!r}".format(agent_name))
        weights = self.agent_weights[agent_name]
        if len(weights) != len(adapter_names):
            raise ValueError(
                "Agent {!r} has {} weights, expected {}".format(
                    agent_name, len(weights), len(adapter_names)
                )
            )
        return AdapterMixture(tuple(adapter_names), tuple(weights))

    def fingerprint(self, precision: int = 10) -> str:
        payload = {
            agent_name: [round(value, precision) for value in weights]
            for agent_name, weights in sorted(self.agent_weights.items())
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            agent_name: list(weights)
            for agent_name, weights in self.agent_weights.items()
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Sequence[float]]) -> "FusionGenome":
        return cls(
            agent_weights={
                str(agent_name): [float(item) for item in weights]
                for agent_name, weights in value.items()
            }
        )


@dataclass
class ScopedFusionGenome:
    """One Adapter vector per Agent and coefficient scope."""

    agent_scope_weights: Dict[str, Dict[str, List[float]]]

    def clone(self) -> "ScopedFusionGenome":
        return ScopedFusionGenome(
            agent_scope_weights={
                agent_name: {
                    scope_name: list(weights)
                    for scope_name, weights in scopes.items()
                }
                for agent_name, scopes in self.agent_scope_weights.items()
            }
        )


    def fingerprint(self, precision: int = 10) -> str:
        payload = {
            agent_name: {
                scope_name: [round(value, precision) for value in weights]
                for scope_name, weights in sorted(scopes.items())
            }
            for agent_name, scopes in sorted(self.agent_scope_weights.items())
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def to_dict(self) -> Dict[str, Dict[str, List[float]]]:
        return {
            agent_name: {
                scope_name: list(weights)
                for scope_name, weights in scopes.items()
            }
            for agent_name, scopes in self.agent_scope_weights.items()
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Mapping[str, Sequence[float]]]
    ) -> "ScopedFusionGenome":
        return cls(
            agent_scope_weights={
                str(agent_name): {
                    str(scope_name): [float(item) for item in weights]
                    for scope_name, weights in scopes.items()
                }
                for agent_name, scopes in value.items()
            }
        )


@dataclass
class NSGA2Individual:
    genome: SearchGenome
    fitness: Optional[Tuple[float, ...]] = None
    evaluation: Optional[EvaluationResult] = None
    rank: int = 0
    crowding_distance: float = 0.0

    @property
    def evaluated(self) -> bool:
        return self.fitness is not None and self.evaluation is not None

    def clone(self) -> "NSGA2Individual":
        return NSGA2Individual(
            genome=self.genome.clone(),
            fitness=self.fitness,
            evaluation=self.evaluation,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
        )


__all__ = ["FusionGenome", "NSGA2Individual", "ScopedFusionGenome"]
