"""Chromosome-to-phenotype translators for fusion search stages."""

from __future__ import annotations

from typing import Any, Sequence

from ..core.policies import FusionPlan, ScopedAdapterFusionPolicy
from .contracts import EvolutionSearchSpace, FusionPlanTranslator, SearchGenome
from .genome import FusionGenome, ScopedFusionGenome
from .representation import (
    AgentSpecificScopedFusionSearchSpace,
    FusionSearchSpace,
    ScopedFusionSearchSpace,
)


class GlobalFusionPlanTranslator:
    """Translate Agent × Adapter chromosomes into global fusion policies."""

    def __init__(
        self, agent_names: Sequence[str], adapter_names: Sequence[str]
    ) -> None:
        self.agent_names = tuple(agent_names)
        self.adapter_names = tuple(adapter_names)

    def plan_for(self, genome: SearchGenome) -> FusionPlan:
        if not isinstance(genome, FusionGenome):
            raise TypeError("Global translator requires FusionGenome")
        return FusionPlan(
            tuple(
                (
                    agent_name,
                    genome.mixture_for(agent_name, self.adapter_names),
                )
                for agent_name in self.agent_names
            )
        )


class ScopedFusionPlanTranslator:
    """Translate Agent × Scope × Adapter chromosomes into scoped policies."""

    def __init__(
        self,
        search_space: Any,
    ) -> None:
        self.agent_names = tuple(search_space.agent_names)
        self.adapter_names = tuple(search_space.adapter_names)
        self.granularity = search_space.granularity
        self.scope_names = tuple(search_space.scope_names)

    def plan_for(self, genome: SearchGenome) -> FusionPlan:
        if not isinstance(genome, ScopedFusionGenome):
            raise TypeError("Scoped translator requires ScopedFusionGenome")
        policies = []
        for agent_name in self.agent_names:
            scopes = genome.agent_scope_weights[agent_name]
            policy = ScopedAdapterFusionPolicy(
                adapter_names=self.adapter_names,
                granularity=self.granularity,
                scope_weights=tuple(
                    (scope_name, tuple(scopes[scope_name]))
                    for scope_name in self.scope_names
                ),
            )
            policies.append((agent_name, policy))
        return FusionPlan(tuple(policies))


def create_plan_translator(
    search_space: EvolutionSearchSpace,
    config: Any = None,
) -> FusionPlanTranslator:
    """Create the translator for a resolved search space.

    Translators own phenotype construction and never call back into
    ``EvolutionSearchSpace``. A future codec can therefore keep its chromosome
    private without changing an optimizer or evaluator.
    """

    del config
    if isinstance(search_space, FusionSearchSpace):
        return GlobalFusionPlanTranslator(
            search_space.agent_names, search_space.adapter_names
        )
    if isinstance(
        search_space,
        (ScopedFusionSearchSpace, AgentSpecificScopedFusionSearchSpace),
    ):
        return ScopedFusionPlanTranslator(search_space)
    raise TypeError(
        "No FusionPlan translator registered for {}".format(
            type(search_space).__name__
        )
    )


def project_genome(
    genome: SearchGenome,
    source_space: EvolutionSearchSpace,
    target_space: EvolutionSearchSpace,
) -> SearchGenome:
    """Project a global chromosome into every scope of a staged search."""

    if isinstance(source_space, FusionSearchSpace) and isinstance(
        target_space, ScopedFusionSearchSpace
    ):
        if not isinstance(genome, FusionGenome):
            raise TypeError("Global projection requires FusionGenome")
        return target_space.broadcast_agent_weights(genome.agent_weights)
    raise ValueError(
        "Unsupported staged transition: {} -> {}".format(
            type(source_space).__name__, type(target_space).__name__
        )
    )


__all__ = [
    "GlobalFusionPlanTranslator",
    "ScopedFusionPlanTranslator",
    "create_plan_translator",
    "project_genome",
]
