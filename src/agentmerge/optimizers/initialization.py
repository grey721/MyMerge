"""Shared admission-aware initial anchor generation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..core.config import ExperimentConfig
from .contracts import EvolutionSearchSpace, SearchGenome


class AdmissionAwareInitializer:
    """Build role-wise one-hot seeds from audited Adapter metadata.

    ``role_admission: critic_positive`` seeds that Adapter on ``critic``.
    Labels such as ``solver_and_critic_positive`` use the first declared role,
    giving every admitted Adapter at most one deterministic seed and avoiding
    an initialization population dominated by duplicate Adapter identities.
    Adapters without positive admission metadata do not create special seeds.
    Generic base/one-hot/uniform baselines remain owned by the search space.
    """

    def build(
        self,
        config: ExperimentConfig,
        search_space: EvolutionSearchSpace,
    ) -> Tuple[SearchGenome, ...]:
        agent_names = set(config.agent_names)
        specifications = []
        for adapter in config.adapters:
            label = str(
                adapter.metadata.get("role_admission", "")
            ).strip().lower()
            suffix = "_positive"
            if not label.endswith(suffix):
                continue
            roles = label[: -len(suffix)].split("_and_")
            selected_role = next(
                (role for role in roles if role in agent_names), None
            )
            if selected_role is None:
                continue
            specifications.append(
                {selected_role: {adapter.name: 1.0}}
            )

        genomes = []  # type: List[SearchGenome]
        seen = set()
        for specification in specifications:
            genome = search_space.seed_from_spec(specification)
            fingerprint = genome.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                genomes.append(genome)
        return tuple(genomes)


def create_initial_genomes(
    config: ExperimentConfig,
    search_space: EvolutionSearchSpace,
) -> Sequence[SearchGenome]:
    """Run the default package-owned initialization policy."""

    return AdmissionAwareInitializer().build(config, search_space)


__all__ = ["AdmissionAwareInitializer", "create_initial_genomes"]
