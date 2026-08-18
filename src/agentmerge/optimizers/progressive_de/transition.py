"""Pure chromosome state transfer between progressive representations."""

from __future__ import annotations

from typing import Sequence, Tuple

from ...core.policies import ModuleTarget
from ..contracts import EvolutionSearchSpace, SearchGenome
from ..de.genome import DEIndividual
from ..genome import FusionGenome, ScopedFusionGenome
from ..representation import FusionSearchSpace, ScopedFusionSearchSpace


def project_to_refined_space(
    genome: SearchGenome,
    source_space: EvolutionSearchSpace,
    target_space: ScopedFusionSearchSpace,
) -> ScopedFusionGenome:
    """Expand parent coefficient rows over their child layer intervals.

    This changes only the chromosome representation. Every target layer keeps
    exactly the same Adapter coefficients, so the represented model behavior
    and its already measured fitness remain unchanged.
    """

    if isinstance(source_space, FusionSearchSpace):
        if not isinstance(genome, FusionGenome):
            raise TypeError("Global refinement requires FusionGenome")
        # Do not call target_space.repair(): the source chromosome is already
        # feasible and even a harmless simplex renormalization can change its
        # floating representation. Promotion must be an exact coordinate copy.
        return ScopedFusionGenome(
            {
                agent_name: {
                    scope_name: list(genome.agent_weights[agent_name])
                    for scope_name in target_space.scope_names
                }
                for agent_name in target_space.agent_names
            }
        )
    if not isinstance(source_space, ScopedFusionSearchSpace):
        raise TypeError("Progressive refinement requires a scoped source space")
    if not isinstance(genome, ScopedFusionGenome):
        raise TypeError("Scoped refinement requires ScopedFusionGenome")

    rows = {}
    target_intervals = getattr(target_space.granularity, "intervals", None)
    if target_intervals is None:
        raise TypeError("Target space is not a recursive layer partition")
    for agent_name in target_space.agent_names:
        source_rows = genome.agent_scope_weights[agent_name]
        rows[agent_name] = {}
        for target_scope, (start, _) in zip(
            target_space.scope_names, target_intervals
        ):
            parent_scope = source_space.granularity.scope_for(
                ModuleTarget(layer_index=start)
            )
            rows[agent_name][target_scope] = list(source_rows[parent_scope])
    # As above, preserve the optimizer variables bit-for-bit. Every target row
    # is copied from a feasible source row, so another repair is unnecessary.
    return ScopedFusionGenome(rows)


def transfer_population(
    population: Sequence[DEIndividual],
    source_space: EvolutionSearchSpace,
    target_space: ScopedFusionSearchSpace,
    expected_size: int,
) -> Tuple[DEIndividual, ...]:
    """Rebuild a complete evaluated population in a refined chromosome.

    There is deliberately no selection, perturbation or evaluation here.
    Standard DE owns all subsequent variation after this one-to-one state
    transfer.
    """

    if len(population) != expected_size:
        raise ValueError(
            "Progressive population transfer has {}, expected {}".format(
                len(population), expected_size
            )
        )
    transferred = []
    for index, individual in enumerate(population):
        if not individual.evaluated:
            raise ValueError(
                "Progressive population item {} is not evaluated".format(index)
            )
        transferred.append(
            DEIndividual(
                genome=project_to_refined_space(
                    individual.genome, source_space, target_space
                ),
                fitness=individual.fitness,
                evaluation=individual.evaluation,
                rank=individual.rank,
                crowding_distance=individual.crowding_distance,
            )
        )
    return tuple(transferred)


__all__ = ["project_to_refined_space", "transfer_population"]
