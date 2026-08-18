"""Build standard MOEA/D-DE stages over recursively refined spaces."""

from __future__ import annotations

from typing import Tuple

from ...core.config import ExperimentConfig
from ..de.components import DEComponents
from ..de.decomposition import create_decomposition
from ..de.survival import DecompositionUpdate
from ..de.variation import DifferentialVariation
from ..initialization import create_initial_genomes
from ..representation import FusionSearchSpace, ScopedFusionSearchSpace
from ..translation import create_plan_translator
from .config import ProgressiveDEConfig
from .partition import build_refinement_partitions


def build_progressive_components(
    experiment: ExperimentConfig,
    settings: ProgressiveDEConfig,
) -> Tuple[DEComponents, ...]:
    partitions = build_refinement_partitions(
        settings.num_layers, settings.refinement_groups
    )
    stages = []
    for index, (group_count, stage_settings) in enumerate(
        zip(settings.refinement_groups, settings.stages)
    ):
        if index == 0:
            search_space = FusionSearchSpace(
                experiment.agent_names,
                experiment.adapter_names,
                stage_settings.min_active_adapters,
                stage_settings.max_active_adapters,
            )
        else:
            search_space = ScopedFusionSearchSpace(
                experiment.agent_names,
                experiment.adapter_names,
                partitions[index],
                stage_settings.min_active_adapters,
                stage_settings.max_active_adapters,
            )
        stages.append(
            DEComponents(
                name="groups_{}".format(group_count),
                settings=stage_settings,
                search_space=search_space,
                translator=create_plan_translator(search_space, experiment),
                variation=DifferentialVariation(
                    search_space,
                    stage_settings.differential_weight,
                    stage_settings.crossover_probability,
                    stage_settings.mutation_probability,
                    stage_settings.mutation_distribution_index,
                ),
                update_strategy=DecompositionUpdate(
                    create_decomposition(
                        stage_settings.decomposition,
                        pbi_theta=stage_settings.pbi_theta,
                    ),
                    max_replacements=stage_settings.max_replacements,
                ),
                initial_genomes=(
                    tuple(create_initial_genomes(experiment, search_space))
                    if index == 0
                    else ()
                ),
            )
        )
    return tuple(stages)


__all__ = ["build_progressive_components"]
