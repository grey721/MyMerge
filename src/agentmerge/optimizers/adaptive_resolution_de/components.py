"""Assemble unchanged MOEA/D-DE operators for every resolution space."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from ...core.config import ExperimentConfig
from ..de.components import DEComponents
from ..de.decomposition import create_decomposition
from ..de.survival import DecompositionUpdate
from ..de.variation import DifferentialVariation
from ..initialization import create_initial_genomes
from ..representation import FusionSearchSpace, ScopedFusionSearchSpace
from ..translation import create_plan_translator
from .config import AdaptiveResolutionConfig
from .hierarchy import ResolutionManager


def build_adaptive_resolution_components(
    experiment: ExperimentConfig,
    settings: AdaptiveResolutionConfig,
    resolution: ResolutionManager,
) -> Tuple[DEComponents, ...]:
    stages = []
    for stage_index, groups in enumerate(resolution.stages):
        stage_settings = replace(
            settings.de,
            seed=settings.de.seed + stage_index,
            representation=("adapter" if stage_index == 0 else "layer_partition"),
        )
        if stage_index == 0:
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
                resolution.recursive_partition(stage_index),
                stage_settings.min_active_adapters,
                stage_settings.max_active_adapters,
            )
        stages.append(
            DEComponents(
                name="groups_{}".format(len(groups)),
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
                    if stage_index == 0
                    else ()
                ),
            )
        )
    return tuple(stages)


__all__ = ["build_adaptive_resolution_components"]
