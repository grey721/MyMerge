"""Configuration owned by progressive recursive-refinement DE."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Tuple

from ...core.config import ConfigError, ExperimentConfig
from ..de.config import DEConfig
from .partition import build_refinement_partitions


_COMMON_DE_OPTIONS = {
    "differential_weight",
    "crossover_probability",
    "mutation_probability",
    "mutation_distribution_index",
    "neighborhood_size",
    "neighborhood_mating_probability",
    "max_replacements",
    "decomposition",
    "pbi_theta",
    "min_active_adapters",
    "max_active_adapters",
}


@dataclass(frozen=True)
class ProgressiveDEConfig:
    num_layers: int
    refinement_groups: Tuple[int, ...]
    stages: Tuple[DEConfig, ...]

    @classmethod
    def from_experiment_config(
        cls,
        experiment: ExperimentConfig,
        option_namespace: str = "progressive_de",
    ) -> "ProgressiveDEConfig":
        search = experiment.search
        unknown_namespaces = sorted(set(search.options) - {option_namespace})
        if unknown_namespaces:
            raise ConfigError(
                "Unknown search.options namespace(s) for progressive DE: {}".format(
                    ", ".join(unknown_namespaces)
                )
            )
        raw = search.options.get(option_namespace, {})
        if not isinstance(raw, Mapping):
            raise ConfigError("search.options.progressive_de must be a mapping")
        options = dict(raw)
        allowed = _COMMON_DE_OPTIONS | {"refinement_groups"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ConfigError(
                "Unknown progressive DE options: {}".format(", ".join(unknown))
            )
        configured_layers = search.granularity_options.get("num_layers")
        if configured_layers is None:
            raise ConfigError("progressive_de requires --num_layers")
        num_layers = int(configured_layers)
        groups = tuple(int(value) for value in options.get("refinement_groups", ()))
        if not groups:
            raise ConfigError("progressive_de requires --refinement_groups")
        try:
            build_refinement_partitions(num_layers, groups)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if search.stage_count != len(groups):
            raise ConfigError(
                "Progressive DE stage count must match refinement_groups"
            )
        expected = ("adapter",) + ("layer_partition",) * (len(groups) - 1)
        if tuple(search.granularities) != expected:
            raise ConfigError(
                "Progressive DE owns its stage representations; expected {}".format(
                    expected
                )
            )

        common = {key: options[key] for key in _COMMON_DE_OPTIONS if key in options}
        sanitized_search = replace(
            search,
            options={option_namespace: common},
        )
        sanitized = replace(experiment, search=sanitized_search)
        stages = tuple(
            DEConfig.from_experiment_config(
                sanitized,
                stage_index=index,
                option_namespace=option_namespace,
            )
            for index in range(len(groups))
        )
        population_sizes = {
            stage.population_size for stage in stages
        }
        if len(population_sizes) != 1:
            raise ConfigError(
                "progressive_de requires one constant population size across "
                "all stages for one-to-one population state transfer"
            )
        return cls(
            num_layers=num_layers,
            refinement_groups=groups,
            stages=stages,
        )


__all__ = ["ProgressiveDEConfig"]
