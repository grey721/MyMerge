"""Central configuration for budget-aware adaptive-resolution MOEA/D-DE."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Tuple

from ...core.config import ConfigError, ExperimentConfig
from ..de.config import DEConfig
from .hierarchy import build_nested_layer_groups


_DE_OPTIONS = {
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

_ADAPTIVE_OPTIONS = {
    "total_budget",
    "min_stage_fe",
    "first_split",
    "num_sectors",
    "sector_seed",
    "direction_window",
    "min_directions_per_sector",
    "min_valid_sector_fraction",
    "direction_norm_eps",
    "coherence_threshold",
    "utility_ratio_threshold",
    "saturation_patience",
    "utility_ema_beta",
    "utility_eps",
    "checkpoint_interval_fe",
    "enable_zero_progress_fallback",
    "utility_zero_eps",
    "symmetry_diagnostic_batches",
}


@dataclass(frozen=True)
class AdaptiveResolutionConfig:
    num_layers: int
    total_budget: int
    min_stage_fe: Tuple[int, ...]
    de: DEConfig
    first_split: int = 4
    num_sectors: int = 5
    sector_seed: int = 0
    direction_window: int = 8
    min_directions_per_sector: int = 4
    min_valid_sector_fraction: float = 0.5
    direction_norm_eps: float = 1e-12
    coherence_threshold: float = 0.0
    utility_ratio_threshold: float = 0.2
    saturation_patience: int = 2
    utility_ema_beta: float = 0.8
    utility_eps: float = 1e-12
    checkpoint_interval_fe: int = 1
    enable_zero_progress_fallback: bool = True
    utility_zero_eps: float = 1e-12
    symmetry_diagnostic_batches: int = 3

    @property
    def stage_count(self) -> int:
        return len(self.min_stage_fe)

    @classmethod
    def from_experiment_config(
        cls,
        experiment: ExperimentConfig,
        option_namespace: str = "adaptive_resolution_de",
    ) -> "AdaptiveResolutionConfig":
        search = experiment.search
        unknown_namespaces = sorted(set(search.options) - {option_namespace})
        if unknown_namespaces:
            raise ConfigError(
                "Unknown search.options namespace(s) for adaptive-resolution "
                "DE: {}".format(", ".join(unknown_namespaces))
            )
        raw = search.options.get(option_namespace, {})
        if not isinstance(raw, Mapping):
            raise ConfigError(
                "search.options.adaptive_resolution_de must be a mapping"
            )
        options = dict(raw)
        unknown = sorted(set(options) - _DE_OPTIONS - _ADAPTIVE_OPTIONS)
        if unknown:
            raise ConfigError(
                "Unknown adaptive-resolution DE options: {}".format(
                    ", ".join(unknown)
                )
            )
        configured_layers = search.granularity_options.get("num_layers")
        if configured_layers is None:
            raise ConfigError("adaptive_resolution_de requires --num_layers")
        num_layers = int(configured_layers)
        first_split = _int_option(options, "first_split", 4)
        try:
            hierarchy = build_nested_layer_groups(num_layers, first_split)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

        if "total_budget" not in options:
            raise ConfigError("adaptive_resolution_de requires --total_budget")
        total_budget = _int_option(options, "total_budget", 0)
        raw_minimums = options.get("min_stage_fe")
        if not isinstance(raw_minimums, (list, tuple)):
            raise ConfigError(
                "adaptive_resolution_de requires --min_stage_fe for every stage"
            )
        min_stage_fe = tuple(int(value) for value in raw_minimums)
        if len(min_stage_fe) != len(hierarchy):
            raise ConfigError(
                "min_stage_fe has {} entries, hierarchy requires {} for group "
                "counts {}".format(
                    len(min_stage_fe),
                    len(hierarchy),
                    [len(stage) for stage in hierarchy],
                )
            )
        if total_budget < 1:
            raise ConfigError("total_budget must be positive")
        if any(value < 0 for value in min_stage_fe):
            raise ConfigError("min_stage_fe values cannot be negative")
        if sum(min_stage_fe) > total_budget:
            raise ConfigError("sum(min_stage_fe) cannot exceed total_budget")
        if min_stage_fe[-1] < 1:
            raise ConfigError("The full-layer stage must reserve at least one FE")

        sanitized_search = replace(
            search,
            granularities=("adapter",),
            stage_population_sizes=(),
            stage_generations=(),
            options={
                option_namespace: {
                    key: options[key] for key in _DE_OPTIONS if key in options
                }
            },
        )
        sanitized = replace(experiment, search=sanitized_search)
        de = DEConfig.from_experiment_config(
            sanitized,
            stage_index=0,
            option_namespace=option_namespace,
        )
        # Adaptive stages terminate by real FE budget, not by a fixed YAML
        # generation count. This high safety ceiling only detects pathological
        # streams that produce no novel phenotype indefinitely.
        de = replace(de, generations=max(de.generations, total_budget + 1))
        if min_stage_fe[0] < de.population_size:
            raise ConfigError(
                "min_stage_fe[0] must be at least population size {} because "
                "the initial population has no inherited fitness".format(
                    de.population_size
                )
            )

        settings = cls(
            num_layers=num_layers,
            total_budget=total_budget,
            min_stage_fe=min_stage_fe,
            de=de,
            first_split=first_split,
            num_sectors=_int_option(options, "num_sectors", 5),
            sector_seed=_int_option(options, "sector_seed", 0),
            direction_window=_int_option(options, "direction_window", 8),
            min_directions_per_sector=_int_option(
                options, "min_directions_per_sector", 4
            ),
            min_valid_sector_fraction=_float_option(
                options, "min_valid_sector_fraction", 0.5
            ),
            direction_norm_eps=_float_option(
                options, "direction_norm_eps", 1e-12
            ),
            coherence_threshold=_float_option(
                options, "coherence_threshold", 0.0
            ),
            utility_ratio_threshold=_float_option(
                options, "utility_ratio_threshold", 0.2
            ),
            saturation_patience=_int_option(
                options, "saturation_patience", 2
            ),
            utility_ema_beta=_float_option(
                options, "utility_ema_beta", 0.8
            ),
            utility_eps=_float_option(options, "utility_eps", 1e-12),
            checkpoint_interval_fe=_int_option(
                options, "checkpoint_interval_fe", de.population_size
            ),
            enable_zero_progress_fallback=_bool_option(
                options, "enable_zero_progress_fallback", True
            ),
            utility_zero_eps=_float_option(
                options, "utility_zero_eps", 1e-12
            ),
            symmetry_diagnostic_batches=_int_option(
                options, "symmetry_diagnostic_batches", 3
            ),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.num_sectors < 1:
            raise ConfigError("num_sectors must be positive")
        if self.direction_window < 2:
            raise ConfigError("direction_window must be at least 2")
        if not 2 <= self.min_directions_per_sector <= self.direction_window:
            raise ConfigError(
                "min_directions_per_sector must be within 2..direction_window"
            )
        if not 0.0 < self.min_valid_sector_fraction <= 1.0:
            raise ConfigError("min_valid_sector_fraction must be in (0, 1]")
        if self.direction_norm_eps <= 0.0 or self.utility_eps <= 0.0:
            raise ConfigError("direction_norm_eps and utility_eps must be positive")
        if not -1.0 <= self.coherence_threshold <= 1.0:
            raise ConfigError("coherence_threshold must be in [-1, 1]")
        if not 0.0 <= self.utility_ratio_threshold <= 1.0:
            raise ConfigError("utility_ratio_threshold must be in [0, 1]")
        if self.saturation_patience < 1:
            raise ConfigError("saturation_patience must be positive")
        if not 0.0 <= self.utility_ema_beta < 1.0:
            raise ConfigError("utility_ema_beta must be in [0, 1)")
        if self.checkpoint_interval_fe < 1:
            raise ConfigError("checkpoint_interval_fe must be positive")
        if self.utility_zero_eps < 0.0:
            raise ConfigError("utility_zero_eps cannot be negative")
        if self.symmetry_diagnostic_batches < 1:
            raise ConfigError("symmetry_diagnostic_batches must be positive")


def _int_option(options: Mapping[str, Any], name: str, default: int) -> int:
    try:
        return int(options.get(name, default))
    except (TypeError, ValueError):
        raise ConfigError("{} must be an integer".format(name))


def _float_option(options: Mapping[str, Any], name: str, default: float) -> float:
    try:
        value = float(options.get(name, default))
    except (TypeError, ValueError):
        raise ConfigError("{} must be numeric".format(name))
    if not math.isfinite(value):
        raise ConfigError("{} must be finite".format(name))
    return value


def _bool_option(options: Mapping[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ConfigError("{} must be boolean".format(name))


__all__ = ["AdaptiveResolutionConfig"]
