"""Configuration owned by the decomposition-based DE optimizer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional

from ...core.config import ConfigError, ExperimentConfig


@dataclass(frozen=True)
class DEConfig:
    population_size: int
    generations: int
    workers: int
    seed: int
    min_active_adapters: int
    max_active_adapters: int
    representation: str
    representation_options: Mapping[str, Any]
    differential_weight: float = 0.5
    crossover_probability: float = 1.0
    mutation_probability: Optional[float] = None
    mutation_distribution_index: float = 20.0
    neighborhood_size: int = 3
    neighborhood_mating_probability: float = 0.9
    max_replacements: int = 2
    decomposition: str = "tchebycheff"
    pbi_theta: float = 5.0

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ConfigError("DE population_size must be at least 4")
        if self.generations < 1:
            raise ConfigError("DE generations must be positive")
        if self.workers < 1:
            raise ConfigError("DE workers must be positive")
        if self.min_active_adapters < 0:
            raise ConfigError("DE min_active_adapters cannot be negative")
        if self.max_active_adapters < self.min_active_adapters:
            raise ConfigError(
                "DE max_active_adapters must be >= min_active_adapters"
            )
        if not 0.0 <= self.differential_weight <= 2.0:
            raise ConfigError("DE differential_weight must be in [0, 2]")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ConfigError("DE crossover_probability must be in [0, 1]")
        if self.mutation_probability is not None and not (
            0.0 <= self.mutation_probability <= 1.0
        ):
            raise ConfigError("DE mutation_probability must be in [0, 1]")
        if self.mutation_distribution_index <= 0.0:
            raise ConfigError(
                "DE mutation_distribution_index must be positive"
            )
        if not 2 <= self.neighborhood_size <= self.population_size:
            raise ConfigError(
                "DE neighborhood_size must be between 2 and population_size"
            )
        if not 0.0 <= self.neighborhood_mating_probability <= 1.0:
            raise ConfigError(
                "DE neighborhood_mating_probability must be in [0, 1]"
            )
        if self.max_replacements < 1:
            raise ConfigError("DE max_replacements must be positive")
        if self.decomposition not in ("tchebycheff", "weighted_sum", "pbi"):
            raise ConfigError("Unknown DE decomposition {!r}".format(self.decomposition))
        if self.pbi_theta <= 0.0:
            raise ConfigError("DE pbi_theta must be positive")

    @classmethod
    def from_experiment_config(
        cls,
        experiment: ExperimentConfig,
        stage_index: int = 0,
        option_namespace: str = "de",
    ) -> "DEConfig":
        search = experiment.search
        if not 0 <= stage_index < search.stage_count:
            raise ConfigError("DE stage index is out of range")
        unknown_namespaces = sorted(set(search.options) - {option_namespace})
        if unknown_namespaces:
            raise ConfigError(
                "Unknown search.options namespace(s) for DE: {}".format(
                    ", ".join(unknown_namespaces)
                )
            )
        raw = search.options.get(option_namespace, {})
        if not isinstance(raw, Mapping):
            raise ConfigError(
                "search.options.{} must be a mapping".format(option_namespace)
            )
        options = dict(raw)
        allowed = {
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
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ConfigError("Unknown DE options: {}".format(", ".join(unknown)))

        population_size = search.population_size_for(stage_index)
        default_neighborhood = min(
            population_size,
            max(3, int(math.ceil(population_size / 10.0))),
        )
        mutation_probability = options.get("mutation_probability")
        settings = cls(
            population_size=population_size,
            generations=search.generations_for(stage_index),
            workers=search.workers,
            seed=search.seed,
            min_active_adapters=int(options.get("min_active_adapters", 0)),
            max_active_adapters=int(
                options.get("max_active_adapters", len(experiment.adapters))
            ),
            representation=search.granularities[stage_index],
            representation_options=dict(search.granularity_options),
            differential_weight=_float_option(
                options, "differential_weight", 0.5
            ),
            crossover_probability=_float_option(
                options, "crossover_probability", 1.0
            ),
            mutation_probability=(
                None
                if mutation_probability is None
                else _float_option(options, "mutation_probability", 0.0)
            ),
            mutation_distribution_index=_float_option(
                options, "mutation_distribution_index", 20.0
            ),
            neighborhood_size=int(
                options.get("neighborhood_size", default_neighborhood)
            ),
            neighborhood_mating_probability=_float_option(
                options, "neighborhood_mating_probability", 0.9
            ),
            max_replacements=int(options.get("max_replacements", 2)),
            decomposition=str(
                options.get("decomposition", "tchebycheff")
            ).lower(),
            pbi_theta=_float_option(options, "pbi_theta", 5.0),
        )
        if settings.max_active_adapters > len(experiment.adapters):
            raise ConfigError(
                "DE max_active_adapters cannot exceed Adapter pool size"
            )
        return settings


def _float_option(
    options: Mapping[str, Any], name: str, default: float
) -> float:
    try:
        return float(options.get(name, default))
    except (TypeError, ValueError):
        raise ConfigError("DE {} must be numeric".format(name))


__all__ = ["DEConfig"]
