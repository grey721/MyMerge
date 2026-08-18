"""Local configuration schema for the NSGA-II package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...core.config import ConfigError, ExperimentConfig


@dataclass(frozen=True)
class NSGA2Config:
    """Fully resolved NSGA-II settings.

    Shared run-budget values are copied from :class:`SearchConfig`. Genome
    representation, fusion granularity, repair constraints, seeds and genetic
    operator settings are all owned and defaulted by this package.
    """

    population_size: int
    generations: int
    workers: int
    seed: int
    min_active_adapters: int
    max_active_adapters: int
    representation: str
    representation_options: Mapping[str, Any]
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    sparsity_mutation_probability: float = 0.15
    mutation_sigma: float = 0.15

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ConfigError("NSGA-II population_size must be at least 4")
        if self.generations < 1:
            raise ConfigError("NSGA-II generations must be positive")
        if self.workers < 1:
            raise ConfigError("NSGA-II workers must be positive")
        if self.min_active_adapters < 0:
            raise ConfigError("NSGA-II min_active_adapters cannot be negative")
        if self.max_active_adapters < self.min_active_adapters:
            raise ConfigError(
                "NSGA-II max_active_adapters must be >= min_active_adapters"
            )
        if not self.representation:
            raise ConfigError("NSGA-II representation name cannot be empty")
        for name in (
            "crossover_probability",
            "mutation_probability",
            "sparsity_mutation_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(
                    "NSGA-II {} must be in [0, 1]".format(name)
                )
        if self.mutation_sigma < 0.0:
            raise ConfigError("NSGA-II mutation_sigma cannot be negative")

    @classmethod
    def from_experiment_config(
        cls,
        experiment: ExperimentConfig,
        stage_index: int = 0,
        option_namespace: str = "nsga2",
    ) -> "NSGA2Config":
        search = experiment.search
        if not 0 <= stage_index < search.stage_count:
            raise ConfigError("NSGA-II stage index is out of range")
        raw_options = search.options
        unknown_namespaces = sorted(set(raw_options) - {option_namespace})
        if unknown_namespaces:
            raise ConfigError(
                "Unknown search.options namespace(s) for NSGA-II: {}".format(
                    ", ".join(unknown_namespaces)
                )
            )
        nested = raw_options.get(option_namespace, {})
        if not isinstance(nested, Mapping):
            raise ConfigError(
                "search.options.{} must be a mapping".format(option_namespace)
            )
        options = dict(nested)

        allowed = {
            "crossover_probability",
            "mutation_probability",
            "sparsity_mutation_probability",
            "mutation_sigma",
            "min_active_adapters",
            "max_active_adapters",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ConfigError(
                "Unknown NSGA-II options: {}".format(", ".join(unknown))
            )
        settings = cls(
            population_size=search.population_size_for(stage_index),
            generations=search.generations_for(stage_index),
            workers=search.workers,
            seed=search.seed,
            min_active_adapters=int(options.get("min_active_adapters", 0)),
            max_active_adapters=int(
                options.get(
                    "max_active_adapters",
                    len(experiment.adapters),
                )
            ),
            representation=search.granularities[stage_index],
            representation_options=dict(search.granularity_options),
            crossover_probability=_float_option(
                options, "crossover_probability", 0.9
            ),
            mutation_probability=_float_option(
                options, "mutation_probability", 0.2
            ),
            sparsity_mutation_probability=_float_option(
                options, "sparsity_mutation_probability", 0.15
            ),
            mutation_sigma=_float_option(options, "mutation_sigma", 0.15),
        )
        if settings.max_active_adapters > len(experiment.adapters):
            raise ConfigError(
                "NSGA-II max_active_adapters cannot exceed Adapter pool size"
            )
        return settings


def _float_option(
    options: Mapping[str, Any], name: str, default: float
) -> float:
    try:
        return float(options.get(name, default))
    except (TypeError, ValueError):
        raise ConfigError("NSGA-II {} must be numeric".format(name))


__all__ = ["NSGA2Config"]
