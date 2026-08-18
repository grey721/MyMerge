"""Adapter-to-layer two-stage decomposition-based DE."""

from __future__ import annotations

from typing import List, Mapping, Sequence

from ...evaluation.benchmarks import FusionPlanEvaluator
from ..base import Optimizer, SearchObserver, SearchResult, StageObserver
from ..de.components import DEComponents
from ..de.optimizer import DecompositionDEOptimizer
from ..fitness import FitnessSpace
from ..translation import project_genome


class TwoStageDEOptimizer(Optimizer):
    def __init__(
        self,
        evaluator: FusionPlanEvaluator,
        fitness_space: FitnessSpace,
        stages: Sequence[DEComponents],
        observer: SearchObserver,
    ) -> None:
        if len(stages) != 2:
            raise ValueError("TwoStageDEOptimizer requires exactly two stages")
        self.evaluator = evaluator
        self.fitness_space = fitness_space
        self.stages = tuple(stages)
        self.observer = observer

    def optimize(self) -> SearchResult:
        previous_result = None
        previous_space = None
        total_unique = 0
        generation_offset = 0
        combined_history: List[Mapping[str, object]] = []

        for stage_index, stage in enumerate(self.stages):
            projected = list(stage.initial_genomes)
            if previous_result is not None and previous_space is not None:
                projected = [
                    project_genome(
                        individual.genome,
                        previous_space,
                        stage.search_space,
                    )
                    for individual in _select_anchors(
                        previous_result.pareto_front,
                        stage.settings.population_size,
                    )
                ] + projected

            self.observer.on_event(
                "search_stage_started",
                {
                    "stage_index": stage_index,
                    "stage_count": 2,
                    "stage": stage.name,
                    "population_size": stage.settings.population_size,
                    "generations": stage.settings.generations,
                    "dimension": stage.search_space.dimension,
                    "translator": type(stage.translator).__name__,
                    "initial_genome_count": len(stage.initial_genomes),
                    "projected_seed_count": len(projected)
                    - len(stage.initial_genomes),
                },
            )
            stage_observer = StageObserver(
                self.observer,
                stage_index,
                2,
                stage.name,
                generation_offset,
                total_unique,
            )
            optimizer = DecompositionDEOptimizer(
                evaluator=self.evaluator,
                fitness_space=self.fitness_space,
                search_space=stage.search_space,
                translator=stage.translator,
                variation=stage.variation,
                update_strategy=stage.update_strategy,
                population_size=stage.settings.population_size,
                generations=stage.settings.generations,
                neighborhood_size=stage.settings.neighborhood_size,
                neighborhood_mating_probability=(
                    stage.settings.neighborhood_mating_probability
                ),
                seed=stage.settings.seed + stage_index,
                workers=stage.settings.workers,
                observer=stage_observer,
                initial_genomes=projected,
            )
            result = optimizer.optimize()
            for item in result.history:
                adjusted = dict(item)
                adjusted["stage"] = stage.name
                adjusted["stage_index"] = stage_index
                adjusted["stage_generation"] = int(item["generation"])
                adjusted["generation"] = generation_offset + int(
                    item["generation"]
                )
                adjusted["unique_evaluations"] = (
                    total_unique + int(item["unique_evaluations"])
                )
                combined_history.append(adjusted)
            total_unique += result.unique_evaluations
            generation_offset += result.generations_completed
            self.observer.on_event(
                "search_stage_completed",
                {
                    "stage_index": stage_index,
                    "stage_count": 2,
                    "stage": stage.name,
                    "stage_unique_evaluations": result.unique_evaluations,
                    "unique_evaluations": total_unique,
                    "pareto_size": len(result.pareto_front),
                },
            )
            previous_result = result
            previous_space = stage.search_space

        assert previous_result is not None
        return SearchResult(
            population=previous_result.population,
            pareto_front=previous_result.pareto_front,
            generations_completed=generation_offset,
            unique_evaluations=total_unique,
            history=tuple(combined_history),
        )


def _select_anchors(population, limit: int):
    """Retain objective extremes and diverse Pareto anchors for projection."""

    return tuple(
        sorted(
            population,
            key=lambda item: (
                -item.crowding_distance,
                item.genome.fingerprint(),
            ),
        )[:limit]
    )


__all__ = ["TwoStageDEOptimizer"]
