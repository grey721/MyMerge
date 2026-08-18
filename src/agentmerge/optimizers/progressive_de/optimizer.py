"""Progressive coarse-to-fine decomposition-based DE."""

from __future__ import annotations

from typing import List, Mapping, Sequence

from ...evaluation.benchmarks import FusionPlanEvaluator
from ...evaluation.cache import CandidateEvaluationService
from ..base import Optimizer, SearchObserver, SearchResult, StageObserver
from ..de.components import DEComponents
from ..de.optimizer import DecompositionDEOptimizer
from ..fitness import FitnessSpace
from ..representation import ScopedFusionSearchSpace
from .config import ProgressiveDEConfig
from .transition import transfer_population


class ProgressiveDEOptimizer(Optimizer):
    """Release layer-wise coefficient freedom through nested bisection stages."""

    def __init__(
        self,
        evaluator: FusionPlanEvaluator,
        fitness_space: FitnessSpace,
        stages: Sequence[DEComponents],
        settings: ProgressiveDEConfig,
        observer: SearchObserver,
    ) -> None:
        if len(stages) < 2:
            raise ValueError("ProgressiveDEOptimizer requires at least two stages")
        if len(stages) != len(settings.refinement_groups):
            raise ValueError("Progressive stage count does not match configuration")
        self.evaluator = evaluator
        self.fitness_space = fitness_space
        self.stages = tuple(stages)
        self.settings = settings
        self.observer = observer

    def optimize(self) -> SearchResult:
        previous_result = None
        previous_space = None
        total_unique = 0
        generation_offset = 0
        combined_history: List[Mapping[str, object]] = []
        stage_count = len(self.stages)
        for stage_index, stage in enumerate(self.stages):
            stage_observer = StageObserver(
                self.observer,
                stage_index,
                stage_count,
                stage.name,
                generation_offset,
                total_unique,
            )
            service = CandidateEvaluationService(
                evaluator=self.evaluator,
                workers=stage.settings.workers,
                observer=stage_observer,
            )
            initial_population = None
            if previous_result is not None and previous_space is not None:
                if not isinstance(stage.search_space, ScopedFusionSearchSpace):
                    raise TypeError("Refined stages require ScopedFusionSearchSpace")
                initial_population = transfer_population(
                    previous_result.population,
                    previous_space,
                    stage.search_space,
                    stage.settings.population_size,
                )
                # Projection changes the genome fingerprint while preserving
                # the represented model and inherited evaluation. Persist the
                # new representation so checkpoints and final Pareto links do
                # not point at a missing candidate record.
                for individual in initial_population:
                    stage_observer.on_evaluation(individual, cached=True)

            granularity = getattr(stage.search_space, "granularity", None)
            intervals = getattr(granularity, "intervals", ())
            self.observer.on_event(
                "search_stage_started",
                {
                    "stage_index": stage_index,
                    "stage_count": stage_count,
                    "stage": stage.name,
                    "population_size": stage.settings.population_size,
                    "generations": stage.settings.generations,
                    "dimension": stage.search_space.dimension,
                    "group_count": self.settings.refinement_groups[stage_index],
                    "layer_intervals": [list(item) for item in intervals],
                    "translator": type(stage.translator).__name__,
                    "inherited_population_count": (
                        len(initial_population)
                        if initial_population is not None
                        else 0
                    ),
                    "fitness_inherited": initial_population is not None,
                },
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
                initial_genomes=(
                    stage.initial_genomes
                    if initial_population is None
                    else ()
                ),
                initial_population=initial_population,
                evaluation_service=service,
            )
            result = optimizer.optimize()
            for item in result.history:
                adjusted = dict(item)
                adjusted["stage"] = stage.name
                adjusted["stage_index"] = stage_index
                adjusted["group_count"] = self.settings.refinement_groups[
                    stage_index
                ]
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
                    "stage_count": stage_count,
                    "stage": stage.name,
                    "group_count": self.settings.refinement_groups[stage_index],
                    "stage_unique_evaluations": result.unique_evaluations,
                    "unique_evaluations": total_unique,
                    "pareto_size": len(result.pareto_front),
                },
            )
            for payload in self.fitness_space.objective_best_payloads(
                result.pareto_front or result.population
            ):
                self.observer.on_event(
                    "objective_best",
                    dict(
                        payload,
                        scope="stage",
                        stage_index=stage_index,
                        stage_count=stage_count,
                        stage=stage.name,
                    ),
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
__all__ = ["ProgressiveDEOptimizer"]
