"""Budget-aware adaptive freedom release around the existing MOEA/D-DE.

The solver remains the repository's standard DE/current/1/bin plus unchanged
decomposition replacement. This orchestrator only changes the nested search
space between normal batches. It observes already evaluated trial directions,
uses an AND trigger between direction saturation and diminishing marginal
utility, and enforces a hard reserve that guarantees entry into the final
full-layer space. No promotion performs a model evaluation.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

from ...evaluation.benchmarks import FusionPlanEvaluator
from ...evaluation.cache import CandidateEvaluationService
from ..base import Optimizer, SearchObserver, SearchResult, StageObserver
from ..de.components import DEComponents
from ..de.genome import DEIndividual
from ..de.optimizer import DecompositionDEOptimizer
from ..de.survival import DecompositionUpdate
from ..fitness import FitnessSpace
from ..representation import ScopedFusionSearchSpace
from .config import AdaptiveResolutionConfig
from .controller import (
    BUDGET_RESERVE,
    BudgetAwareResolutionController,
)
from .hierarchy import ResolutionManager, SymmetryReleaseDiagnostic
from .monitor import DirectionalMonitor


class AdaptiveResolutionDEOptimizer(Optimizer):
    """Run standard MOEA/D-DE while adaptively releasing layer freedom."""

    def __init__(
        self,
        evaluator: FusionPlanEvaluator,
        fitness_space: FitnessSpace,
        stages: Sequence[DEComponents],
        settings: AdaptiveResolutionConfig,
        resolution: ResolutionManager,
        observer: SearchObserver,
    ) -> None:
        if len(stages) != resolution.stage_count:
            raise ValueError("Resolution stages and DE components differ")
        if len(stages) != settings.stage_count:
            raise ValueError("min_stage_fe does not match resolution stages")
        if any(
            stage.settings.population_size != settings.de.population_size
            for stage in stages
        ):
            raise ValueError("Adaptive resolution requires one population size")
        self.evaluator = evaluator
        self.fitness_space = fitness_space
        self.stages = tuple(stages)
        self.settings = settings
        self.resolution = resolution
        self.observer = observer
        self.controller = BudgetAwareResolutionController(settings)
        self.evaluation_service = CandidateEvaluationService(
            evaluator=evaluator,
            workers=settings.de.workers,
            observer=observer,
            max_unique_evaluations=settings.total_budget,
        )
        reference_directions = self._reference_directions()
        self.monitor = DirectionalMonitor(
            reference_directions,
            num_sectors=settings.num_sectors,
            sector_seed=settings.sector_seed,
            direction_window=settings.direction_window,
            min_directions_per_sector=settings.min_directions_per_sector,
            direction_norm_eps=settings.direction_norm_eps,
            utility_ema_beta=settings.utility_ema_beta,
            utility_eps=settings.utility_eps,
            structural_weights=resolution.structural_weights(
                self.stages[0].search_space.bounds
            ),
        )
        self._global_generation = 0
        self._fe_since_checkpoint = 0
        self._history: List[Mapping[str, object]] = []
        self._trajectory: List[Mapping[str, object]] = []
        self._stage_start_fe = 0
        self._symmetry_diagnostic = None  # type: Optional[SymmetryReleaseDiagnostic]

    def optimize(self) -> SearchResult:
        initial_population = None
        initial_archive = None
        initial_ideal = None
        final_optimizer = None

        while self.controller.fe_total < self.settings.total_budget:
            stage_index = self.resolution.stage
            stage = self.stages[stage_index]
            stage_observer = StageObserver(
                self.observer,
                stage_index,
                self.resolution.stage_count,
                stage.name,
                self._global_generation,
                0,
            )
            optimizer = self._create_stage_optimizer(
                stage,
                stage_observer,
                initial_population,
                initial_archive,
                initial_ideal,
            )
            self._stage_started(stage, initial_population)
            promoted = False

            while self.controller.fe_total < self.settings.total_budget:
                if optimizer.done:
                    raise RuntimeError(
                        "Adaptive DE exhausted its no-progress safety batch "
                        "ceiling before spending the configured real-FE budget"
                    )
                population_before = optimizer.population
                ideal_snapshot = optimizer.ideal_point
                candidates = tuple(optimizer.ask(announce=False))
                plans = tuple(
                    stage.translator.plan_for(item.genome)
                    for item in candidates
                )
                planned_fe = self.evaluation_service.required_evaluations(plans)
                promote_after_partial_batch = False

                if self.controller.should_promote_before_next_eval(planned_fe):
                    current_minimum = self.settings.min_stage_fe[
                        self.resolution.stage
                    ]
                    if self.controller.stage_fe >= current_minimum:
                        optimizer.cancel_pending_batch()
                        (
                            initial_population,
                            initial_archive,
                            initial_ideal,
                        ) = self._promote(optimizer, stage, BUDGET_RESERVE)
                        promoted = True
                        break
                    # The generic DE supports a restricted ordered batch. Use
                    # exactly the remaining current-stage slack so cache hits
                    # cannot leave this stage below its configured minimum.
                    candidates, plans = self._fit_exact_budget(
                        optimizer,
                        candidates,
                        plans,
                        self.controller.slack_fe,
                    )
                    planned_fe = self.evaluation_service.required_evaluations(
                        plans
                    )
                    promote_after_partial_batch = True

                remaining = self.controller.remaining_fe
                if self.resolution.is_final and planned_fe > remaining:
                    candidates, plans = self._fit_exact_budget(
                        optimizer, candidates, plans, remaining
                    )
                    planned_fe = self.evaluation_service.required_evaluations(plans)
                optimizer.announce_pending_batch(len(candidates))

                uncached = set(
                    self.evaluation_service.uncached_fingerprints(plans)
                )
                fe_before = self.evaluation_service.unique_evaluations
                optimizer.evaluate_candidates(candidates)
                fe_after = self.evaluation_service.unique_evaluations
                actual_fe = fe_after - fe_before
                if actual_fe != len(uncached):
                    raise AssertionError("Evaluation service FE accounting drifted")

                self._observe_trials(
                    optimizer,
                    stage,
                    candidates,
                    plans,
                    population_before,
                    ideal_snapshot,
                    uncached,
                )
                self.controller.note_fe(actual_fe)
                self._fe_since_checkpoint += actual_fe
                optimizer.tell(candidates)
                self._record_generation(optimizer, stage)
                self._observe_symmetry(stage, candidates)
                self._global_generation += 1
                final_optimizer = optimizer

                if promote_after_partial_batch:
                    current_minimum = self.settings.min_stage_fe[
                        self.resolution.stage
                    ]
                    if self.controller.stage_fe < current_minimum:
                        raise AssertionError(
                            "Partial DE batch failed to satisfy stage minimum FE"
                        )
                    (
                        initial_population,
                        initial_archive,
                        initial_ideal,
                    ) = self._promote(optimizer, stage, BUDGET_RESERVE)
                    promoted = True
                    break

                if (
                    self._fe_since_checkpoint
                    >= self.settings.checkpoint_interval_fe
                ):
                    checkpoint = self.monitor.checkpoint()
                    stats = self.controller.on_checkpoint(
                        checkpoint,
                        group_count=self.resolution.group_count,
                        decision_dim=stage.search_space.dimension,
                    )
                    self._fe_since_checkpoint = 0
                    self.observer.on_event(
                        "resolution_checkpoint", stats.to_dict()
                    )
                    if stats.promotion_requested:
                        assert stats.promotion_reason is not None
                        (
                            initial_population,
                            initial_archive,
                            initial_ideal,
                        ) = self._promote(
                            optimizer, stage, stats.promotion_reason
                        )
                        promoted = True
                        break

            if not promoted:
                break

        if final_optimizer is None:
            raise RuntimeError("Adaptive DE completed without an evaluated population")
        if not self.resolution.is_final:
            raise AssertionError("FE reserve failed to reach full-layer resolution")
        if self.controller.stage_fe < self.settings.min_stage_fe[-1]:
            raise AssertionError("Full-layer stage received less than its FE reserve")

        self._complete_stage(final_optimizer, self.stages[-1], "budget_exhausted")
        self.observer.on_event(
            "resolution_trajectory", {"stages": list(self._trajectory)}
        )
        pareto = tuple(
            sorted(
                final_optimizer.archive,
                key=lambda item: item.fitness or (),
            )
        )
        return SearchResult(
            population=final_optimizer.population,
            pareto_front=pareto,
            generations_completed=self._global_generation,
            unique_evaluations=self.controller.fe_total,
            history=tuple(self._history),
        )

    def _create_stage_optimizer(
        self,
        stage: DEComponents,
        observer: SearchObserver,
        population: Optional[Sequence[DEIndividual]],
        archive: Optional[Sequence[DEIndividual]],
        ideal_point: Optional[Sequence[float]],
    ) -> DecompositionDEOptimizer:
        return DecompositionDEOptimizer(
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
            seed=stage.settings.seed,
            workers=stage.settings.workers,
            observer=observer,
            initial_genomes=stage.initial_genomes if population is None else (),
            initial_population=population,
            initial_archive=archive,
            initial_ideal_point=ideal_point,
            evaluation_service=self.evaluation_service,
        )

    def _reference_directions(self) -> Tuple[Tuple[float, ...], ...]:
        # All stages use the same population size and objective count, hence
        # the same deterministic MOEA/D reference vectors and sector mapping.
        from ..de.decomposition import uniform_reference_directions

        return uniform_reference_directions(
            self.settings.de.population_size,
            len(self.fitness_space.definitions),
        )

    def _observe_trials(
        self,
        optimizer: DecompositionDEOptimizer,
        stage: DEComponents,
        candidates: Sequence[DEIndividual],
        plans: Sequence[object],
        population_before: Sequence[DEIndividual],
        ideal_snapshot: Optional[Sequence[float]],
        uncached: set,
    ) -> None:
        if ideal_snapshot is None:
            return
        if not isinstance(stage.update_strategy, DecompositionUpdate):
            raise TypeError(
                "Adaptive direction gain requires decomposition replacement"
            )
        decomposition = stage.update_strategy.decomposition
        observed = set()
        for trial, plan in zip(candidates, plans):
            key = plan.fingerprint()
            if key not in uncached or key in observed:
                continue
            observed.add(key)
            target_index = trial.target_index
            if target_index is None or trial.fitness is None:
                continue
            target = population_before[target_index]
            if target.fitness is None:
                raise ValueError("DE target has no fitness")
            weights = optimizer.reference_directions[target_index]
            scalar_before = decomposition.value(
                target.fitness, weights, ideal_snapshot
            )
            scalar_trial = decomposition.value(
                trial.fitness, weights, ideal_snapshot
            )
            self.monitor.observe(
                target_index,
                stage.search_space.encode(target.genome),
                stage.search_space.encode(trial.genome),
                scalar_before,
                scalar_trial,
                sample_id=key,
            )

    def _promote(
        self,
        optimizer: DecompositionDEOptimizer,
        source_stage: DEComponents,
        reason: str,
    ) -> Tuple[
        Tuple[DEIndividual, ...],
        Tuple[DEIndividual, ...],
        Optional[Tuple[float, ...]],
    ]:
        if self.resolution.is_final:
            raise RuntimeError("Final resolution cannot be promoted")
        target_stage = self.stages[self.resolution.stage + 1]
        if not isinstance(target_stage.search_space, ScopedFusionSearchSpace):
            raise TypeError("Refined resolution requires a scoped search space")
        fe_before = self.evaluation_service.unique_evaluations
        sibling_groups = self.resolution.sibling_coordinate_groups()
        population = self.resolution.promote_individuals(
            optimizer.population,
            source_stage.search_space,
            target_stage.search_space,
        )
        archive = self.resolution.promote_individuals(
            optimizer.archive,
            source_stage.search_space,
            target_stage.search_space,
        )
        if self.evaluation_service.unique_evaluations != fe_before:
            raise AssertionError("Resolution promotion consumed an FE")

        old_stage = self.resolution.stage
        old_groups = self.resolution.group_count
        old_dimension = source_stage.search_space.dimension
        self._complete_stage(optimizer, source_stage, reason)
        self.resolution.advance()
        self.controller.on_promotion()
        self.monitor.reset_for_new_stage(
            self.resolution.structural_weights(target_stage.search_space.bounds)
        )
        self._fe_since_checkpoint = 0
        self._symmetry_diagnostic = SymmetryReleaseDiagnostic(
            sibling_groups,
            self.settings.symmetry_diagnostic_batches,
            tolerance=self.settings.direction_norm_eps,
        )
        for individual in {
            item.genome.fingerprint(): item
            for item in tuple(population) + tuple(archive)
        }.values():
            self.observer.on_evaluation(individual, cached=True)
        self.observer.on_event(
            "resolution_promoted",
            {
                "old_stage": old_stage,
                "new_stage": self.resolution.stage,
                "stage_count": self.resolution.stage_count,
                "old_group_count": old_groups,
                "new_group_count": self.resolution.group_count,
                "old_dimension": old_dimension,
                "new_dimension": target_stage.search_space.dimension,
                "fe_at_promotion": fe_before,
                "reason": reason,
                "old_partition": [
                    list(group.interval)
                    for group in self.resolution.stages[old_stage]
                ],
                "new_partition": [
                    list(group.interval) for group in self.resolution.partition
                ],
            },
        )
        self._stage_start_fe = fe_before
        return population, archive, optimizer.ideal_point

    def _stage_started(
        self,
        stage: DEComponents,
        inherited: Optional[Sequence[DEIndividual]],
    ) -> None:
        self.observer.on_event(
            "search_stage_started",
            {
                "stage_index": self.resolution.stage,
                "stage_count": self.resolution.stage_count,
                "stage": stage.name,
                "population_size": stage.settings.population_size,
                "generations": "adaptive",
                "dimension": stage.search_space.dimension,
                "group_count": self.resolution.group_count,
                "layer_intervals": [
                    list(group.interval) for group in self.resolution.partition
                ],
                "translator": type(stage.translator).__name__,
                "inherited_population_count": len(inherited or ()),
                "fitness_inherited": inherited is not None,
                "min_stage_fe": self.settings.min_stage_fe[
                    self.resolution.stage
                ],
                "future_reserve": self.controller.future_reserve,
            },
        )

    def _complete_stage(
        self,
        optimizer: DecompositionDEOptimizer,
        stage: DEComponents,
        reason: str,
    ) -> None:
        entry = {
            "stage": self.resolution.stage,
            "start_fe": self._stage_start_fe,
            "end_fe": self.controller.fe_total,
            "fe_spent": self.controller.stage_fe,
            "group_count": self.resolution.group_count,
            "decision_dim": stage.search_space.dimension,
            "promotion_reason": reason,
        }
        self._trajectory.append(entry)
        self.observer.on_event(
            "search_stage_completed",
            {
                "stage_index": self.resolution.stage,
                "stage_count": self.resolution.stage_count,
                "stage": stage.name,
                "group_count": self.resolution.group_count,
                "stage_unique_evaluations": self.controller.stage_fe,
                "unique_evaluations": self.controller.fe_total,
                "pareto_size": len(optimizer.archive),
                "promotion_reason": reason,
            },
        )
        for payload in self.fitness_space.objective_best_payloads(
            optimizer.archive or optimizer.population
        ):
            self.observer.on_event(
                "objective_best",
                dict(
                    payload,
                    scope="stage",
                    stage_index=self.resolution.stage,
                    stage_count=self.resolution.stage_count,
                    stage=stage.name,
                ),
            )

    def _record_generation(
        self,
        optimizer: DecompositionDEOptimizer,
        stage: DEComponents,
    ) -> None:
        if not optimizer.history:
            return
        item = dict(optimizer.history[-1])
        item.update(
            {
                "generation": self._global_generation,
                "stage_generation": int(item["generation"]),
                "stage": stage.name,
                "stage_index": self.resolution.stage,
                "group_count": self.resolution.group_count,
                "decision_dim": stage.search_space.dimension,
                "unique_evaluations": self.controller.fe_total,
            }
        )
        self._history.append(item)

    def _observe_symmetry(
        self,
        stage: DEComponents,
        candidates: Sequence[DEIndividual],
    ) -> None:
        diagnostic = self._symmetry_diagnostic
        if diagnostic is None or diagnostic.complete:
            return
        diagnostic.observe(
            [stage.search_space.encode(item.genome) for item in candidates]
        )
        if diagnostic.complete:
            event = (
                "resolution_symmetry_released"
                if diagnostic.diverged
                else "resolution_symmetry_warning"
            )
            self.observer.on_event(
                event,
                {
                    "stage_index": self.resolution.stage,
                    "stage_count": self.resolution.stage_count,
                    "stage": stage.name,
                    "batches_observed": diagnostic.batches_observed,
                    "message": (
                        "Normal DE variation separated replicated child coordinates"
                        if diagnostic.diverged
                        else "The current DE variation operator preserves the "
                        "replicated-coordinate symmetry after resolution promotion."
                    ),
                },
            )

    def _fit_exact_budget(
        self,
        optimizer: DecompositionDEOptimizer,
        candidates: Sequence[DEIndividual],
        plans: Sequence[object],
        remaining: int,
    ) -> Tuple[Tuple[DEIndividual, ...], Tuple[object, ...]]:
        uncached = set(self.evaluation_service.uncached_fingerprints(plans))
        selected_candidates = []
        selected_plans = []
        selected_new = set()
        for candidate, plan in zip(candidates, plans):
            key = plan.fingerprint()
            if (
                key not in uncached
                or key in selected_new
                or len(selected_new) < remaining
            ):
                selected_candidates.append(candidate)
                selected_plans.append(plan)
                if key in uncached:
                    selected_new.add(key)
        if not selected_candidates or len(selected_new) != remaining:
            raise RuntimeError(
                "DE batch could not supply enough novel candidates for the "
                "exact real-FE allocation"
            )
        optimizer.restrict_pending_batch(selected_candidates)
        return tuple(selected_candidates), tuple(selected_plans)


__all__ = ["AdaptiveResolutionDEOptimizer"]
