"""Dependency-light NSGA-II for sparse role-specific adapter mixtures."""

from __future__ import annotations

import random
from typing import List, Mapping, Optional, Sequence

from ..genome import NSGA2Individual
from ...evaluation.benchmarks import FusionPlanEvaluator
from ..fitness import FitnessSpace
from ..contracts import EvolutionSearchSpace, FusionPlanTranslator
from ..contracts import SearchGenome
from ..base import (
    AskTellOptimizer,
    NullSearchObserver,
    SearchObserver,
    SearchResult,
)
from ...evaluation.cache import CandidateEvaluationService
from ..pareto import assign_crowding_distance, fast_non_dominated_sort
from .variation import NSGA2Variation


class NSGA2Optimizer(AskTellOptimizer):
    def __init__(
        self,
        evaluator: FusionPlanEvaluator,
        fitness_space: FitnessSpace,
        search_space: EvolutionSearchSpace,
        translator: FusionPlanTranslator,
        variation: NSGA2Variation,
        population_size: int,
        generations: int,
        seed: int = 42,
        workers: int = 1,
        observer: Optional[SearchObserver] = None,
        initial_genomes: Optional[Sequence[SearchGenome]] = None,
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        if generations < 1:
            raise ValueError("generations must be positive")
        if workers < 1:
            raise ValueError("workers must be positive")
        self.evaluator = evaluator
        self.fitness_space = fitness_space
        self.search_space = search_space
        self.translator = translator
        self.variation = variation
        self.population_size = population_size
        self.generations = generations
        self.rng = random.Random(seed)
        self.workers = workers
        self.observer = observer or NullSearchObserver()
        self.initial_genomes = tuple(
            genome.clone() for genome in (initial_genomes or ())
        )
        self.evaluation_service = CandidateEvaluationService(
            evaluator=evaluator,
            workers=workers,
            observer=self.observer,
        )
        self._history = []  # type: List[Mapping[str, object]]
        self._population = None  # type: Optional[List[NSGA2Individual]]
        self._generation = -1
        self._pending_fingerprints = None  # type: Optional[Sequence[str]]
        self._pending_generation = None  # type: Optional[int]

    @property
    def unique_evaluations(self) -> int:
        return self.evaluation_service.unique_evaluations

    @property
    def done(self) -> bool:
        return (
            self._population is not None
            and self._generation >= self.generations - 1
            and self._pending_fingerprints is None
        )

    def ask(self) -> Sequence[NSGA2Individual]:
        if self._pending_fingerprints is not None:
            raise RuntimeError("tell() must consume the previous NSGA-II batch")
        if self.done:
            return ()
        if self._population is None:
            candidates = self._initial_population()
            generation = 0
        else:
            candidates = self._make_offspring(self._population)
            generation = self._generation + 1
        self._pending_fingerprints = tuple(
            candidate.genome.fingerprint() for candidate in candidates
        )
        self._pending_generation = generation
        self.observer.on_event(
            "generation_started",
            {
                "generation": generation,
                "generation_count": self.generations,
                "candidate_count": len(candidates),
            },
        )
        return candidates

    def evaluate_candidates(
        self, candidates: Sequence[NSGA2Individual]
    ) -> None:
        records = self.evaluation_service.evaluate(
            tuple(
                self.translator.plan_for(candidate.genome)
                for candidate in candidates
            )
        )
        for candidate, record in zip(candidates, records):
            candidate.evaluation = record.evaluation
            candidate.fitness = self.fitness_space.encode(
                record.evaluation.metrics
            )
            self.observer.on_evaluation(candidate, cached=record.cached)

    def tell(self, candidates: Sequence[NSGA2Individual]) -> None:
        if self._pending_fingerprints is None or self._pending_generation is None:
            raise RuntimeError("ask() must be called before tell()")
        fingerprints = tuple(
            candidate.genome.fingerprint() for candidate in candidates
        )
        if fingerprints != tuple(self._pending_fingerprints):
            raise ValueError("tell() candidates differ from the pending ask() batch")
        if any(not candidate.evaluated for candidate in candidates):
            raise ValueError("tell() requires evaluated candidates")

        generation = self._pending_generation
        if self._population is None:
            population = [candidate.clone() for candidate in candidates]
            self._assign_rank_and_crowding(population)
        else:
            population = self._environmental_selection(
                self._population + list(candidates)
            )
            self._assign_rank_and_crowding(population)
        self._population = population
        self._generation = generation
        self._pending_fingerprints = None
        self._pending_generation = None
        self._record_generation(generation, population)
        best_metrics = self.fitness_space.primary_metrics(population)
        self.observer.on_event(
            "generation_completed",
            {
                "generation": generation,
                "generation_count": self.generations,
                "unique_evaluations": self.unique_evaluations,
                "primary_objective": self.fitness_space.metric_names[0],
                "best_metrics": best_metrics,
            },
        )

    def result(self) -> SearchResult:
        if not self.done or self._population is None:
            raise RuntimeError("NSGA-II result is unavailable before completion")
        population = self._population
        unique_pareto = {}
        for individual in population:
            if individual.rank == 0:
                unique_pareto.setdefault(
                    individual.genome.fingerprint(), individual.clone()
                )
        pareto = tuple(unique_pareto.values())
        pareto = tuple(
            sorted(
                pareto,
                key=lambda individual: individual.fitness or (),
            )
        )
        return SearchResult(
            population=tuple(individual.clone() for individual in population),
            pareto_front=pareto,
            generations_completed=self._generation + 1,
            unique_evaluations=self.unique_evaluations,
            history=tuple(self._history),
        )

    def _initial_population(self) -> List[NSGA2Individual]:
        genomes = []  # type: List[SearchGenome]
        seen = set()

        def append_unique(genome: SearchGenome) -> None:
            if len(genomes) >= self.population_size:
                return
            fingerprint = genome.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                genomes.append(genome.clone())

        for genome in self.initial_genomes:
            append_unique(genome)
        for genome in self.search_space.initial_genomes():
            append_unique(genome)
        attempts = 0
        max_unique_attempts = self.population_size * 100
        while len(genomes) < self.population_size and attempts < max_unique_attempts:
            append_unique(self.search_space.sample(self.rng))
            attempts += 1
        while len(genomes) < self.population_size:
            genomes.append(self.search_space.sample(self.rng))
        return [NSGA2Individual(genome=genome) for genome in genomes]

    def _make_offspring(
        self, population: Sequence[NSGA2Individual]
    ) -> List[NSGA2Individual]:
        offspring: List[NSGA2Individual] = []
        while len(offspring) < self.population_size:
            left = self._tournament(population)
            right = self._tournament(population)
            genome = self.variation.crossover(
                left.genome, right.genome, self.rng
            )
            genome = self.variation.mutate(genome, self.rng)
            offspring.append(NSGA2Individual(genome=genome))
        return offspring

    def _tournament(
        self, population: Sequence[NSGA2Individual]
    ) -> NSGA2Individual:
        left, right = self.rng.sample(list(population), 2)
        if left.rank != right.rank:
            return left if left.rank < right.rank else right
        if left.crowding_distance != right.crowding_distance:
            return (
                left
                if left.crowding_distance > right.crowding_distance
                else right
            )
        return left if self.rng.random() < 0.5 else right

    def _environmental_selection(
        self, combined: Sequence[NSGA2Individual]
    ) -> List[NSGA2Individual]:
        fronts = fast_non_dominated_sort(combined)
        selected: List[NSGA2Individual] = []
        for rank, front in enumerate(fronts):
            for individual in front:
                individual.rank = rank
            assign_crowding_distance(front)
            remaining = self.population_size - len(selected)
            if remaining <= 0:
                break
            if len(front) <= remaining:
                selected.extend(front)
            else:
                selected.extend(
                    sorted(
                        front,
                        key=lambda individual: individual.crowding_distance,
                        reverse=True,
                    )[:remaining]
                )
                break
        return [individual.clone() for individual in selected]

    def _assign_rank_and_crowding(
        self, population: Sequence[NSGA2Individual]
    ) -> None:
        fronts = fast_non_dominated_sort(population)
        for rank, front in enumerate(fronts):
            for individual in front:
                individual.rank = rank
            assign_crowding_distance(front)

    def _record_generation(
        self, generation: int, population: Sequence[NSGA2Individual]
    ) -> None:
        pareto_count = len(
            {
                individual.genome.fingerprint()
                for individual in population
                if individual.rank == 0
            }
        )
        summary = {
            "generation": generation,
            "unique_evaluations": self.unique_evaluations,
            "pareto_size": pareto_count,
            "objective_minima": [
                min(
                    individual.fitness[index]
                    for individual in population
                    if individual.fitness is not None
                )
                for index in range(len(self.fitness_space.definitions))
            ],
        }
        self._history.append(summary)
        self.observer.on_generation(
            generation, population, self.unique_evaluations
        )
