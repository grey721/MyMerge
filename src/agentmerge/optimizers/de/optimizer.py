"""Parallel-batch MOEA/D-DE optimizer for expensive evaluations."""

from __future__ import annotations

import random
from typing import List, Mapping, Optional, Sequence, Tuple

from ...evaluation.benchmarks import FusionPlanEvaluator
from ...evaluation.cache import CandidateEvaluationService
from ..base import AskTellOptimizer, NullSearchObserver, SearchObserver, SearchResult
from ..pareto import assign_crowding_distance, fast_non_dominated_sort
from ..contracts import EvolutionSearchSpace, FusionPlanTranslator, SearchGenome
from ..fitness import FitnessSpace
from .decomposition import reference_neighborhoods, uniform_reference_directions
from .genome import DEIndividual
from .survival import PopulationUpdateStrategy
from .variation import DEVariation


class DecompositionDEOptimizer(AskTellOptimizer):
    """MOEA/D-DE with one asynchronously evaluable batch per generation.

    Reproduction reads a generation snapshot, so all expensive offspring can
    be evaluated concurrently. Replacement then applies the standard local
    scalar comparison to each subproblem.
    """

    def __init__(
        self,
        evaluator: FusionPlanEvaluator,
        fitness_space: FitnessSpace,
        search_space: EvolutionSearchSpace,
        translator: FusionPlanTranslator,
        variation: DEVariation,
        update_strategy: PopulationUpdateStrategy,
        population_size: int,
        generations: int,
        neighborhood_size: int,
        neighborhood_mating_probability: float,
        seed: int = 42,
        workers: int = 1,
        observer: Optional[SearchObserver] = None,
        initial_genomes: Optional[Sequence[SearchGenome]] = None,
        initial_population: Optional[Sequence[DEIndividual]] = None,
        initial_archive: Optional[Sequence[DEIndividual]] = None,
        initial_ideal_point: Optional[Sequence[float]] = None,
        evaluation_service: Optional[CandidateEvaluationService] = None,
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        if generations < 1:
            raise ValueError("generations must be positive")
        self.evaluator = evaluator
        self.fitness_space = fitness_space
        self.search_space = search_space
        self.translator = translator
        self.variation = variation
        self.update_strategy = update_strategy
        self.population_size = population_size
        self.generations = generations
        self.neighborhood_mating_probability = neighborhood_mating_probability
        self.rng = random.Random(seed)
        self.workers = workers
        self.observer = observer or NullSearchObserver()
        self.initial_genomes = tuple(
            genome.clone() for genome in (initial_genomes or ())
        )
        if initial_population is not None and self.initial_genomes:
            raise ValueError(
                "initial_genomes and initial_population are mutually exclusive"
            )
        self.reference_directions = uniform_reference_directions(
            population_size, len(fitness_space.definitions)
        )
        self.neighborhoods = reference_neighborhoods(
            self.reference_directions, neighborhood_size
        )
        self.evaluation_service = evaluation_service or CandidateEvaluationService(
            evaluator=evaluator,
            workers=workers,
            observer=self.observer,
        )
        self._history: List[Mapping[str, object]] = []
        self._population = self._prepare_initial_population(
            initial_population
        )
        self._archive: List[DEIndividual] = []
        self._ideal_point = (
            tuple(float(value) for value in initial_ideal_point)
            if initial_ideal_point is not None
            else None
        )  # type: Optional[Tuple[float, ...]]
        self._generation = -1
        self._pending_fingerprints: Optional[Tuple[str, ...]] = None
        self._pending_generation: Optional[int] = None
        self._pending_announced = False
        self._last_evaluation_cached: Tuple[bool, ...] = ()
        if self._population is not None:
            if self._ideal_point is not None and len(self._ideal_point) != len(
                self.fitness_space.definitions
            ):
                raise ValueError("Initial ideal point has the wrong dimension")
            self._update_ideal(self._population)
            if initial_archive is not None:
                prepared_archive = self._prepare_initial_archive(initial_archive)
                self._update_archive(prepared_archive)
            self._update_archive(self._population)
            self._assign_pareto_state(self._population)
            primed = {
                item.genome.fingerprint(): item
                for item in tuple(self._population) + tuple(self._archive)
            }
            for individual in primed.values():
                assert individual.evaluation is not None
                self.evaluation_service.prime(
                    self.translator.plan_for(individual.genome),
                    individual.evaluation,
                )

    @property
    def unique_evaluations(self) -> int:
        return self.evaluation_service.unique_evaluations

    @property
    def population(self) -> Tuple[DEIndividual, ...]:
        return tuple(item.clone() for item in (self._population or ()))

    @property
    def archive(self) -> Tuple[DEIndividual, ...]:
        return tuple(item.clone() for item in self._archive)

    @property
    def ideal_point(self) -> Optional[Tuple[float, ...]]:
        return self._ideal_point

    @property
    def history(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(self._history)

    @property
    def last_evaluation_cached(self) -> Tuple[bool, ...]:
        return self._last_evaluation_cached

    @property
    def done(self) -> bool:
        return (
            self._population is not None
            and self._generation >= self.generations - 1
            and self._pending_fingerprints is None
        )

    def ask(self, announce: bool = True) -> Sequence[DEIndividual]:
        if self._pending_fingerprints is not None:
            raise RuntimeError("tell() must consume the previous DE batch")
        if self.done:
            return ()
        generation = self._generation + 1
        candidates = (
            self._initial_population()
            if self._population is None
            else self._make_trials(self._population)
        )
        self._pending_fingerprints = tuple(
            item.genome.fingerprint() for item in candidates
        )
        self._pending_generation = generation
        self._pending_announced = False
        self._last_evaluation_cached = ()
        if announce:
            self.announce_pending_batch(len(candidates))
        return candidates

    def announce_pending_batch(self, candidate_count: int) -> None:
        """Publish a silently proposed batch immediately before evaluation."""

        if self._pending_generation is None:
            raise RuntimeError("No pending DE batch to announce")
        if self._pending_announced:
            return
        self.observer.on_event(
            "generation_started",
            {
                "generation": self._pending_generation,
                "generation_count": self.generations,
                "candidate_count": int(candidate_count),
            },
        )
        self._pending_announced = True

    def cancel_pending_batch(self) -> None:
        """Discard an unevaluated proposal without changing population state."""

        if self._pending_fingerprints is None:
            raise RuntimeError("No pending DE batch to cancel")
        self._pending_fingerprints = None
        self._pending_generation = None
        self._pending_announced = False
        self._last_evaluation_cached = ()

    def restrict_pending_batch(
        self,
        candidates: Sequence[DEIndividual],
    ) -> None:
        """Restrict a proposal to a non-empty ordered subset before evaluation.

        This supports an exact final FE budget without changing variation,
        scalar replacement, or any already evaluated state.
        """

        if self._pending_fingerprints is None or self._pending_announced:
            raise RuntimeError(
                "A pending unannounced DE batch is required for restriction"
            )
        fingerprints = tuple(
            item.genome.fingerprint() for item in candidates
        )
        if not fingerprints:
            raise ValueError("A restricted DE batch cannot be empty")
        pending = list(self._pending_fingerprints)
        cursor = 0
        for fingerprint in fingerprints:
            try:
                cursor = pending.index(fingerprint, cursor) + 1
            except ValueError as exc:
                raise ValueError(
                    "Restricted candidates are not an ordered pending subset"
                ) from exc
        self._pending_fingerprints = fingerprints

    def evaluate_candidates(self, candidates: Sequence[DEIndividual]) -> None:
        records = self.evaluation_service.evaluate(
            tuple(self.translator.plan_for(item.genome) for item in candidates)
        )
        for individual, record in zip(candidates, records):
            individual.evaluation = record.evaluation
            individual.fitness = self.fitness_space.encode(
                record.evaluation.metrics
            )
            self.observer.on_evaluation(individual, cached=record.cached)
        self._last_evaluation_cached = tuple(record.cached for record in records)

    def tell(self, candidates: Sequence[DEIndividual]) -> None:
        if self._pending_fingerprints is None or self._pending_generation is None:
            raise RuntimeError("ask() must be called before tell()")
        if tuple(item.genome.fingerprint() for item in candidates) != self._pending_fingerprints:
            raise ValueError("tell() candidates differ from the pending DE batch")
        if any(not item.evaluated for item in candidates):
            raise ValueError("tell() requires evaluated DE candidates")

        generation = self._pending_generation
        self._update_ideal(candidates)
        self._update_archive(candidates)
        if self._population is None:
            self._population = [item.clone() for item in candidates]
        else:
            assert self._ideal_point is not None
            self._population = self.update_strategy.update(
                self._population,
                candidates,
                self.reference_directions,
                self._ideal_point,
            )

        self._assign_pareto_state(self._population)
        self._generation = generation
        self._pending_fingerprints = None
        self._pending_generation = None
        self._pending_announced = False
        self._record_generation(generation, self._population)
        best_metrics = self.fitness_space.primary_metrics(self._archive)
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
        for payload in self.fitness_space.objective_best_payloads(self._archive):
            self.observer.on_event(
                "objective_best",
                dict(
                    payload,
                    scope="generation",
                    generation=generation,
                    generation_count=self.generations,
                ),
            )

    def result(self) -> SearchResult:
        if not self.done or self._population is None:
            raise RuntimeError("DE result is unavailable before completion")
        pareto = tuple(
            sorted(
                (item.clone() for item in self._archive),
                key=lambda item: item.fitness or (),
            )
        )
        return SearchResult(
            population=tuple(item.clone() for item in self._population),
            pareto_front=pareto,
            generations_completed=self._generation + 1,
            unique_evaluations=self.unique_evaluations,
            history=tuple(self._history),
        )

    def _initial_population(self) -> List[DEIndividual]:
        genomes: List[SearchGenome] = []
        seen = set()

        def append_unique(genome: SearchGenome) -> None:
            fingerprint = genome.fingerprint()
            if len(genomes) < self.population_size and fingerprint not in seen:
                seen.add(fingerprint)
                genomes.append(genome.clone())

        for genome in self.initial_genomes:
            append_unique(genome)
        for genome in self.search_space.initial_genomes():
            append_unique(genome)
        attempts = 0
        while len(genomes) < self.population_size and attempts < self.population_size * 100:
            append_unique(self.search_space.sample(self.rng))
            attempts += 1
        while len(genomes) < self.population_size:
            genomes.append(self.search_space.sample(self.rng))
        return [DEIndividual(genome=item) for item in genomes]

    def _prepare_initial_population(
        self,
        population: Optional[Sequence[DEIndividual]],
    ) -> Optional[List[DEIndividual]]:
        """Validate an evaluated warm start without consuming a generation."""

        if population is None:
            return None
        if len(population) != self.population_size:
            raise ValueError(
                "Initial DE population has {}, expected {}".format(
                    len(population), self.population_size
                )
            )
        prepared = []
        for index, individual in enumerate(population):
            if not individual.evaluated:
                raise ValueError(
                    "Initial DE population item {} is not evaluated".format(
                        index
                    )
                )
            # Encoding validates that the migrated chromosome belongs to this
            # stage's representation and has the expected dimension.
            self.search_space.encode(individual.genome)
            clone = individual.clone()
            clone.target_index = None
            clone.replacement_indices = ()
            prepared.append(clone)
        return prepared

    def _prepare_initial_archive(
        self,
        archive: Sequence[DEIndividual],
    ) -> List[DEIndividual]:
        prepared = []
        for index, individual in enumerate(archive):
            if not individual.evaluated:
                raise ValueError(
                    "Initial DE archive item {} is not evaluated".format(index)
                )
            self.search_space.encode(individual.genome)
            clone = individual.clone()
            clone.target_index = None
            clone.replacement_indices = ()
            prepared.append(clone)
        return prepared

    def _make_trials(
        self, population: Sequence[DEIndividual]
    ) -> List[DEIndividual]:
        trials = []
        all_indices = tuple(range(self.population_size))
        for target_index, target in enumerate(population):
            use_neighborhood = self.rng.random() < self.neighborhood_mating_probability
            mating_pool = (
                self.neighborhoods[target_index]
                if use_neighborhood
                else all_indices
            )
            donors = [index for index in mating_pool if index != target_index]
            if len(donors) < 2:
                donors = [index for index in all_indices if index != target_index]
            left_index, right_index = self.rng.sample(donors, 2)
            replacement_indices = list(mating_pool)
            self.rng.shuffle(replacement_indices)
            trials.append(
                DEIndividual(
                    genome=self.variation.create(
                        target.genome,
                        population[left_index].genome,
                        population[right_index].genome,
                        self.rng,
                    ),
                    target_index=target_index,
                    replacement_indices=tuple(replacement_indices),
                )
            )
        return trials

    def _update_ideal(self, candidates: Sequence[DEIndividual]) -> None:
        objective_count = len(self.fitness_space.definitions)
        minima = tuple(
            min(item.fitness[index] for item in candidates if item.fitness is not None)
            for index in range(objective_count)
        )
        if self._ideal_point is None:
            self._ideal_point = minima
        else:
            self._ideal_point = tuple(
                min(old, new) for old, new in zip(self._ideal_point, minima)
            )

    @staticmethod
    def _assign_pareto_state(population: Sequence[DEIndividual]) -> None:
        for rank, front in enumerate(fast_non_dominated_sort(population)):
            for individual in front:
                individual.rank = rank
            assign_crowding_distance(front)

    def _update_archive(self, candidates: Sequence[DEIndividual]) -> None:
        unique = {
            item.genome.fingerprint(): item.clone()
            for item in tuple(self._archive) + tuple(candidates)
        }
        fronts = fast_non_dominated_sort(tuple(unique.values()))
        self._archive = [
            item.clone() for item in (fronts[0] if fronts else ())
        ]
        self._assign_pareto_state(self._archive)

    def _record_generation(
        self, generation: int, population: Sequence[DEIndividual]
    ) -> None:
        summary = {
            "generation": generation,
            "unique_evaluations": self.unique_evaluations,
            "pareto_size": len(self._archive),
            "objective_minima": [
                min(item.fitness[index] for item in population if item.fitness is not None)
                for index in range(len(self.fitness_space.definitions))
            ],
            "ideal_point": list(self._ideal_point or ()),
        }
        self._history.append(summary)
        self.observer.on_generation(
            generation, population, self.unique_evaluations
        )
__all__ = ["DecompositionDEOptimizer"]
