from dataclasses import dataclass, field, replace
import random

import pytest

from agentmerge.core.config import apply_experiment_overrides, load_experiment_config
from agentmerge.core.domain import EvaluationResult
from agentmerge.evaluation.benchmarks import SyntheticMultiObjectiveEvaluator
from agentmerge.optimizers.adaptive_resolution_de import (
    AdaptiveResolutionDEOptimizer,
    BudgetAwareResolutionController,
    DirectionalMonitor,
    ResolutionManager,
    build_nested_layer_groups,
)
from agentmerge.optimizers.adaptive_resolution_de.config import (
    AdaptiveResolutionConfig,
)
from agentmerge.optimizers.adaptive_resolution_de.controller import (
    BUDGET_RESERVE,
    SEARCH_SATURATION,
    ZERO_PROGRESS_FALLBACK,
)
from agentmerge.optimizers.adaptive_resolution_de.hierarchy import (
    SymmetryReleaseDiagnostic,
)
from agentmerge.optimizers.adaptive_resolution_de.monitor import (
    DirectionalCheckpoint,
)
from agentmerge.optimizers.base import NullSearchObserver
from agentmerge.optimizers.de.config import DEConfig
from agentmerge.optimizers.de.genome import DEIndividual
from agentmerge.optimizers.de.variation import DifferentialVariation
from agentmerge.optimizers.representation import (
    FusionSearchSpace,
    ScopedFusionSearchSpace,
)
from agentmerge.runtime.experiment import ComponentFactory


def _settings(minimums=(6, 6, 12), budget=30, **overrides):
    de = DEConfig(
        population_size=6,
        generations=100,
        workers=1,
        seed=3,
        min_active_adapters=0,
        max_active_adapters=3,
        representation="adapter",
        representation_options={},
        neighborhood_size=3,
    )
    settings = AdaptiveResolutionConfig(
        num_layers=4,
        total_budget=budget,
        min_stage_fe=tuple(minimums),
        de=de,
        first_split=2,
        checkpoint_interval_fe=6,
    )
    return replace(settings, **overrides)


@pytest.mark.parametrize("num_layers", [1, 7, 12, 28, 32, 36])
def test_nested_hierarchy_covers_refines_and_finishes_layerwise(num_layers):
    stages = build_nested_layer_groups(num_layers, first_split=4)
    for groups in stages:
        assert groups[0].start_layer == 0
        assert groups[-1].end_layer == num_layers
        assert all(
            left.end_layer == right.start_layer
            for left, right in zip(groups, groups[1:])
        )
        assert sum(group.size for group in groups) == num_layers
    for parents, children in zip(stages, stages[1:]):
        for child in children:
            parent = parents[child.parent_group_id]
            assert parent.start_layer <= child.start_layer
            assert child.end_layer <= parent.end_layer
    assert all(group.size == 1 for group in stages[-1])


def test_lossless_vector_promotion_and_structural_geometry():
    manager = ResolutionManager(7, {"planner": 3, "solver": 2}, first_split=4)
    rng = random.Random(9)
    coarse = tuple(rng.random() for _ in range(manager.dimension))
    before = manager.decode_to_full_layer_coefficients(coarse)
    promoted = manager.promote_vector(coarse)
    manager.advance()
    after = manager.decode_to_full_layer_coefficients(promoted)

    assert before == after
    bounds = ((0.0, 1.0),) * manager.dimension
    direction = [0.0] * manager.dimension
    direction[0] = 1.0
    weighted = manager.structural_direction(direction, bounds)
    assert weighted[0] == pytest.approx(
        manager.partition[0].size ** 0.5
    )


def _monitor(width=4, **overrides):
    options = {
        "num_sectors": 1,
        "sector_seed": 0,
        "direction_window": 8,
        "min_directions_per_sector": 4,
        "direction_norm_eps": 1e-12,
        "utility_ema_beta": 0.5,
        "utility_eps": 1e-12,
        "structural_weights": (1.0,) * width,
    }
    options.update(overrides)
    return DirectionalMonitor(((1.0, 0.0),) * 8, **options)


def test_directional_coherence_identical_orthogonal_and_cancelling():
    identical = _monitor()
    for index in range(4):
        identical.observe(
            index,
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            1.0,
            0.5,
            sample_id=index,
        )
    assert identical.checkpoint().directional_coherence == pytest.approx(1.0)

    orthogonal = _monitor()
    for index in range(4):
        trial = [0.0] * 4
        trial[index] = 1.0
        orthogonal.observe(index, (0.0,) * 4, trial, 1.0, 0.5)
    assert orthogonal.checkpoint().directional_coherence == pytest.approx(0.0)

    cancelling = _monitor()
    for index, sign in enumerate((1.0, -1.0, 1.0, -1.0)):
        cancelling.observe(
            index,
            (0.0,) * 4,
            (sign, 0.0, 0.0, 0.0),
            1.0,
            0.5,
        )
    assert cancelling.checkpoint().directional_coherence < 0.0


def test_direction_monitor_utility_declines_and_deduplicates_offspring():
    monitor = _monitor(min_directions_per_sector=2)
    assert monitor.observe(
        0, (0.0,) * 4, (1.0, 0.0, 0.0, 0.0), 1.0, 0.0, sample_id="trial"
    )
    assert not monitor.observe(
        0, (0.0,) * 4, (1.0, 0.0, 0.0, 0.0), 1.0, 0.0, sample_id="trial"
    )
    assert monitor.direction_counts == (1,)
    early = monitor.checkpoint()
    for _ in range(5):
        monitor.observe(0, (0.0,) * 4, (0.1, 0.0, 0.0, 0.0), 1.0, 0.99)
        late = monitor.checkpoint()
    assert late.utility_ema < early.utility_ema
    assert late.utility_ratio < 0.2
    monitor.reset_for_new_stage((1.0,) * 4)
    assert monitor.direction_counts == (0,)
    reset = monitor.checkpoint()
    assert reset.utility_peak == 0.0
    assert reset.utility_ema == 0.0


def _checkpoint(coherence, ratio, marginal=0.1, valid_fraction=1.0):
    return DirectionalCheckpoint(
        valid_direction_sectors=(1 if valid_fraction else 0),
        total_sectors=1,
        valid_sector_fraction=valid_fraction,
        directional_coherence=coherence,
        marginal_utility=marginal,
        utility_ema=marginal,
        utility_peak=max(marginal, 0.1),
        utility_ratio=ratio,
    )


def test_and_trigger_minimum_budget_and_monitor_reset():
    settings = _settings(
        coherence_threshold=0.0,
        utility_ratio_threshold=0.2,
        saturation_patience=2,
    )

    low_direction_high_gain = BudgetAwareResolutionController(settings)
    low_direction_high_gain.note_fe(6)
    stats = low_direction_high_gain.on_checkpoint(
        _checkpoint(-0.2, 0.9), group_count=1, decision_dim=15
    )
    assert not stats.promotion_requested

    high_direction_low_gain = BudgetAwareResolutionController(settings)
    high_direction_low_gain.note_fe(6)
    stats = high_direction_low_gain.on_checkpoint(
        _checkpoint(0.8, 0.1), group_count=1, decision_dim=15
    )
    assert not stats.promotion_requested

    both = BudgetAwareResolutionController(settings)
    both.note_fe(5)
    stats = both.on_checkpoint(
        _checkpoint(-0.2, 0.1), group_count=1, decision_dim=15
    )
    assert not stats.promotion_requested
    both.note_fe(1)
    for _ in range(2):
        stats = both.on_checkpoint(
            _checkpoint(-0.2, 0.1), group_count=1, decision_dim=15
        )
    assert stats.promotion_reason == SEARCH_SATURATION
    both.on_promotion()
    assert both.stage_fe == 0
    assert both.saturation_streak == 0
    assert both.zero_progress_streak == 0


def test_budget_reserve_and_toy_state_machine_reach_full_stage():
    settings = _settings(minimums=(6, 6, 12), budget=30)
    controller = BudgetAwareResolutionController(settings)
    controller.note_fe(12)
    assert controller.should_promote_before_next_eval(1)
    controller.on_promotion()
    controller.note_fe(6)
    assert controller.should_promote_before_next_eval(1)
    controller.on_promotion()
    assert controller.is_final
    controller.note_fe(12)
    assert controller.fe_total == 30
    assert controller.stage_fe >= settings.min_stage_fe[-1]


def test_toy_direction_stream_search_promotes_then_budget_promotes():
    settings = _settings(
        minimums=(4, 4, 8),
        budget=32,
        direction_window=4,
        min_directions_per_sector=4,
        min_valid_sector_fraction=1.0,
        utility_ema_beta=0.0,
        coherence_threshold=0.0,
        utility_ratio_threshold=0.1,
        saturation_patience=2,
    )
    controller = BudgetAwareResolutionController(settings)
    monitor = DirectionalMonitor(
        ((1.0, 0.0),) * 4,
        num_sectors=1,
        sector_seed=0,
        direction_window=4,
        min_directions_per_sector=4,
        direction_norm_eps=1e-12,
        utility_ema_beta=0.0,
        utility_eps=1e-12,
        structural_weights=(1.0, 1.0),
    )

    def observe_checkpoint(directions, scalar_trial, prefix):
        for index, direction in enumerate(directions):
            monitor.observe(
                index,
                (0.0, 0.0),
                direction,
                1.0,
                scalar_trial,
                sample_id="{}-{}".format(prefix, index),
            )
        controller.note_fe(4)
        return controller.on_checkpoint(
            monitor.checkpoint(), group_count=1, decision_dim=2
        )

    coherent = ((1.0, 0.0),) * 4
    cancelling = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
    )
    assert not observe_checkpoint(coherent, 0.5, "early").promotion_requested
    assert not observe_checkpoint(cancelling, 0.99, "late-1").promotion_requested
    saturated = observe_checkpoint(cancelling, 0.99, "late-2")
    assert saturated.promotion_reason == SEARCH_SATURATION

    reasons = [saturated.promotion_reason]
    controller.on_promotion()
    monitor.reset_for_new_stage((1.0, 1.0))
    for batch in range(3):
        sustained = observe_checkpoint(coherent, 0.5, "stage1-{}".format(batch))
        assert not sustained.promotion_requested
    assert controller.should_promote_before_next_eval(1)
    reasons.append(BUDGET_RESERVE)
    controller.on_promotion()
    monitor.reset_for_new_stage((1.0, 1.0))
    controller.note_fe(8)

    assert reasons == [SEARCH_SATURATION, BUDGET_RESERVE]
    assert controller.is_final
    assert controller.fe_total == settings.total_budget
    assert controller.stage_fe == settings.min_stage_fe[-1]


def test_zero_progress_fallback_requires_patience_and_invalid_directions():
    settings = _settings(saturation_patience=2)
    controller = BudgetAwareResolutionController(settings)
    controller.note_fe(6)
    empty = _checkpoint(None, 0.0, marginal=0.0, valid_fraction=0.0)
    first = controller.on_checkpoint(empty, group_count=1, decision_dim=15)
    second = controller.on_checkpoint(empty, group_count=1, decision_dim=15)
    assert not first.promotion_requested
    assert second.promotion_reason == ZERO_PROGRESS_FALLBACK


def test_normal_de_variation_releases_replicated_coordinate_symmetry():
    manager = ResolutionManager(2, {"agent": 2}, first_split=2)
    source = FusionSearchSpace(("agent",), ("a", "b"), 0, 2)
    target = ScopedFusionSearchSpace(
        ("agent",),
        ("a", "b"),
        manager.recursive_partition(1),
        0,
        2,
    )
    genomes = [
        source.seed_from_spec({"agent": {"a": a, "b": 1.0 - a}})
        for a in (1.0, 0.0, 0.8, 0.2)
    ]
    individuals = [
        DEIndividual(
            genome=genome,
            fitness=(float(index),),
            evaluation=EvaluationResult({"quality": float(index)}),
        )
        for index, genome in enumerate(genomes)
    ]
    promoted = manager.promote_individuals(individuals, source, target)
    groups = manager.sibling_coordinate_groups()
    manager.advance()
    variation = DifferentialVariation(target, 0.5, 0.5, 0.0, 20.0)
    trials = []
    for index, item in enumerate(promoted):
        trials.append(
            variation.create(
                item.genome,
                promoted[(index + 1) % 4].genome,
                promoted[(index + 2) % 4].genome,
                random.Random(index + 3),
            )
        )
    diagnostic = SymmetryReleaseDiagnostic(groups, max_batches=2)
    diagnostic.observe([target.encode(item) for item in trials])
    assert diagnostic.diverged


@dataclass
class RecordingObserver(NullSearchObserver):
    events: list = field(default_factory=list)

    def on_event(self, event_type, payload=None):
        self.events.append((event_type, dict(payload or {})))


def test_adaptive_optimizer_mock_run_uses_exact_budget_and_zero_fe_promotions():
    config = load_experiment_config("experiments/smoke.yaml")
    config = apply_experiment_overrides(
        config,
        algorithm="adaptive_resolution_de",
        num_layers=4,
        populations=(6,),
        total_budget=30,
        min_stage_fe=(6, 6, 12),
        first_split=2,
        neighborhood_size=3,
        checkpoint_interval_fe=6,
    )
    observer = RecordingObserver()
    evaluator = SyntheticMultiObjectiveEvaluator(
        config.agent_names, config.adapter_names, seed=3
    )
    optimizer = ComponentFactory().create_optimizer(config, evaluator, observer)
    assert isinstance(optimizer, AdaptiveResolutionDEOptimizer)

    result = optimizer.optimize()

    assert result.unique_evaluations == 30
    assert optimizer.resolution.is_final
    assert optimizer.controller.stage_fe >= 12
    assert all(
        isinstance(item.genome, type(optimizer.stages[-1].search_space.uniform()))
        for item in result.population
    )
    promotions = [
        payload
        for event, payload in observer.events
        if event == "resolution_promoted"
    ]
    assert len(promotions) == 2
    assert [item["fe_at_promotion"] for item in promotions] == [12, 18]
    assert all(item["reason"] == BUDGET_RESERVE for item in promotions)
    trajectories = [
        payload
        for event, payload in observer.events
        if event == "resolution_trajectory"
    ]
    assert len(trajectories) == 1
    assert [item["fe_spent"] for item in trajectories[0]["stages"]] == [12, 6, 12]
