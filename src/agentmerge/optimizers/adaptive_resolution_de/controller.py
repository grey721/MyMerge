"""Hard FE reserve and adaptive resolution-promotion state machine.

Directional coherence and normalized improvement-per-FE must both saturate
before a search-driven promotion. The future reserve is different: it is a
hard budget-feasibility constraint ensuring eventual full layer-wise search,
even when search statistics never request promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from .config import AdaptiveResolutionConfig
from .monitor import DirectionalCheckpoint


SEARCH_SATURATION = "search_saturation"
ZERO_PROGRESS_FALLBACK = "zero_progress_fallback"
BUDGET_RESERVE = "budget_reserve"


@dataclass(frozen=True)
class MonitorStats:
    fe_total: int
    stage: int
    stage_fe: int
    group_count: int
    decision_dim: int
    valid_direction_sectors: int
    total_sectors: int
    valid_sector_fraction: float
    directional_coherence: Optional[float]
    marginal_utility: float
    utility_ema: float
    utility_peak: float
    utility_ratio: float
    directional_saturated: bool
    utility_saturated: bool
    zero_progress: bool
    saturation_streak: int
    remaining_fe: int
    future_reserve: int
    slack_fe: int
    promotion_requested: bool
    promotion_reason: Optional[str]

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


class BudgetAwareResolutionController:
    """Combine an AND search trigger with a strict future-stage reserve."""

    def __init__(self, config: AdaptiveResolutionConfig) -> None:
        if sum(config.min_stage_fe) > config.total_budget:
            raise ValueError("Stage minimum FE exceeds total budget")
        self.config = config
        self.stage = 0
        self.fe_total = 0
        self.stage_fe = 0
        self.saturation_streak = 0
        self.zero_progress_streak = 0

    @property
    def is_final(self) -> bool:
        return self.stage == len(self.config.min_stage_fe) - 1

    @property
    def remaining_fe(self) -> int:
        return self.config.total_budget - self.fe_total

    @property
    def future_reserve(self) -> int:
        return sum(self.config.min_stage_fe[self.stage + 1 :])

    @property
    def slack_fe(self) -> int:
        return self.remaining_fe - self.future_reserve

    def note_fe(self, count: int = 1) -> None:
        value = int(count)
        if value < 0:
            raise ValueError("FE increment cannot be negative")
        if self.fe_total + value > self.config.total_budget:
            raise ValueError("FE increment exceeds the total budget")
        self.fe_total += value
        self.stage_fe += value

    def should_promote_before_next_eval(self, planned_fe: int = 1) -> bool:
        """Enforce ``remaining - next >= future minimum allocations``.

        If remaining FE equals the future reserve, the current stage has zero
        slack. Spending even one more FE would make at least one future minimum
        allocation infeasible, so promotion is mandatory rather than heuristic.
        """

        planned = int(planned_fe)
        if planned < 0:
            raise ValueError("planned_fe cannot be negative")
        return (
            not self.is_final
            and self.remaining_fe - planned < self.future_reserve
        )

    def on_checkpoint(
        self,
        checkpoint: DirectionalCheckpoint,
        *,
        group_count: int,
        decision_dim: int,
    ) -> MonitorStats:
        direction_valid = (
            checkpoint.directional_coherence is not None
            and checkpoint.valid_sector_fraction
            >= self.config.min_valid_sector_fraction
        )
        directional_saturated = bool(
            direction_valid
            and checkpoint.directional_coherence
            <= self.config.coherence_threshold
        )
        utility_saturated = (
            checkpoint.utility_ratio <= self.config.utility_ratio_threshold
        )
        minimum_met = self.stage_fe >= self.config.min_stage_fe[self.stage]
        saturated = minimum_met and directional_saturated and utility_saturated
        self.saturation_streak = (
            self.saturation_streak + 1 if saturated else 0
        )

        zero_progress = bool(
            minimum_met
            and checkpoint.valid_sector_fraction
            < self.config.min_valid_sector_fraction
            and checkpoint.marginal_utility <= self.config.utility_zero_eps
        )
        self.zero_progress_streak = (
            self.zero_progress_streak + 1 if zero_progress else 0
        )

        reason = None
        if not self.is_final:
            if self.saturation_streak >= self.config.saturation_patience:
                reason = SEARCH_SATURATION
            elif (
                self.config.enable_zero_progress_fallback
                and self.zero_progress_streak
                >= self.config.saturation_patience
            ):
                reason = ZERO_PROGRESS_FALLBACK
        return MonitorStats(
            fe_total=self.fe_total,
            stage=self.stage,
            stage_fe=self.stage_fe,
            group_count=int(group_count),
            decision_dim=int(decision_dim),
            valid_direction_sectors=checkpoint.valid_direction_sectors,
            total_sectors=checkpoint.total_sectors,
            valid_sector_fraction=checkpoint.valid_sector_fraction,
            directional_coherence=checkpoint.directional_coherence,
            marginal_utility=checkpoint.marginal_utility,
            utility_ema=checkpoint.utility_ema,
            utility_peak=checkpoint.utility_peak,
            utility_ratio=checkpoint.utility_ratio,
            directional_saturated=directional_saturated,
            utility_saturated=utility_saturated,
            zero_progress=zero_progress,
            saturation_streak=self.saturation_streak,
            remaining_fe=self.remaining_fe,
            future_reserve=self.future_reserve,
            slack_fe=self.slack_fe,
            promotion_requested=reason is not None,
            promotion_reason=reason,
        )

    def on_promotion(self) -> None:
        if self.is_final:
            raise RuntimeError("Cannot promote beyond the full-layer stage")
        self.stage += 1
        self.stage_fe = 0
        self.saturation_streak = 0
        self.zero_progress_streak = 0


__all__ = [
    "BUDGET_RESERVE",
    "SEARCH_SATURATION",
    "ZERO_PROGRESS_FALLBACK",
    "BudgetAwareResolutionController",
    "MonitorStats",
]
