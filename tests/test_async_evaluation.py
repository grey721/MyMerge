import time

from agentmerge.core.domain import AdapterMixture, EvaluationResult
from agentmerge.core.policies import FusionPlan
from agentmerge.evaluation.benchmarks import FusionPlanEvaluator


class _PlanDelayEvaluator(FusionPlanEvaluator):
    def __init__(self, delays):
        self.delays = delays

    def evaluate(self, plan):
        enabled = bool(plan.policy_for("agent").weights[0])
        delay = self.delays[0] if enabled else self.delays[1]
        time.sleep(delay)
        return EvaluationResult({"delay": delay})


def test_default_evaluator_stream_yields_completion_order():
    plans = (
        FusionPlan((("agent", AdapterMixture(("delay",), (1.0,))),)),
        FusionPlan((("agent", AdapterMixture(("delay",), (0.0,))),)),
    )

    # Use metadata-free plans with deterministic delays derived from weights.
    delayed = _PlanDelayEvaluator((0.05, 0.0))
    completed = list(delayed.evaluate_stream(plans, workers=2))

    assert [index for index, _ in completed] == [1, 0]
    ordered = delayed.evaluate_many(plans, workers=2)
    assert [item.metrics["delay"] for item in ordered] == [0.05, 0.0]
