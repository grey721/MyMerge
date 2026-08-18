"""Benchmark cases, scorers, and system-level fusion-plan evaluators."""

from __future__ import annotations

import hashlib
import ast
import json
import math
import random
import re
import string
import subprocess
import sys
import time
from itertools import groupby
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..evaluation.system import AgentSystem
from ..core.config import BenchmarkConfig
from ..core.domain import EvaluationResult, average
from ..core.policies import (
    FusionPlan,
    average_policy_mixture,
    iter_policy_mixtures,
)
from .metrics import BenchmarkMetricCatalog, SYNTHETIC_METRIC_NAMES


class FusionPlanEvaluator(ABC):
    @property
    def metric_names(self) -> Tuple[str, ...]:
        """Metrics this evaluator can expose for objective selection."""

        return ()

    @abstractmethod
    def evaluate(self, plan: FusionPlan) -> EvaluationResult:
        """Evaluate a complete multi-agent coefficient assignment."""

    def evaluate_many(
        self, plans: Sequence[FusionPlan], workers: int = 1
    ) -> Sequence[EvaluationResult]:
        """Evaluate a batch while preserving input order.

        Local evaluators use threads by default. Device-aware evaluators can
        override this method with process- or service-level scheduling.
        """

        results = [None] * len(plans)
        for index, evaluation in self.evaluate_stream(plans, workers):
            results[index] = evaluation
        if any(result is None for result in results):
            raise RuntimeError("Evaluator stream completed with missing results")
        return results

    def evaluate_stream(
        self, plans: Sequence[FusionPlan], workers: int = 1
    ) -> Iterator[Tuple[int, EvaluationResult]]:
        """Yield evaluations in completion order with stable input indices."""

        if workers <= 1:
            for index, plan in enumerate(plans):
                yield index, self.evaluate(plan)
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_indices = {
                executor.submit(self.evaluate, plan): index
                for index, plan in enumerate(plans)
            }
            for future in as_completed(future_indices):
                yield future_indices[future], future.result()

    def close(self) -> None:
        """Release evaluator-owned processes or remote resources."""

    def abort(self) -> None:
        """Promptly release resources after interruption."""

        self.close()

    def release_unreferenced_adapters(
        self, reference_counts: Mapping[str, int]
    ) -> Mapping[str, Any]:
        """Optional Stage-I/II resource barrier for model-backed evaluators."""

        return {
            "supported": False,
            "released_adapter_keys": [],
            "released_coordinate_indices": [],
            "released_source_count": 0,
            "released_source_bytes": 0,
            "resident_adapter_keys": [],
            "resident_coordinate_indices": [],
            "resident_source_count": 0,
            "resident_source_bytes": 0,
        }

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {"evaluator": type(self).__name__, "mode": "local"}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    input_text: str
    target: str
    group: str = "default"
    scorer_name: Optional[str] = None
    metadata: Mapping[str, Any] = None


class OutputScorer(ABC):
    @abstractmethod
    def score(self, prediction: str, target: str) -> float:
        pass


class ExactMatchScorer(OutputScorer):
    def score(self, prediction: str, target: str) -> float:
        return float(_normalize_answer(prediction) == _normalize_answer(target))


class NumericAnswerScorer(OutputScorer):
    """Compare the last numeric value in prediction and target."""

    _NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance

    def score(self, prediction: str, target: str) -> float:
        predicted = self._NUMBER.findall(prediction.replace(",", ""))
        expected = self._NUMBER.findall(target.replace(",", ""))
        if not predicted or not expected:
            return 0.0
        return float(
            math.isclose(
                float(predicted[-1]),
                float(expected[-1]),
                rel_tol=self.tolerance,
                abs_tol=self.tolerance,
            )
        )


class HotpotF1Scorer(OutputScorer):
    """Official-style normalized token F1 used by HotpotQA."""

    def score(self, prediction: str, target: str) -> float:
        predicted_tokens = _hotpot_normalize(prediction).split()
        target_tokens = _hotpot_normalize(target).split()
        if not predicted_tokens or not target_tokens:
            return float(predicted_tokens == target_tokens)

        predicted_counts: Dict[str, int] = {}
        target_counts: Dict[str, int] = {}
        for token in predicted_tokens:
            predicted_counts[token] = predicted_counts.get(token, 0) + 1
        for token in target_tokens:
            target_counts[token] = target_counts.get(token, 0) + 1
        common = sum(
            min(count, target_counts.get(token, 0))
            for token, count in predicted_counts.items()
        )
        if common == 0:
            return 0.0
        precision = common / len(predicted_tokens)
        recall = common / len(target_tokens)
        return 2.0 * precision * recall / (precision + recall)


class MathAnswerScorer(OutputScorer):
    """Compare concise MATH-style answers without executing model output.

    It first performs normalized symbolic-string comparison and then supports a
    deliberately small numeric subset (integers, decimals and simple fractions).
    A full CAS is intentionally avoided in the admission benchmark.
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = tolerance

    def score(self, prediction: str, target: str) -> float:
        predicted = _normalize_math(_extract_final_answer(prediction))
        expected = _normalize_math(_extract_final_answer(target))
        if predicted == expected:
            return 1.0
        predicted_number = _parse_simple_number(predicted)
        expected_number = _parse_simple_number(expected)
        if predicted_number is None or expected_number is None:
            return 0.0
        return float(
            math.isclose(
                predicted_number,
                expected_number,
                rel_tol=self.tolerance,
                abs_tol=self.tolerance,
            )
        )


class PythonLiteralScorer(OutputScorer):
    """Compare Python-literal values without executing model output."""

    def score(self, prediction: str, target: str) -> float:
        predicted = _extract_python_literal(prediction)
        expected = _extract_python_literal(target)
        if predicted is None or expected is None:
            return 0.0
        return float(predicted == expected)


class ChoiceAnswerScorer(OutputScorer):
    """Score concise classification or multiple-choice answers robustly."""

    _LETTER = re.compile(r"\(([A-Z])\)", re.IGNORECASE)
    _NLI = re.compile(
        r"\b(not[_ -]?entailment|entailment|contradiction|neutral)\b",
        re.IGNORECASE,
    )

    def score(self, prediction: str, target: str) -> float:
        expected = self._canonical(target)
        candidates = self._labels(prediction)
        if not expected:
            return 0.0
        if candidates:
            return float(candidates[-1] == expected)

        # A few official BBH tasks (for example dyck_languages and
        # word_sorting) have concise free-form targets instead of option
        # letters.  Accept only the extracted final line/answer, never an
        # arbitrary rationale substring.
        final_answer = _normalize_answer(_extract_final_answer(prediction))
        return float(final_answer == expected)

    @classmethod
    def _canonical(cls, value: str) -> str:
        labels = cls._labels(value)
        if labels:
            return labels[-1]
        return _normalize_answer(value)

    @classmethod
    def _labels(cls, value: str) -> List[str]:
        labels = [match.upper() for match in cls._LETTER.findall(value)]
        if labels:
            return labels
        return [
            match.lower().replace(" ", "_").replace("-", "_")
            for match in cls._NLI.findall(value)
        ]


class PythonCodeScorer(OutputScorer):
    """Run HumanEval/MBPP base tests in a constrained child process.

    Generated code never executes in the long-lived evaluator process. The
    worker applies a conservative AST policy and OS resource limits before
    running trusted benchmark assertions. This is not a general security
    sandbox, so only the bundled benchmark tests should be used as targets.
    """

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.worker_path = Path(__file__).with_name("code_worker.py")

    def score(self, prediction: str, target: str) -> float:
        try:
            target_specification = json.loads(target)
            payload = json.dumps(
                {
                    "prediction": prediction,
                    "target": target_specification,
                },
                ensure_ascii=False,
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(self.worker_path)],
                input=payload,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
            return 0.0
        return float(completed.returncode == 0)


class BfclAstScorer(OutputScorer):
    """BFCL V4 single-call AST scorer for prompt-mode local models."""

    def score(self, prediction: str, target: str) -> float:
        try:
            specification = json.loads(target)
            calls = _extract_function_calls(prediction)
            functions = specification["functions"]
            answers = specification["ground_truth"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, SyntaxError):
            return 0.0
        if len(calls) != 1 or len(answers) != 1:
            return 0.0
        expected_name = next(iter(answers[0]), None)
        if expected_name is None or expected_name not in calls[0]:
            return 0.0
        description = next(
            (item for item in functions if item.get("name") == expected_name),
            None,
        )
        if description is None:
            return 0.0
        expected_params = answers[0][expected_name]
        actual_params = calls[0][expected_name]
        properties = description.get("parameters", {}).get("properties", {})
        required = description.get("parameters", {}).get("required", ())
        if any(name not in actual_params for name in required):
            return 0.0
        if any(
            name not in properties or name not in expected_params
            for name in actual_params
        ):
            return 0.0
        for name, value in actual_params.items():
            alternatives = expected_params[name]
            if alternatives == []:
                if value != []:
                    return 0.0
                continue
            if not any(_bfcl_value_equal(value, item) for item in alternatives):
                return 0.0
        if any(
            name not in actual_params and "" not in alternatives
            for name, alternatives in expected_params.items()
        ):
            return 0.0
        return 1.0


class GridPathScorer(OutputScorer):
    """Execute a U/D/L/R path on a fixed obstacle grid."""

    def score(self, prediction: str, target: str) -> float:
        try:
            specification = json.loads(target)
            rows = tuple(str(row) for row in specification["grid"])
            max_steps = int(specification["max_steps"])
            position = _find_grid_cell(rows, "S")
            goal = _find_grid_cell(rows, "G")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        moves = _extract_moves(prediction)
        if not moves or len(moves) > max_steps:
            return 0.0
        for move in moves:
            candidate = _move(position, move)
            if not _is_open_grid_cell(rows, candidate):
                return 0.0
            position = candidate
        return float(position == goal)


class SudokuScorer(OutputScorer):
    """Validate a completed square Sudoku against its original clues."""

    def score(self, prediction: str, target: str) -> float:
        try:
            specification = json.loads(target)
            puzzle = specification["puzzle"]
            candidate = _extract_python_literal(prediction)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        if not isinstance(candidate, (list, tuple)):
            return 0.0
        size = len(puzzle)
        if size < 1 or len(candidate) != size:
            return 0.0
        expected = set(range(1, size + 1))
        box_size = int(math.sqrt(size))
        if box_size * box_size != size:
            return 0.0
        try:
            grid = [list(map(int, row)) for row in candidate]
        except (TypeError, ValueError):
            return 0.0
        if any(len(row) != size or set(row) != expected for row in grid):
            return 0.0
        if any(
            {grid[row][column] for row in range(size)} != expected
            for column in range(size)
        ):
            return 0.0
        for row_start in range(0, size, box_size):
            for column_start in range(0, size, box_size):
                values = {
                    grid[row][column]
                    for row in range(row_start, row_start + box_size)
                    for column in range(column_start, column_start + box_size)
                }
                if values != expected:
                    return 0.0
        for row in range(size):
            for column in range(size):
                clue = int(puzzle[row][column])
                if clue and grid[row][column] != clue:
                    return 0.0
        return 1.0


class SokobanScorer(OutputScorer):
    """Execute a move sequence in a compact Sokoban state."""

    def score(self, prediction: str, target: str) -> float:
        try:
            specification = json.loads(target)
            rows = tuple(str(row) for row in specification["grid"])
            max_steps = int(specification["max_steps"])
            player, boxes, goals, walls = _parse_sokoban_grid(rows)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        moves = _extract_moves(prediction)
        if not moves or len(moves) > max_steps:
            return 0.0
        boxes = set(boxes)
        for move in moves:
            destination = _move(player, move)
            if (
                not _is_open_grid_cell(rows, destination)
                or destination in walls
            ):
                return 0.0
            if destination in boxes:
                pushed = _move(destination, move)
                if (
                    not _is_open_grid_cell(rows, pushed)
                    or pushed in walls
                    or pushed in boxes
                ):
                    return 0.0
                boxes.remove(destination)
                boxes.add(pushed)
            player = destination
        return float(boxes == goals)


class JsonlBenchmark:
    def __init__(
        self, config: BenchmarkConfig, scorer: OutputScorer
    ) -> None:
        if config.path is None:
            raise ValueError("JSONL benchmark requires benchmark.path")
        self.config = config
        self.scorer = scorer
        self.scorers: Dict[str, OutputScorer] = {}
        self.scorer_field = str(config.options.get("scorer_field", "scorer"))
        self.cases = self._load(Path(config.path))

    def _load(self, path: Path) -> Tuple[BenchmarkCase, ...]:
        cases: List[BenchmarkCase] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                case_id = str(
                    item.get(self.config.id_field, "case-{}".format(line_number))
                )
                cases.append(
                    BenchmarkCase(
                        case_id=case_id,
                        input_text=str(item[self.config.input_field]),
                        target=str(item[self.config.target_field]),
                        group=str(item.get(self.config.group_field, "default")),
                        scorer_name=(
                            str(item[self.scorer_field])
                            if item.get(self.scorer_field) is not None
                            else None
                        ),
                        metadata={
                            key: value
                            for key, value in item.items()
                            if key
                            not in {
                                self.config.id_field,
                                self.config.input_field,
                                self.config.target_field,
                                self.config.group_field,
                                self.scorer_field,
                            }
                        },
                    )
                )
                if (
                    self.config.max_cases is not None
                    and len(cases) >= self.config.max_cases
                ):
                    break
        if not cases:
            raise ValueError("Benchmark contains no cases: {}".format(path))
        return tuple(cases)

    def score(self, case: BenchmarkCase, prediction: str) -> float:
        if case.scorer_name is None:
            scorer = self.scorer
        else:
            scorer_name = case.scorer_name.lower()
            if scorer_name not in self.scorers:
                self.scorers[scorer_name] = build_scorer_by_name(
                    scorer_name, self.config.options
                )
            scorer = self.scorers[scorer_name]
        return scorer.score(prediction, case.target)


class SystemBenchmarkEvaluator(FusionPlanEvaluator):
    """Evaluate end-to-end output and aggregate actual runtime costs."""

    def __init__(
        self,
        system: AgentSystem,
        benchmark: JsonlBenchmark,
    ) -> None:
        self.system = system
        self.benchmark = benchmark
        self.batch_size = benchmark.config.inference_batch_size
        self.metric_catalog = BenchmarkMetricCatalog.from_groups(
            tuple(case.group for case in benchmark.cases),
            quality_domains=benchmark.config.options.get(
                "quality_domains", {}
            ),
        )

    @property
    def metric_names(self) -> Tuple[str, ...]:
        return self.metric_catalog.names

    def evaluate(self, plan: FusionPlan) -> EvaluationResult:
        scores: List[float] = []
        group_scores: Dict[str, List[float]] = {}
        runs = []
        runtime_before = self.system.backend.runtime_snapshot()
        candidate_started = time.perf_counter()
        # A materializing backend compiles each distinct Agent mixture lazily
        # on first use, then reuses it for every benchmark micro-batch of this
        # plan. Exiting this scope releases all temporary adapters.
        with self.system.candidate_session():
            for start in range(0, len(self.benchmark.cases), self.batch_size):
                batch = self.benchmark.cases[start : start + self.batch_size]
                runs.extend(
                    self.system.run_many(
                        tuple(case.input_text for case in batch),
                        plan,
                        tuple(
                            {
                                **dict(case.metadata or {}),
                                "case_id": case.case_id,
                                "group": case.group,
                            }
                            for case in batch
                        ),
                    )
                )
        candidate_session_seconds = time.perf_counter() - candidate_started
        runtime_after = self.system.backend.runtime_snapshot()
        fusion_runtime = _fusion_runtime_delta(runtime_before, runtime_after)
        output_agent = self.system.topology.output_agent
        output_agent_config = self.system.agents[output_agent]
        output_limit = output_agent_config.max_new_tokens
        if output_limit is None:
            output_limit = self.system.generation.max_new_tokens
        output_lengths: List[int] = []
        output_completion_tokens: List[int] = []
        output_empty_count = 0
        output_limit_hit_count = 0
        repetitive_output_count = 0
        group_statistics: Dict[str, Dict[str, Any]] = {}
        for case, run in zip(self.benchmark.cases, runs):
            score = self.benchmark.score(case, run.output)
            scores.append(score)
            group_scores.setdefault(case.group, []).append(score)
            output_lengths.append(len(run.output))
            if not run.output:
                output_empty_count += 1
            if _is_repetitive_output(run.output):
                repetitive_output_count += 1
            output_trace = next(
                (
                    trace
                    for trace in reversed(run.traces)
                    if trace.agent_name == output_agent
                ),
                None,
            )
            completion_tokens = (
                0
                if output_trace is None
                else output_trace.generation.completion_tokens
            )
            output_completion_tokens.append(completion_tokens)
            if completion_tokens >= int(output_limit):
                output_limit_hit_count += 1

        for group, values in sorted(group_scores.items()):
            group_statistics[group] = _score_statistics(values)

        policies = [
            self.system.policy_for(plan, agent)
            for agent in self.system.topology.agent_names
        ]
        active_counts = [
            average(local.active_count for local in iter_policy_mixtures(policy))
            for policy in policies
        ]
        fusion_rank_proxy = average(
            self.system.backend.estimate_fusion_rank(policy)
            for policy in policies
        )
        model_latency_seconds = average(run.latency_seconds for run in runs)
        metrics = self.metric_catalog.aggregate_quality(group_scores)
        metrics.update({
            # Complete per-case Agent-system wall time, including tokenization,
            # candidate materialization, reuse, and final temporary release.
            "latency_seconds": candidate_session_seconds / len(runs),
            "model_latency_seconds": model_latency_seconds,
            "candidate_session_seconds": candidate_session_seconds,
            "fusion_materialization_seconds": float(
                fusion_runtime.get("materialization_seconds", 0.0)
            ),
            "fusion_release_seconds": float(
                fusion_runtime.get("release_seconds", 0.0)
            ),
            "prompt_tokens": average(run.prompt_tokens for run in runs),
            "completion_tokens": average(
                run.completion_tokens for run in runs
            ),
            "total_tokens": average(run.total_tokens for run in runs),
            "active_adapters": average(active_counts),
            "fusion_rank_proxy": fusion_rank_proxy,
        })
        return EvaluationResult(
            metrics=metrics,
            details={
                "statistics": {
                    "case_count": len(scores),
                    "score": _score_statistics(scores),
                    "groups": group_statistics,
                    "output": {
                        "character_count": _numeric_statistics(output_lengths),
                        "completion_token_count": _numeric_statistics(
                            output_completion_tokens
                        ),
                        "empty_count": output_empty_count,
                        "generation_limit_hit_count": output_limit_hit_count,
                        "repetitive_output_count": repetitive_output_count,
                    },
                    "system": {
                        "termination_reasons": _count_strings(
                            tuple(run.termination_reason for run in runs)
                        ),
                        "steps": _numeric_statistics(
                            tuple(run.steps for run in runs)
                        ),
                        "tool_error_count": sum(
                            run.tool_error_count for run in runs
                        ),
                    },
                },
                "inference_batch_size": self.batch_size,
                "_runtime": {
                    "candidate_session_seconds": candidate_session_seconds,
                    "model_latency_seconds_per_case": model_latency_seconds,
                    "fusion": fusion_runtime,
                },
            },
        )

    def release_unreferenced_adapters(
        self, reference_counts: Mapping[str, int]
    ) -> Mapping[str, Any]:
        return self.system.backend.release_unreferenced_adapters(
            reference_counts
        )


def _score_statistics(values: Sequence[float]) -> Mapping[str, Any]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": average(numeric),
        "min": min(numeric) if numeric else 0.0,
        "max": max(numeric) if numeric else 0.0,
        "perfect_count": sum(value >= 1.0 for value in numeric),
        "zero_count": sum(value <= 0.0 for value in numeric),
    }


def _numeric_statistics(values: Sequence[int]) -> Mapping[str, Any]:
    numeric = [int(value) for value in values]
    return {
        "count": len(numeric),
        "mean": average(numeric),
        "max": max(numeric) if numeric else 0,
        "total": sum(numeric),
    }


def _count_strings(values: Sequence[str]) -> Mapping[str, int]:
    counts = {}  # type: Dict[str, int]
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _is_repetitive_output(value: str) -> bool:
    """Flag likely decoding degeneration without persisting the text."""

    if len(value) < 16:
        return False
    longest_run = max(
        (sum(1 for _ in run) for _, run in groupby(value)),
        default=0,
    )
    return longest_run / len(value) >= 0.9


def _fusion_runtime_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return per-candidate deltas from cumulative materializer counters."""

    before_materializer = dict(
        before.get("fusion", {}).get("materializer", {})
    )
    after_materializer = dict(
        after.get("fusion", {}).get("materializer", {})
    )
    integer_fields = (
        "materialization_count",
        "release_count",
        "target_allocation_count",
        "copied_module_count",
        "factor_copy_operations",
        "factor_bytes_copied",
        "registration_count",
        "unregistration_count",
    )
    float_fields = (
        "materialization_seconds",
        "release_seconds",
        "plan_compilation_seconds",
        "registration_seconds",
        "unregistration_seconds",
    )
    delta = {
        field: int(after_materializer.get(field, 0))
        - int(before_materializer.get(field, 0))
        for field in integer_fields
    }
    delta.update(
        {
            field: max(
                0.0,
                float(after_materializer.get(field, 0.0))
                - float(before_materializer.get(field, 0.0)),
            )
            for field in float_fields
        }
    )
    delta["resident_temporary_count_after"] = int(
        after.get("fusion", {}).get("resident_temporary_count", 0)
    )
    delta["unreleased_adapter_names"] = list(
        after.get("fusion", {}).get("unreleased_adapter_names", ())
    )
    return delta


class SyntheticMultiObjectiveEvaluator(FusionPlanEvaluator):
    """Deterministic proxy problem for verifying the complete search pipeline.

    Each agent has a hidden sparse adapter preference. ``quality`` rewards
    proximity to that preference, while ``robustness`` is the worst role score
    and ``active_adapters`` exposes a real sparsity trade-off. This evaluator is
    not a scientific benchmark.
    """

    def __init__(
        self,
        agent_names: Sequence[str],
        adapter_names: Sequence[str],
        seed: int = 42,
    ) -> None:
        self.agent_names = tuple(agent_names)
        self.adapter_names = tuple(adapter_names)
        self.targets = self._build_targets(seed)

    @property
    def metric_names(self) -> Tuple[str, ...]:
        return SYNTHETIC_METRIC_NAMES

    def _build_targets(self, seed: int) -> Dict[str, List[float]]:
        targets = {}
        count = len(self.adapter_names)
        for agent_name in self.agent_names:
            digest = hashlib.sha256(
                "{}:{}".format(seed, agent_name).encode("utf-8")
            ).digest()
            local_seed = int.from_bytes(digest[:8], "big")
            rng = random.Random(local_seed)
            active_count = min(count, 2)
            active = rng.sample(range(count), active_count)
            raw = [rng.uniform(0.2, 1.0) for _ in active]
            total = sum(raw)
            vector = [0.0] * count
            for index, value in zip(active, raw):
                vector[index] = value / total
            targets[agent_name] = vector
        return targets

    def evaluate(self, plan: FusionPlan) -> EvaluationResult:
        role_scores = {}
        active_counts = []
        role_vectors = []
        for agent_name in self.agent_names:
            policy = plan.policy_for(agent_name)
            if tuple(policy.adapter_names) != self.adapter_names:
                raise ValueError("FusionPlan and evaluator Adapter pools differ")
            representative = average_policy_mixture(policy)
            weights = representative.weights
            target = self.targets[agent_name]
            squared_distance = sum(
                (weight - desired) ** 2
                for weight, desired in zip(weights, target)
            )
            role_scores[agent_name] = max(0.0, 1.0 - squared_distance / 2.0)
            active_counts.append(
                average(
                    local.active_count
                    for local in iter_policy_mixtures(policy)
                )
            )
            role_vectors.append(weights)

        mean_quality = average(role_scores.values())
        worst_quality = min(role_scores.values())
        diversity_bonus = _mean_pairwise_l1(
            role_vectors
        )
        quality = min(1.0, mean_quality + 0.03 * diversity_bonus)
        active_adapters = average(active_counts)
        metrics = {
            "quality": quality,
            "robustness": worst_quality,
            "worst_group_quality": worst_quality,
            "active_adapters": active_adapters,
            "latency_proxy": 1.0 + 0.08 * active_adapters,
            "fusion_rank_proxy": 16.0 * active_adapters,
        }
        return EvaluationResult(
            metrics=metrics,
            details={
                "role_scores": role_scores,
                "synthetic": True,
            },
        )


def build_scorer(config: BenchmarkConfig) -> OutputScorer:
    name = str(config.options.get("scorer", "exact_match")).lower()
    return build_scorer_by_name(name, config.options)


def build_scorer_by_name(
    name: str, options: Mapping[str, Any]
) -> OutputScorer:
    if name == "exact_match":
        return ExactMatchScorer()
    if name == "numeric":
        return NumericAnswerScorer(
            tolerance=float(options.get("tolerance", 1e-6))
        )
    if name in {"hotpot_f1", "token_f1"}:
        return HotpotF1Scorer()
    if name in {"math", "math_answer"}:
        return MathAnswerScorer(
            tolerance=float(options.get("tolerance", 1e-6))
        )
    if name in {"python_literal", "literal"}:
        return PythonLiteralScorer()
    if name in {"choice", "classification"}:
        return ChoiceAnswerScorer()
    if name in {"python_code", "humaneval", "mbpp"}:
        return PythonCodeScorer(
            timeout_seconds=float(options.get("code_timeout_seconds", 3.0))
        )
    if name in {"bfcl_ast", "function_call"}:
        return BfclAstScorer()
    if name in {"grid_path", "plan_path"}:
        return GridPathScorer()
    if name == "sudoku":
        return SudokuScorer()
    if name == "sokoban":
        return SokobanScorer()
    raise ValueError("Unknown scorer {!r}".format(name))


def _extract_python_literal(value: str) -> Any:
    stripped = value.strip()
    candidates = [stripped, _extract_final_answer(stripped)]
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", stripped, re.DOTALL)
    candidates.extend(reversed(fenced))
    candidates.extend(reversed(stripped.splitlines()))
    for candidate in candidates:
        candidate = candidate.strip().rstrip(".")
        if not candidate:
            continue
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
    return None


def _extract_function_calls(value: str) -> List[Dict[str, Dict[str, Any]]]:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", value, re.DOTALL)
    candidates = list(reversed(fenced))
    candidates.extend(reversed(value.strip().splitlines()))
    candidates.append(value.strip())
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            expression = ast.parse(candidate, mode="eval").body
            nodes = (
                list(expression.elts)
                if isinstance(expression, (ast.List, ast.Tuple))
                else [expression]
            )
            calls = []
            for node in nodes:
                if not isinstance(node, ast.Call) or node.args:
                    raise ValueError("Only keyword function calls are supported")
                name = _function_name(node.func)
                if not name or any(keyword.arg is None for keyword in node.keywords):
                    raise ValueError("Invalid function call")
                calls.append(
                    {
                        name: {
                            keyword.arg: ast.literal_eval(keyword.value)
                            for keyword in node.keywords
                        }
                    }
                )
            return calls
        except (SyntaxError, ValueError):
            continue
    return []


def _function_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _bfcl_value_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        normalize = lambda value: re.sub(
            r"[ ,./\-_*^]", "", value
        ).casefold().replace("'", '"')
        return normalize(actual) == normalize(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _bfcl_value_equal(left, right)
            for left, right in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _bfcl_value_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6
        )
    return actual == expected


_MOVE_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _extract_moves(value: str) -> Tuple[str, ...]:
    candidates = list(reversed(value.strip().splitlines()))
    candidates.append(value.strip())
    for candidate in candidates:
        compact = re.sub(r"[\s,;\-\>]", "", candidate.upper())
        if compact and all(character in _MOVE_DELTAS for character in compact):
            return tuple(compact)
    return ()


def _move(position: Tuple[int, int], direction: str) -> Tuple[int, int]:
    row_delta, column_delta = _MOVE_DELTAS[direction]
    return position[0] + row_delta, position[1] + column_delta


def _find_grid_cell(rows: Sequence[str], marker: str) -> Tuple[int, int]:
    matches = [
        (row_index, column_index)
        for row_index, row in enumerate(rows)
        for column_index, value in enumerate(row)
        if value == marker
    ]
    if len(matches) != 1:
        raise ValueError("Grid must contain exactly one {!r}".format(marker))
    return matches[0]


def _is_open_grid_cell(
    rows: Sequence[str], position: Tuple[int, int]
) -> bool:
    row, column = position
    return (
        0 <= row < len(rows)
        and 0 <= column < len(rows[row])
        and rows[row][column] != "#"
    )


def _parse_sokoban_grid(rows: Sequence[str]):
    player = None
    boxes = set()
    goals = set()
    walls = set()
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            position = (row_index, column_index)
            if value == "#":
                walls.add(position)
            if value in {"@", "+"}:
                if player is not None:
                    raise ValueError("Sokoban grid has multiple players")
                player = position
            if value in {"$", "*"}:
                boxes.add(position)
            if value in {".", "*", "+"}:
                goals.add(position)
    if player is None or not boxes or len(boxes) != len(goals):
        raise ValueError("Invalid Sokoban grid")
    return player, boxes, goals, walls


def _normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().rstrip(".").split())


def _hotpot_normalize(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _extract_final_answer(value: str) -> str:
    boxed_start = value.rfind(r"\boxed{")
    if boxed_start >= 0:
        content_start = boxed_start + len(r"\boxed{")
        depth = 1
        for index in range(content_start, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    return value[content_start:index]
    markers = re.findall(
        r"(?:final answer|answer)\s*(?:is|:)\s*(.+)",
        value,
        flags=re.IGNORECASE,
    )
    if markers:
        return markers[-1].strip().splitlines()[0]
    return value.strip().splitlines()[-1] if value.strip() else ""


def _normalize_math(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\,", "").replace("\\!", "")
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("\\%", "%")
    normalized = re.sub(r"\\text\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.rstrip(".")


def _parse_simple_number(value: str) -> Optional[float]:
    candidate = value
    percent = candidate.endswith("%")
    if percent:
        candidate = candidate[:-1]
    latex_fraction = re.fullmatch(
        r"([-+]?)\\(?:d?frac)\{([-+]?\d+)\}\{([-+]?\d+)\}",
        candidate,
    )
    if latex_fraction:
        sign, numerator, denominator = latex_fraction.groups()
        fraction = Fraction(int(numerator), int(denominator))
        result = float(-fraction if sign == "-" else fraction)
    else:
        simple_fraction = re.fullmatch(r"([-+]?\d+)/([-+]?\d+)", candidate)
        if simple_fraction:
            result = float(
                Fraction(
                    int(simple_fraction.group(1)),
                    int(simple_fraction.group(2)),
                )
            )
        else:
            try:
                result = float(candidate)
            except ValueError:
                return None
    return result / 100.0 if percent else result


def _mean_pairwise_l1(vectors: Sequence[Sequence[float]]) -> float:
    distances = []
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            distances.append(
                sum(
                    abs(left - right)
                    for left, right in zip(
                        vectors[left_index], vectors[right_index]
                    )
                )
                / 2.0
            )
    return average(distances)
