import json
from contextlib import contextmanager

import pytest

from agentmerge.evaluation.system import FixedTopology, MultiAgentSystem
from agentmerge.backends import MockModelBackend
from agentmerge.core.config import (
    AdapterConfig,
    AgentConfig,
    BenchmarkConfig,
    GenerationConfig,
    TopologyConfig,
)
from agentmerge.core.domain import AdapterMixture
from agentmerge.core.policies import FusionPlan
from agentmerge.evaluation.benchmarks import (
    BfclAstScorer,
    ChoiceAnswerScorer,
    GridPathScorer,
    HotpotF1Scorer,
    JsonlBenchmark,
    MathAnswerScorer,
    PythonLiteralScorer,
    PythonCodeScorer,
    SokobanScorer,
    SudokuScorer,
    SystemBenchmarkEvaluator,
    _fusion_runtime_delta,
    build_scorer,
)


def test_hotpot_f1_uses_normalized_token_overlap():
    scorer = HotpotF1Scorer()
    assert scorer.score("The Eiffel Tower.", "Eiffel Tower") == 1.0
    assert scorer.score("Paris, France", "Paris") == pytest.approx(2.0 / 3.0)
    assert scorer.score("London", "Paris") == 0.0


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        ("The final answer is 0.5", r"\frac{1}{2}"),
        (r"\boxed{\frac{3}{4}}", "0.75"),
        ("Answer: 25%", "0.25"),
        ("42", "42"),
    ],
)
def test_math_answer_scorer_handles_simple_equivalence(prediction, target):
    assert MathAnswerScorer().score(prediction, target) == 1.0


def test_python_literal_scorer_is_structural_and_never_executes_code():
    scorer = PythonLiteralScorer()
    assert scorer.score("Final answer: {'b': 2, 'a': 1}", "{'a': 1, 'b': 2}") == 1.0
    assert scorer.score("[1, 2, 3]", "(1, 2, 3)") == 0.0
    assert scorer.score("__import__('os').system('false')", "0") == 0.0


def test_choice_scorer_supports_nli_bbh_options_and_concise_bbh_answers():
    scorer = ChoiceAnswerScorer()
    assert scorer.score("The relation is entailment.", "entailment") == 1.0
    assert scorer.score("Reasoning... Final answer: (C)", "(C)") == 1.0
    assert scorer.score("Reasoning... Final answer: True", "True") == 1.0
    assert scorer.score("The statement mentions True, but answer: False", "True") == 0.0


def test_python_code_scorer_runs_base_tests_outside_evaluator_process():
    scorer = PythonCodeScorer(timeout_seconds=2.0)
    mbpp_target = json.dumps(
        {
            "kind": "mbpp",
            "entry_point": "double",
            "tests": ["assert double(3) == 6", "assert double(-2) == -4"],
            "test_setup": "",
        }
    )
    humaneval_target = json.dumps(
        {
            "kind": "humaneval",
            "prompt": "def add_one(value):\n",
            "test": "def check(candidate):\n    assert candidate(4) == 5\n",
            "entry_point": "add_one",
        }
    )
    assert scorer.score("def double(value):\n    return value * 2", mbpp_target) == 1.0
    assert scorer.score("def double(value):\n    return value", mbpp_target) == 0.0
    assert scorer.score(
        "Here is the implementation:\n"
        "```python\ndef double(value):\n    return value * 2\n```\n"
        "A usage example follows:\n```python\nprint(double(3))\n```",
        mbpp_target,
    ) == 1.0
    assert scorer.score(
        "def double(value):\n    return value * 2\n\n"
        "if __name__ == '__main__':\n    raise RuntimeError('demo only')",
        mbpp_target,
    ) == 1.0
    assert scorer.score("    return value + 1", humaneval_target) == 1.0


def test_bfcl_ast_scorer_accepts_valid_single_keyword_call_only():
    scorer = BfclAstScorer()
    target = json.dumps(
        {
            "functions": [
                {
                    "name": "weather",
                    "parameters": {
                        "required": ["city", "tags"],
                        "properties": {"city": {}, "unit": {}, "tags": {}},
                    },
                }
            ],
            "ground_truth": [
                {
                    "weather": {
                        "city": ["Paris"],
                        "unit": ["", "celsius"],
                        "tags": [],
                    }
                }
            ],
        }
    )
    assert scorer.score("weather(city='Paris', tags=[])", target) == 1.0
    assert scorer.score("weather(city='London', tags=[])", target) == 0.0
    assert scorer.score("weather('Paris')", target) == 0.0


def test_grid_path_scorer_executes_moves_and_rejects_walls():
    scorer = GridPathScorer()
    target = json.dumps(
        {"grid": ["#####", "#S#G#", "#...#", "#####"], "max_steps": 8}
    )
    assert scorer.score("D R R U", target) == 1.0
    assert scorer.score("R R", target) == 0.0


def test_sudoku_scorer_validates_clues_rows_columns_and_boxes():
    scorer = SudokuScorer()
    target = json.dumps(
        {
            "puzzle": [
                [1, 0, 0, 4],
                [0, 4, 1, 0],
                [0, 1, 4, 0],
                [4, 0, 0, 1],
            ]
        }
    )
    solution = "[[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]"
    assert scorer.score(solution, target) == 1.0
    assert scorer.score("[[1, 2, 3, 4]]", target) == 0.0


def test_sokoban_scorer_accepts_any_valid_solution_sequence():
    scorer = SokobanScorer()
    target = json.dumps(
        {"grid": ["#####", "#@$.#", "#####"], "max_steps": 3}
    )
    assert scorer.score("R", target) == 1.0
    assert scorer.score("L", target) == 0.0


def test_jsonl_benchmark_supports_per_case_scorers(tmp_path):
    path = tmp_path / "mixed.jsonl"
    rows = [
        {
            "id": "qa-1",
            "input": "question",
            "target": "The Eiffel Tower",
            "group": "qa",
            "scorer": "hotpot_f1",
            "source": "unit",
        },
        {
            "id": "math-1",
            "input": "problem",
            "target": r"\frac{1}{2}",
            "group": "math",
            "scorer": "math",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    config = BenchmarkConfig(
        kind="jsonl",
        path=str(path),
        options={"scorer": "exact_match", "scorer_field": "scorer"},
    )
    benchmark = JsonlBenchmark(config, build_scorer(config))

    assert benchmark.score(benchmark.cases[0], "Eiffel Tower") == 1.0
    assert benchmark.score(benchmark.cases[1], "0.5") == 1.0
    assert benchmark.cases[0].metadata == {"source": "unit"}


class _RecordingBatchBackend(MockModelBackend):
    def __init__(self, adapters):
        super().__init__(adapters)
        self.batch_sizes = []
        self.session_count = 0

    def generate_many(self, requests, mixtures):
        self.batch_sizes.append(len(requests))
        return super().generate_many(requests, mixtures)

    @contextmanager
    def candidate_session(self):
        self.session_count += 1
        yield


def test_system_evaluator_microbatches_cases_by_topology_stage(tmp_path):
    path = tmp_path / "batched.jsonl"
    rows = [
        {
            "id": str(index),
            "input": "Return [expected: {}]".format(index),
            "target": str(index),
        }
        for index in range(5)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    benchmark_config = BenchmarkConfig(
        kind="jsonl",
        path=str(path),
        options={"scorer": "exact_match", "inference_batch_size": 2},
    )
    benchmark = JsonlBenchmark(
        benchmark_config, build_scorer(benchmark_config)
    )
    adapters = (AdapterConfig("a", "a", rank=8),)
    backend = _RecordingBatchBackend(adapters)
    agents = (
        AgentConfig("first", "planner", "plan"),
        AgentConfig("last", "critic", "final"),
    )
    topology = FixedTopology(
        ("first", "last"),
        TopologyConfig(
            edges=(("first", "last"),),
            output_agent="last",
            kind="fixed",
        ),
    )
    system = MultiAgentSystem(
        backend, agents, topology, GenerationConfig(seed=3)
    )
    evaluator = SystemBenchmarkEvaluator(system, benchmark)

    result = evaluator.evaluate(
        FusionPlan(
            (
                ("first", AdapterMixture(("a",), (1.0,))),
                ("last", AdapterMixture(("a",), (1.0,))),
            )
        )
    )

    assert result.metrics["quality"] == 1.0
    assert backend.batch_sizes == [2, 2, 2, 2, 1, 1]
    assert result.details["inference_batch_size"] == 2
    assert "cases" not in result.details
    assert result.details["statistics"]["case_count"] == 5
    assert "prediction" not in json.dumps(result.details)
    assert result.metrics["candidate_session_seconds"] > 0.0
    assert result.metrics["latency_seconds"] > 0.0
    assert result.details["_runtime"]["fusion"]["materialization_count"] == 0
    assert result.details["statistics"]["system"] == {
        "termination_reasons": {"completed": 5},
        "steps": {"count": 5, "mean": 2.0, "max": 2, "total": 10},
        "tool_error_count": 0,
    }
    assert backend.session_count == 1


def test_fusion_runtime_delta_reports_candidate_lifecycle_totals():
    before = {
        "fusion": {
            "resident_temporary_count": 0,
            "materializer": {
                "materialization_count": 3,
                "materialization_seconds": 1.0,
                "release_count": 3,
                "release_seconds": 0.2,
                "factor_bytes_copied": 100,
            },
        }
    }
    after = {
        "fusion": {
            "resident_temporary_count": 0,
            "unreleased_adapter_names": [],
            "materializer": {
                "materialization_count": 5,
                "materialization_seconds": 1.5,
                "release_count": 5,
                "release_seconds": 0.3,
                "factor_bytes_copied": 340,
            },
        }
    }

    delta = _fusion_runtime_delta(before, after)

    assert delta["materialization_count"] == 2
    assert delta["release_count"] == 2
    assert delta["materialization_seconds"] == pytest.approx(0.5)
    assert delta["release_seconds"] == pytest.approx(0.1)
    assert delta["factor_bytes_copied"] == 240
    assert delta["resident_temporary_count_after"] == 0
