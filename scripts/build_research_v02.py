#!/usr/bin/env python3
"""Build the harder, multi-domain AgentMerge research benchmark suite."""

from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
SELECTION_SEED = 20260814
PAGE_SIZE = 100
SPLIT_COUNTS = {"search": 28, "validation": 40, "test": 72}
TOTAL_PER_TASK = sum(SPLIT_COUNTS.values())


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    config: str
    split: str
    row_count: int
    id_field: str


SOURCES = {
    "math_olympiad": SourceSpec(
        "Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "train", 674, "id"
    ),
    "qa_hotpot": SourceSpec(
        "hotpotqa/hotpot_qa", "distractor", "validation", 1200, "id"
    ),
    "qa_musique": SourceSpec(
        "dgslibisey/MuSiQue", "default", "validation", 2417, "id"
    ),
    "code_cruxeval": SourceSpec(
        "cruxeval-org/cruxeval", "default", "test", 800, "id"
    ),
    "logic_deduction": SourceSpec(
        "tasksource/bigbench", "logical_deduction", "validation", 300, "idx"
    ),
}


class ResearchV02Builder:
    """Fetch, filter and deterministically split eight balanced task groups."""

    def __init__(self, output_dir: Path, cache_dir: Path) -> None:
        self.output_dir = output_dir
        self.cache_dir = cache_dir

    def build(self) -> Mapping[str, Any]:
        remote_rows = {
            task: self._prepare_remote_rows(task, self._fetch_source(spec))
            for task, spec in SOURCES.items()
        }
        selected = {
            task: self._select_rows(task, rows, SOURCES[task].id_field)
            for task, rows in remote_rows.items()
        }
        generated = {
            "planning_path": self._generate_plan_paths(),
            "game_sudoku": self._generate_sudokus(),
            "game_sokoban": self._generate_sokoban(),
        }
        selected.update(generated)

        split_rows: Dict[str, List[Mapping[str, Any]]] = {
            name: [] for name in SPLIT_COUNTS
        }
        split_ids: Dict[str, Dict[str, List[str]]] = {
            name: {} for name in SPLIT_COUNTS
        }
        cursor = 0
        task_names = tuple(selected)
        for split_name, count in SPLIT_COUNTS.items():
            for task in task_names:
                rows = selected[task][cursor : cursor + count]
                split_ids[split_name][task] = [str(row["id"]) for row in rows]
            # Round-robin ordering keeps every --max-cases prefix balanced
            # whenever the limit is a multiple of the task count.
            for local_index in range(count):
                for task in task_names:
                    split_rows[split_name].append(
                        selected[task][cursor + local_index]
                    )
            cursor += count

        for split_name, rows in split_rows.items():
            self._write_jsonl(
                self.output_dir / "research_v02_{}.jsonl".format(split_name),
                rows,
            )

        manifest = {
            "schema_version": 2,
            "selection_seed": SELECTION_SEED,
            "benchmark_profile": "agent_finetuning_multidomain_hard",
            "reference_design": {
                "primary": "Stronger-MAS / PettingLLMs",
                "domains": ["math", "code", "planning", "game", "multi_hop_qa"],
                "local_paper": (
                    "references/papers/Zhao 等 - 2026 - Stronger-MAS "
                    "Multi-Agent Reinforcement Learning for Collaborative LLMs.pdf"
                ),
            },
            "split_counts_per_task": SPLIT_COUNTS,
            "task_count": len(task_names),
            "tasks": {
                "math_olympiad": {
                    "source": SOURCES["math_olympiad"].dataset,
                    "filter": "English text-only competition math, numerical single answer",
                    "difficulty": "olympiad",
                },
                "qa_hotpot": {
                    "source": SOURCES["qa_hotpot"].dataset,
                    "filter": "distractor validation, hard two-hop",
                    "difficulty": "hard multi-hop",
                },
                "qa_musique": {
                    "source": SOURCES["qa_musique"].dataset,
                    "filter": "answerable 3-hop and 4-hop with distractors",
                    "difficulty": "hard compositional multi-hop",
                },
                "code_cruxeval": {
                    "source": SOURCES["code_cruxeval"].dataset,
                    "filter": "top half by Python AST node count; literal outputs only",
                    "difficulty": "hard code execution reasoning",
                },
                "logic_deduction": {
                    "source": SOURCES["logic_deduction"].dataset,
                    "filter": "seven-object logical deduction only",
                    "difficulty": "BBH hard",
                },
                "planning_path": {
                    "source": "deterministic generator",
                    "filter": "10x10, unique shortest path, >=18 moves and >=5 turns",
                    "difficulty": "long-horizon planning",
                },
                "game_sudoku": {
                    "source": "deterministic generator",
                    "filter": "4x4, six clues, unique solution",
                    "difficulty": "constraint satisfaction",
                },
                "game_sokoban": {
                    "source": "deterministic generator",
                    "filter": "6x6, two boxes, shortest solution >=8 moves",
                    "difficulty": "long-horizon state planning",
                },
            },
            "splits": {
                split_name: {
                    "total_count": len(split_rows[split_name]),
                    "per_task_count": SPLIT_COUNTS[split_name],
                    "ids": split_ids[split_name],
                }
                for split_name in SPLIT_COUNTS
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "research_v02_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _fetch_source(self, spec: SourceSpec) -> List[Mapping[str, Any]]:
        rows: List[Mapping[str, Any]] = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stem = "{}--{}--{}".format(
            spec.dataset.replace("/", "--"), spec.config, spec.split
        )
        for offset in range(0, spec.row_count, PAGE_SIZE):
            length = min(PAGE_SIZE, spec.row_count - offset)
            cache_path = self.cache_dir / "{}-{:04d}.json".format(stem, offset)
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                payload = self._download_page(spec, offset, length)
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                time.sleep(0.15)
            rows.extend(entry["row"] for entry in payload["rows"])
        if len(rows) != spec.row_count:
            raise RuntimeError(
                "{} returned {} rows, expected {}".format(
                    spec.dataset, len(rows), spec.row_count
                )
            )
        return rows

    @staticmethod
    def _download_page(
        spec: SourceSpec, offset: int, length: int
    ) -> Mapping[str, Any]:
        query = urllib.parse.urlencode(
            {
                "dataset": spec.dataset,
                "config": spec.config,
                "split": spec.split,
                "offset": offset,
                "length": length,
            }
        )
        request = urllib.request.Request(
            "{}?{}".format(DATASETS_SERVER, query),
            headers={"User-Agent": "AgentMerge-benchmark-builder/0.3"},
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 5:
                    raise
                time.sleep(2 ** attempt)
            except urllib.error.URLError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Unreachable download retry state")

    def _prepare_remote_rows(
        self, task: str, rows: Sequence[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        if task == "math_olympiad":
            eligible = [
                row
                for row in rows
                if row.get("answer_type") == "Numerical"
                and not row.get("is_multiple_answer")
                and len(row.get("final_answer") or ()) == 1
            ]
        elif task == "qa_hotpot":
            eligible = [row for row in rows if row.get("level") == "hard"]
        elif task == "qa_musique":
            eligible = [
                row
                for row in rows
                if row.get("answerable")
                and str(row.get("id", "")).startswith(("3hop", "4hop"))
            ]
        elif task == "code_cruxeval":
            eligible = [
                row
                for row in rows
                if self._literal_valid(str(row.get("output", "")))
            ]
            eligible.sort(
                key=lambda row: (
                    -self._ast_complexity(str(row.get("code", ""))),
                    str(row.get("id", "")),
                )
            )
            eligible = eligible[: max(400, TOTAL_PER_TASK)]
        elif task == "logic_deduction":
            eligible = [
                row
                for row in rows
                if "set of seven objects" in str(row.get("inputs", ""))
            ]
        else:
            raise ValueError("Unknown remote task {!r}".format(task))
        if len(eligible) < TOTAL_PER_TASK:
            raise RuntimeError(
                "{} has {} eligible rows, {} required".format(
                    task, len(eligible), TOTAL_PER_TASK
                )
            )
        return eligible

    def _select_rows(
        self,
        task: str,
        rows: Sequence[Mapping[str, Any]],
        id_field: str,
    ) -> List[Mapping[str, Any]]:
        ordered = sorted(
            rows,
            key=lambda row: (
                self._stable_hash(task, str(row[id_field])),
                str(row[id_field]),
            ),
        )[:TOTAL_PER_TASK]
        converter = getattr(self, "_convert_{}".format(task))
        return [converter(row) for row in ordered]

    @staticmethod
    def _convert_math_olympiad(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": "math_olympiad:{}".format(row["id"]),
            "input": (
                "Solve this olympiad-level mathematics problem. Return only "
                "the final answer, without derivation.\n\n{}"
            ).format(row["question"]),
            "target": str(row["final_answer"][0]),
            "group": "math_olympiad",
            "scorer": "math",
            "source": SOURCES["math_olympiad"].dataset,
            "source_id": str(row["id"]),
            "difficulty": row.get("difficulty"),
            "subject": row.get("subject"),
            "subfield": row.get("subfield"),
        }

    @staticmethod
    def _convert_qa_hotpot(row: Mapping[str, Any]) -> Mapping[str, Any]:
        passages = [
            "[{}]\n{}".format(title, " ".join(sentences).strip())
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"]
            )
        ]
        return {
            "id": "qa_hotpot:{}".format(row["id"]),
            "input": (
                "Answer the multi-hop question using the passages. Return only "
                "a concise answer.\n\nPassages:\n{}\n\nQuestion: {}"
            ).format("\n\n".join(passages), row["question"]),
            "target": row["answer"],
            "group": "qa_hotpot",
            "scorer": "hotpot_f1",
            "source": SOURCES["qa_hotpot"].dataset,
            "source_id": row["id"],
            "difficulty": "hard",
            "question_type": row.get("type"),
        }

    @staticmethod
    def _convert_qa_musique(row: Mapping[str, Any]) -> Mapping[str, Any]:
        passages = [
            "[{}]\n{}".format(item["title"], item["paragraph_text"])
            for item in row["paragraphs"]
        ]
        hop_count = int(str(row["id"])[0])
        return {
            "id": "qa_musique:{}".format(row["id"]),
            "input": (
                "Answer the compositional multi-hop question using the passages. "
                "Resolve all required hops and return only a concise answer.\n\n"
                "Passages:\n{}\n\nQuestion: {}"
            ).format("\n\n".join(passages), row["question"]),
            "target": row["answer"],
            "group": "qa_musique",
            "scorer": "hotpot_f1",
            "source": SOURCES["qa_musique"].dataset,
            "source_id": row["id"],
            "hop_count": hop_count,
            "distractor_count": sum(
                not item["is_supporting"] for item in row["paragraphs"]
            ),
        }

    @staticmethod
    def _convert_code_cruxeval(row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": "code_cruxeval:{}".format(row["id"]),
            "input": (
                "Determine the exact Python value returned by f for the given "
                "input. Do not execute code. Return only one valid Python "
                "literal.\n\nCode:\n```python\n{}\n```\n\nCall: f({})"
            ).format(row["code"], row["input"]),
            "target": row["output"],
            "group": "code_cruxeval",
            "scorer": "python_literal",
            "source": SOURCES["code_cruxeval"].dataset,
            "source_id": row["id"],
            "ast_node_count": ResearchV02Builder._ast_complexity(row["code"]),
        }

    @staticmethod
    def _convert_logic_deduction(row: Mapping[str, Any]) -> Mapping[str, Any]:
        choices = "\n".join(
            "{}. {}".format(chr(65 + index), choice)
            for index, choice in enumerate(row["multiple_choice_targets"])
        )
        return {
            "id": "logic_deduction:{}".format(row["idx"]),
            "input": (
                "Solve the seven-object ordering problem. Return only the exact "
                "text of the correct option.\n\n{}\n\nOptions:\n{}"
            ).format(row["inputs"], choices),
            "target": row["targets"][0],
            "group": "logic_deduction",
            "scorer": "exact_match",
            "source": SOURCES["logic_deduction"].dataset,
            "source_id": str(row["idx"]),
            "object_count": 7,
        }

    def _generate_plan_paths(self) -> List[Mapping[str, Any]]:
        cases = []
        seen = set()
        candidate_seed = 0
        while len(cases) < TOTAL_PER_TASK:
            rng = random.Random(SELECTION_SEED * 100000 + candidate_seed)
            candidate_seed += 1
            generated = self._plan_path_candidate(rng)
            if generated is None:
                continue
            rows, path = generated
            signature = tuple(rows)
            if signature in seen:
                continue
            seen.add(signature)
            case_index = len(cases)
            cases.append(
                {
                    "id": "planning_path:{:04d}".format(case_index),
                    "input": (
                        "Navigate from S to G in the 10x10 grid. # cells are "
                        "blocked. Return only a sequence of U, D, L, R moves; "
                        "spaces are optional.\n\n{}"
                    ).format("\n".join(rows)),
                    "target": json.dumps(
                        {"grid": rows, "max_steps": len(path) + 12},
                        separators=(",", ":"),
                    ),
                    "group": "planning_path",
                    "scorer": "grid_path",
                    "source": "agentmerge/generated-plan-path-v1",
                    "source_id": str(case_index),
                    "shortest_path_length": len(path),
                    "turn_count": self._turn_count(path),
                }
            )
        return cases

    @staticmethod
    def _plan_path_candidate(
        rng: random.Random,
    ) -> Optional[Tuple[List[str], Tuple[str, ...]]]:
        size = 10
        cells = [(row, column) for row in range(size) for column in range(size)]
        start, goal = rng.sample(cells, 2)
        walls = {
            cell
            for cell in cells
            if cell not in {start, goal} and rng.random() < 0.28
        }
        result = _unique_shortest_grid_path(size, start, goal, walls)
        if result is None:
            return None
        path, shortest_path_count = result
        if (
            shortest_path_count != 1
            or len(path) < 18
            or len(path) > 42
            or ResearchV02Builder._turn_count(path) < 5
        ):
            return None
        rows = []
        for row in range(size):
            rows.append(
                "".join(
                    "S"
                    if (row, column) == start
                    else "G"
                    if (row, column) == goal
                    else "#"
                    if (row, column) in walls
                    else "."
                    for column in range(size)
                )
            )
        return rows, path

    def _generate_sudokus(self) -> List[Mapping[str, Any]]:
        cases = []
        seen = set()
        candidate_seed = 0
        while len(cases) < TOTAL_PER_TASK:
            rng = random.Random(SELECTION_SEED * 200000 + candidate_seed)
            candidate_seed += 1
            solution = _random_sudoku_solution(rng)
            puzzle = _make_unique_sudoku_puzzle(solution, rng, clue_count=6)
            if puzzle is None:
                continue
            signature = tuple(tuple(row) for row in puzzle)
            if signature in seen:
                continue
            seen.add(signature)
            case_index = len(cases)
            display = "\n".join(
                " ".join(str(value) if value else "." for value in row)
                for row in puzzle
            )
            cases.append(
                {
                    "id": "game_sudoku:{:04d}".format(case_index),
                    "input": (
                        "Solve the 4x4 Sudoku using digits 1-4. Each row, column "
                        "and 2x2 box must contain every digit once. Return only "
                        "the completed grid as a Python nested list.\n\n{}"
                    ).format(display),
                    "target": json.dumps(
                        {"puzzle": puzzle}, separators=(",", ":")
                    ),
                    "group": "game_sudoku",
                    "scorer": "sudoku",
                    "source": "agentmerge/generated-sudoku-v1",
                    "source_id": str(case_index),
                    "grid_size": 4,
                    "clue_count": 6,
                }
            )
        return cases

    def _generate_sokoban(self) -> List[Mapping[str, Any]]:
        cases = []
        seen = set()
        candidate_seed = 0
        while len(cases) < TOTAL_PER_TASK:
            rng = random.Random(SELECTION_SEED * 300000 + candidate_seed)
            candidate_seed += 1
            generated = _random_sokoban_candidate(rng)
            if generated is None:
                continue
            rows, shortest_path = generated
            signature = tuple(rows)
            if signature in seen:
                continue
            seen.add(signature)
            case_index = len(cases)
            cases.append(
                {
                    "id": "game_sokoban:{:04d}".format(case_index),
                    "input": (
                        "Solve this 6x6 Sokoban puzzle. Push both boxes ($) onto "
                        "the goals (.). # is a wall and @ is the player. Return "
                        "only a U, D, L, R move sequence; spaces are optional.\n\n{}"
                    ).format("\n".join(rows)),
                    "target": json.dumps(
                        {
                            "grid": rows,
                            "max_steps": min(80, len(shortest_path) + 20),
                        },
                        separators=(",", ":"),
                    ),
                    "group": "game_sokoban",
                    "scorer": "sokoban",
                    "source": "agentmerge/generated-sokoban-v1",
                    "source_id": str(case_index),
                    "shortest_solution_length": len(shortest_path),
                    "box_count": 2,
                }
            )
        return cases

    @staticmethod
    def _stable_hash(task: str, identifier: str) -> str:
        return hashlib.sha256(
            "{}:{}:{}".format(SELECTION_SEED, task, identifier).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _literal_valid(value: str) -> bool:
        try:
            ast.literal_eval(value)
            return True
        except (SyntaxError, ValueError):
            return False

    @staticmethod
    def _ast_complexity(code: str) -> int:
        try:
            return sum(1 for _ in ast.walk(ast.parse(code)))
        except SyntaxError:
            return 0

    @staticmethod
    def _turn_count(path: Sequence[str]) -> int:
        return sum(left != right for left, right in zip(path, path[1:]))

    @staticmethod
    def _write_jsonl(
        path: Path, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


_DIRECTIONS = (
    ("U", (-1, 0)),
    ("D", (1, 0)),
    ("L", (0, -1)),
    ("R", (0, 1)),
)


def _unique_shortest_grid_path(size, start, goal, walls):
    queue = deque([start])
    distance = {start: 0}
    path_count = {start: 1}
    predecessor = {}
    while queue:
        current = queue.popleft()
        for move, (row_delta, column_delta) in _DIRECTIONS:
            neighbor = current[0] + row_delta, current[1] + column_delta
            if (
                not 0 <= neighbor[0] < size
                or not 0 <= neighbor[1] < size
                or neighbor in walls
            ):
                continue
            candidate_distance = distance[current] + 1
            if neighbor not in distance:
                distance[neighbor] = candidate_distance
                path_count[neighbor] = path_count[current]
                predecessor[neighbor] = current, move
                queue.append(neighbor)
            elif distance[neighbor] == candidate_distance:
                path_count[neighbor] = min(
                    2, path_count[neighbor] + path_count[current]
                )
    if goal not in distance:
        return None
    path = []
    current = goal
    while current != start:
        current, move = predecessor[current]
        path.append(move)
    path.reverse()
    return tuple(path), path_count[goal]


def _random_sudoku_solution(rng: random.Random) -> List[List[int]]:
    base = [[(row * 2 + row // 2 + column) % 4 + 1 for column in range(4)] for row in range(4)]
    digits = [1, 2, 3, 4]
    rng.shuffle(digits)
    mapped = [[digits[value - 1] for value in row] for row in base]
    row_bands = [[0, 1], [2, 3]]
    column_bands = [[0, 1], [2, 3]]
    rng.shuffle(row_bands)
    rng.shuffle(column_bands)
    for band in row_bands:
        rng.shuffle(band)
    for band in column_bands:
        rng.shuffle(band)
    row_order = [row for band in row_bands for row in band]
    column_order = [column for band in column_bands for column in band]
    return [[mapped[row][column] for column in column_order] for row in row_order]


def _make_unique_sudoku_puzzle(solution, rng, clue_count):
    puzzle = [list(row) for row in solution]
    positions = [(row, column) for row in range(4) for column in range(4)]
    rng.shuffle(positions)
    for row, column in positions:
        if sum(value != 0 for values in puzzle for value in values) <= clue_count:
            break
        old = puzzle[row][column]
        puzzle[row][column] = 0
        if _count_sudoku_solutions(puzzle, limit=2) != 1:
            puzzle[row][column] = old
    if sum(value != 0 for row in puzzle for value in row) != clue_count:
        return None
    return puzzle


def _count_sudoku_solutions(puzzle, limit):
    grid = [list(row) for row in puzzle]
    count = 0

    def search():
        nonlocal count
        if count >= limit:
            return
        best = None
        best_values = None
        for row in range(4):
            for column in range(4):
                if grid[row][column] != 0:
                    continue
                used = set(grid[row])
                used.update(grid[index][column] for index in range(4))
                box_row, box_column = 2 * (row // 2), 2 * (column // 2)
                used.update(
                    grid[index][other]
                    for index in range(box_row, box_row + 2)
                    for other in range(box_column, box_column + 2)
                )
                values = [value for value in range(1, 5) if value not in used]
                if not values:
                    return
                if best_values is None or len(values) < len(best_values):
                    best = row, column
                    best_values = values
        if best is None:
            count += 1
            return
        row, column = best
        for value in best_values:
            grid[row][column] = value
            search()
            grid[row][column] = 0

    search()
    return count


def _random_sokoban_candidate(rng: random.Random):
    size = 6
    interior = [(row, column) for row in range(1, 5) for column in range(1, 5)]
    walls = {
        (row, column)
        for row in range(size)
        for column in range(size)
        if row in {0, size - 1} or column in {0, size - 1}
    }
    internal_walls = set(rng.sample(interior, rng.randint(0, 2)))
    walls.update(internal_walls)
    available = [cell for cell in interior if cell not in internal_walls]
    player, box_a, box_b, goal_a, goal_b = rng.sample(available, 5)
    boxes = frozenset((box_a, box_b))
    goals = frozenset((goal_a, goal_b))
    path = _solve_sokoban(player, boxes, goals, walls, state_limit=100000)
    if path is None or len(path) < 8 or len(path) > 55:
        return None
    rows = []
    for row in range(size):
        values = []
        for column in range(size):
            position = row, column
            if position in walls:
                values.append("#")
            elif position == player:
                values.append("@")
            elif position in boxes:
                values.append("$")
            elif position in goals:
                values.append(".")
            else:
                values.append(" ")
        rows.append("".join(values))
    return rows, path


def _solve_sokoban(player, boxes, goals, walls, state_limit):
    start = player, tuple(sorted(boxes))
    queue = deque([start])
    predecessor = {start: None}
    while queue and len(predecessor) <= state_limit:
        state = queue.popleft()
        player, box_tuple = state
        box_set = set(box_tuple)
        if box_set == set(goals):
            path = []
            while predecessor[state] is not None:
                state, move = predecessor[state]
                path.append(move)
            path.reverse()
            return tuple(path)
        for move, (row_delta, column_delta) in _DIRECTIONS:
            destination = player[0] + row_delta, player[1] + column_delta
            if destination in walls:
                continue
            next_boxes = set(box_set)
            if destination in next_boxes:
                pushed = destination[0] + row_delta, destination[1] + column_delta
                if pushed in walls or pushed in next_boxes:
                    continue
                next_boxes.remove(destination)
                next_boxes.add(pushed)
            next_state = destination, tuple(sorted(next_boxes))
            if next_state in predecessor:
                continue
            predecessor[next_state] = state, move
            queue.append(next_state)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/.cache/research_v02")
    )
    args = parser.parse_args()
    manifest = ResearchV02Builder(args.output_dir, args.cache_dir).build()
    for split_name, count in SPLIT_COUNTS.items():
        print(
            "research_v02_{}.jsonl: {} cases ({} x {} tasks)".format(
                split_name, count * manifest["task_count"], count, manifest["task_count"]
            )
        )


if __name__ == "__main__":
    main()
