#!/usr/bin/env python3
"""Build the literature-aligned AgentMerge research_v03 benchmark.

The search split stays small enough for expensive evolutionary evaluation,
while validation and test are larger, disjoint, deterministic holdouts. Public
remote rows are cached page-by-page; BFCL is read from a pinned local reference
repository.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time
import urllib.parse
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
SELECTION_SEED = 20260815
PAGE_SIZE = 100
SPLITS = ("search", "validation", "test")


@dataclass(frozen=True)
class RemoteSpec:
    dataset: str
    config: str
    split: str
    row_count: int


NLI_SPECS = {
    "mnli": RemoteSpec("nyu-mll/glue", "mnli", "validation_matched", 9815),
    "qnli": RemoteSpec("nyu-mll/glue", "qnli", "validation", 5463),
    "snli": RemoteSpec("stanfordnlp/snli", "plain_text", "validation", 10000),
    "rte": RemoteSpec("nyu-mll/glue", "rte", "validation", 277),
    "sick": RemoteSpec("Kaiyang92/sick_nli_knots", "default", "validation", 495),
    "scitail": RemoteSpec("allenai/scitail", "tsv_format", "validation", 1304),
}
GSM8K_SPEC = RemoteSpec("openai/gsm8k", "main", "test", 1319)
MATH_SPEC = RemoteSpec(
    "DigitalLearningGmbH/MATH-lighteval", "default", "test", 5000
)

NLI_COUNTS = {"search": 16, "validation": 27, "test": 48}
MATH_COUNTS = {"search": 48, "validation": 81, "test": 144}
HUMANEVAL_COUNTS = {"search": 48, "validation": 48, "test": 64}
MBPP_COUNTS = {"search": 48, "validation": 114, "test": 224}
BFCL_COUNTS = {"search": 96, "validation": 162, "test": 288}


class ResearchV03Builder:
    def __init__(self, root: Path, output_dir: Path, cache_dir: Path) -> None:
        self.root = root
        self.output_dir = output_dir
        self.cache_dir = cache_dir

    def build(self) -> Mapping[str, Any]:
        nli = {
            name: self._build_nli(name, spec)
            for name, spec in NLI_SPECS.items()
        }
        gsm8k = self._build_gsm8k()
        math_rows = self._build_math()
        humaneval = self._build_humaneval()
        mbpp = self._build_mbpp()
        bfcl = self._build_bfcl()

        split_rows: Dict[str, List[Mapping[str, Any]]] = {}
        split_ids: Dict[str, Dict[str, List[str]]] = {}
        cursors: Dict[str, int] = {}

        def take(name: str, rows: Sequence[Mapping[str, Any]], count: int):
            start = cursors.get(name, 0)
            selected = list(rows[start : start + count])
            if len(selected) != count:
                raise RuntimeError(
                    "{} has {} remaining rows, {} required".format(
                        name, len(selected), count
                    )
                )
            cursors[name] = start + count
            return selected

        for split_name in SPLITS:
            nli_rows = _round_robin(
                [
                    take(
                        "nli_{}".format(name),
                        rows,
                        NLI_COUNTS[split_name],
                    )
                    for name, rows in nli.items()
                ]
            )
            language = nli_rows
            mathematics = _round_robin(
                [
                    take("gsm8k", gsm8k, MATH_COUNTS[split_name]),
                    take("math", math_rows, MATH_COUNTS[split_name]),
                ]
            )
            code = _round_robin(
                [
                    take(
                        "humaneval",
                        humaneval,
                        HUMANEVAL_COUNTS[split_name],
                    ),
                    take("mbpp", mbpp, MBPP_COUNTS[split_name]),
                ]
            )
            tool = take("bfcl", bfcl, BFCL_COUNTS[split_name])
            domains = {
                "language_reasoning": language,
                "mathematics": mathematics,
                "code": code,
                "tool_use": tool,
            }
            sizes = {name: len(rows) for name, rows in domains.items()}
            if len(set(sizes.values())) != 1:
                raise RuntimeError(
                    "Domain split is not balanced for {}: {}".format(
                        split_name, sizes
                    )
                )
            rows = _round_robin(list(domains.values()))
            split_rows[split_name] = rows
            groups = sorted({str(row["group"]) for row in rows})
            split_ids[split_name] = {
                group: [
                    str(row["id"]) for row in rows if row["group"] == group
                ]
                for group in groups
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, rows in split_rows.items():
            self._write_jsonl(
                self.output_dir / "research_v03_{}.jsonl".format(split_name),
                rows,
            )

        manifest = {
            "schema_version": 1,
            "selection_seed": SELECTION_SEED,
            "benchmark_profile": "literature_aligned_static_multidomain",
            "objectives": {
                "language_reasoning": [
                    "nli_mnli",
                    "nli_qnli",
                    "nli_snli",
                    "nli_rte",
                    "nli_sick",
                    "nli_scitail",
                ],
                "mathematics": ["math_gsm8k", "math_math"],
                "code": ["code_humaneval", "code_mbpp"],
                "tool_use": ["tool_bfcl"],
            },
            "protocol": {
                "humaneval_mbpp": (
                    "official base tests in constrained local subprocess; pass@1; "
                    "MBPP exposes its required entry point and first assertion "
                    "as a public example while scoring all assertions; "
                    "HumanEval/160 is excluded because its reference uses eval; "
                    "MBPP/123 is excluded because its reference exceeds the "
                    "per-case CPU budget"
                ),
                "bfcl": (
                    "BFCL V4 simple_python and live_simple, prompt-mode AST, "
                    "no live API execution"
                ),
            },
            "sources": {
                "6_nli": [spec.dataset for spec in NLI_SPECS.values()],
                "gsm8k": GSM8K_SPEC.dataset,
                "math": MATH_SPEC.dataset,
                "humaneval": "openai/human-eval",
                "mbpp": "google-research/mbpp sanitized",
                "bfcl": (
                    "references/codes/gorilla/"
                    "berkeley-function-call-leaderboard"
                ),
            },
            "splits": {
                split_name: {
                    "total_count": len(rows),
                    "domain_counts": _domain_counts(rows),
                    "group_counts": _group_counts(rows),
                    "ids": split_ids[split_name],
                }
                for split_name, rows in split_rows.items()
            },
        }
        (self.output_dir / "research_v03_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _build_nli(self, name: str, spec: RemoteSpec):
        required = sum(NLI_COUNTS.values())
        raw = self._fetch_window(spec, required * 5)
        rows = []
        for index, row in raw:
            converted = self._convert_nli(name, index, row)
            if converted is not None:
                rows.append(converted)
        return _balanced_by(rows, required, "target", name)

    @staticmethod
    def _convert_nli(name: str, index: int, row: Mapping[str, Any]):
        if name == "mnli":
            premise, hypothesis = row["premise"], row["hypothesis"]
            labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
        elif name == "qnli":
            premise, hypothesis = row["sentence"], row["question"]
            labels = {0: "entailment", 1: "not_entailment"}
        elif name == "snli":
            premise, hypothesis = row["premise"], row["hypothesis"]
            labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
        elif name == "rte":
            premise, hypothesis = row["sentence1"], row["sentence2"]
            labels = {0: "entailment", 1: "not_entailment"}
        elif name == "sick":
            premise, hypothesis = row["sentence_A"], row["sentence_B"]
            labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
        elif name == "scitail":
            premise, hypothesis = row["premise"], row["hypothesis"]
            labels = {"entails": "entailment", "neutral": "not_entailment"}
        else:
            raise ValueError("Unknown NLI source {!r}".format(name))
        label = labels.get(row.get("label"))
        if label is None:
            return None
        choices = sorted(set(labels.values()))
        return {
            "id": "nli_{}:{}".format(name, row.get("idx", row.get("id", index))),
            "input": (
                "Classify the relationship between the premise and hypothesis. "
                "Return only one label from: {}.\n\nPremise: {}\n\nHypothesis: {}"
            ).format(", ".join(choices), premise, hypothesis),
            "target": label,
            "group": "nli_{}".format(name),
            "domain": "language_reasoning",
            "scorer": "choice",
            "source": NLI_SPECS[name].dataset,
            "source_id": str(row.get("idx", row.get("id", index))),
        }

    def _build_gsm8k(self):
        required = sum(MATH_COUNTS.values())
        raw = self._fetch_window(GSM8K_SPEC, required * 3)
        rows = []
        for index, row in raw:
            answer = str(row["answer"]).rsplit("####", 1)[-1].strip()
            rows.append(
                {
                    "id": "gsm8k:{:04d}".format(index),
                    "input": (
                        "Solve the grade-school mathematics problem. Return only "
                        "the final numeric answer.\n\n{}"
                    ).format(row["question"]),
                    "target": answer,
                    "group": "math_gsm8k",
                    "domain": "mathematics",
                    "scorer": "numeric",
                    "source": GSM8K_SPEC.dataset,
                    "source_id": str(index),
                }
            )
        return _stable_select(rows, required, "gsm8k")

    def _build_math(self):
        required = sum(MATH_COUNTS.values())
        raw = self._fetch_window(MATH_SPEC, required * 3)
        rows = [
            {
                "id": "math:{:04d}".format(index),
                "input": (
                    "Solve the MATH problem. Return only the final answer.\n\n{}"
                ).format(row["problem"]),
                "target": str(row["solution"]),
                "group": "math_math",
                "domain": "mathematics",
                "scorer": "math",
                "source": MATH_SPEC.dataset,
                "source_id": str(index),
                "level": row.get("level"),
                "subject": row.get("type"),
            }
            for index, row in raw
        ]
        return _balanced_by(rows, required, "level", "math")

    def _build_humaneval(self):
        url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
        path = self.cache_dir / "HumanEval.jsonl.gz"
        self._download(url, path)
        with gzip.open(str(path), "rt", encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle if line.strip()]
        rows = []
        for row in source_rows:
            if row["task_id"] == "HumanEval/160":
                continue
            target = {
                "kind": "humaneval",
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
            rows.append(
                {
                    "id": "humaneval:{}".format(row["task_id"]),
                    "input": (
                        "Complete the Python function below. Return only complete "
                        "Python code defining the requested function.\n\n```python\n{}\n```"
                    ).format(row["prompt"]),
                    "target": json.dumps(target, separators=(",", ":")),
                    "group": "code_humaneval",
                    "domain": "code",
                    "scorer": "python_code",
                    "source": "openai/human-eval",
                    "source_id": row["task_id"],
                }
            )
        return _stable_select(rows, sum(HUMANEVAL_COUNTS.values()), "humaneval")

    def _build_mbpp(self):
        url = (
            "https://raw.githubusercontent.com/google-research/"
            "google-research/master/mbpp/sanitized-mbpp.json"
        )
        path = self.cache_dir / "sanitized-mbpp.json"
        self._download(url, path)
        source_rows = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for row in source_rows:
            if int(row["task_id"]) == 123:
                continue
            entry_point = _mbpp_entry_point(row["test_list"])
            if not entry_point:
                continue
            target = {
                "kind": "mbpp",
                "entry_point": entry_point,
                "tests": row["test_list"],
                "public_tests": row["test_list"][:1],
                "test_setup": "\n".join(row.get("test_imports", ())),
            }
            rows.append(
                {
                    "id": "mbpp:{}".format(row["task_id"]),
                    "input": (
                        "Write a Python function that solves the task. Return only "
                        "complete Python code.\n\nProblem:\n{}\n\nRequired function: "
                        "{}\n\nPublic test example:\n{}"
                    ).format(row["prompt"], entry_point, row["test_list"][0]),
                    "target": json.dumps(target, separators=(",", ":")),
                    "group": "code_mbpp",
                    "domain": "code",
                    "scorer": "python_code",
                    "source": "google-research/mbpp sanitized",
                    "source_id": str(row["task_id"]),
                }
            )
        return _stable_select(rows, sum(MBPP_COUNTS.values()), "mbpp")

    def _build_bfcl(self):
        base = (
            self.root
            / "references"
            / "codes"
            / "gorilla"
            / "berkeley-function-call-leaderboard"
            / "bfcl_eval"
            / "data"
        )
        rows = []
        for category in ("simple_python", "live_simple"):
            prompts = _load_jsonl(base / "BFCL_v4_{}.json".format(category))
            answers = {
                item["id"]: item["ground_truth"]
                for item in _load_jsonl(
                    base / "possible_answer" / "BFCL_v4_{}.json".format(category)
                )
            }
            for item in prompts:
                if item["id"] not in answers:
                    continue
                functions = item["function"]
                question = item["question"][0][-1]["content"]
                target = {
                    "functions": functions,
                    "ground_truth": answers[item["id"]],
                }
                rows.append(
                    {
                        "id": "bfcl:{}".format(item["id"]),
                        "input": (
                            "Choose and call the appropriate function. Return "
                            "exactly one Python function-call expression using "
                            "keyword arguments, with no explanation.\n\nAvailable "
                            "functions:\n{}\n\nUser request:\n{}"
                        ).format(json.dumps(functions, ensure_ascii=False), question),
                        "target": json.dumps(target, separators=(",", ":")),
                        "group": "tool_bfcl",
                        "domain": "tool_use",
                        "scorer": "bfcl_ast",
                        "source": "BFCL V4",
                        "source_id": item["id"],
                        "bfcl_category": category,
                    }
                )
        return _balanced_by(
            rows, sum(BFCL_COUNTS.values()), "bfcl_category", "bfcl"
        )

    def _fetch_window(self, spec: RemoteSpec, desired: int):
        desired = min(spec.row_count, max(desired, PAGE_SIZE * 5))
        page_count = min(
            int(math.ceil(spec.row_count / PAGE_SIZE)),
            int(math.ceil(desired / PAGE_SIZE)),
        )
        page_indices = list(range(int(math.ceil(spec.row_count / PAGE_SIZE))))
        random.Random(
            _stable_integer(
                "{}:{}:{}".format(spec.dataset, spec.config, spec.split)
            )
        ).shuffle(page_indices)
        rows = []
        for page_index in page_indices[:page_count]:
            offset = page_index * PAGE_SIZE
            length = min(PAGE_SIZE, spec.row_count - offset)
            payload = self._download_dataset_page(spec, offset, length)
            rows.extend(
                (int(item["row_idx"]), item["row"]) for item in payload["rows"]
            )
        return rows

    def _download_dataset_page(
        self, spec: RemoteSpec, offset: int, length: int
    ) -> Mapping[str, Any]:
        stem = "{}--{}--{}--{:06d}.json".format(
            spec.dataset.replace("/", "--"), spec.config, spec.split, offset
        )
        path = self.cache_dir / stem
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
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
            headers={"User-Agent": "AgentMerge-benchmark-builder/0.4"},
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                path.write_text(json.dumps(payload), encoding="utf-8")
                # The public datasets-server is intentionally rate limited.
                # A small delay keeps deterministic rebuilds polite and avoids
                # bursting when a cold cache needs several pages.
                time.sleep(0.5)
                return payload
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 5:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 10.0 * (attempt + 1)
                time.sleep(max(1.0, delay))
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Unreachable retry state")

    def _download(self, url: str, path: Path) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "AgentMerge-benchmark-builder/0.4"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            path.write_bytes(response.read())

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows
            ),
            encoding="utf-8",
        )


def _stable_select(
    rows: Sequence[Mapping[str, Any]], count: int, namespace: str
) -> List[Mapping[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            "{}:{}:{}".format(SELECTION_SEED, namespace, row["id"]).encode()
        ).hexdigest(),
    )
    if len(ordered) < count:
        raise RuntimeError(
            "{} provides {} rows, {} required".format(namespace, len(ordered), count)
        )
    return ordered[:count]


def _balanced_by(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    field: str,
    namespace: str,
) -> List[Mapping[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for row in _stable_select(rows, len(rows), namespace):
        buckets.setdefault(str(row.get(field)), []).append(row)
    ordered = _round_robin([buckets[name] for name in sorted(buckets)])
    if len(ordered) < count:
        raise RuntimeError(
            "{} provides {} balanced rows, {} required".format(
                namespace, len(ordered), count
            )
        )
    return ordered[:count]


def _round_robin(groups: Sequence[Sequence[Mapping[str, Any]]]):
    groups = [list(group) for group in groups]
    result = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                result.append(group[index])
    return result


def _mbpp_entry_point(tests: Sequence[str]) -> str:
    for test in tests:
        match = __import__("re").search(r"assert\s+(?:set\()?([A-Za-z_]\w*)\(", test)
        if match:
            return match.group(1)
    return ""


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _group_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    result: Dict[str, int] = {}
    for row in rows:
        name = str(row["group"])
        result[name] = result.get(name, 0) + 1
    return dict(sorted(result.items()))


def _domain_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    result: Dict[str, int] = {}
    for row in rows:
        name = str(row["domain"])
        result[name] = result.get(name, 0) + 1
    return dict(sorted(result.items()))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / ".cache" / "research_v03",
    )
    args = parser.parse_args()
    manifest = ResearchV03Builder(root, args.output_dir, args.cache_dir).build()
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
