#!/usr/bin/env python3
"""Build the small, contamination-aware adapter admission benchmark.

The selected IDs are pinned in this file. Network responses are used only to
materialize those immutable examples and are never sampled at build time.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


MATH_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=HuggingFaceH4%2FMATH-500&config=default"
    "&split=test&offset=0&length=100"
)
HOTPOT_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=hotpotqa%2Fhotpot_qa&config=distractor"
    "&split=validation&offset=0&length=100"
)

MATH_IDS = (
    "test/algebra/2584.json",
    "test/counting_and_probability/525.json",
    "test/geometry/967.json",
    "test/intermediate_algebra/1197.json",
    "test/number_theory/572.json",
    "test/prealgebra/1622.json",
    "test/precalculus/285.json",
    "test/algebra/1837.json",
)
HOTPOT_IDS = (
    "5a8c7595554299585d9e36b6",
    "5a85ea095542994775f606a8",
    "5a8e3ea95542995a26add48d",
    "5abd94525542992ac4f382d2",
    "5a8b57f25542995d1e6f1371",
    "5adbf0a255429947ff17385a",
    "5a8db19d5542994ba4e3dd00",
    "5ab56e32554299637185c594",
)


def fetch_rows(
    url: str, cache_path: Optional[Path]
) -> List[Mapping[str, Any]]:
    if cache_path is not None:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "AgentMerge-benchmark-builder/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    return [entry["row"] for entry in payload["rows"]]


def select_by_id(
    rows: Iterable[Mapping[str, Any]],
    ids: Iterable[str],
    id_field: str,
) -> List[Mapping[str, Any]]:
    indexed = {str(row[id_field]): row for row in rows}
    missing = [item_id for item_id in ids if item_id not in indexed]
    if missing:
        raise RuntimeError("Pinned dataset IDs not found: {}".format(missing))
    return [indexed[item_id] for item_id in ids]


def math_cases(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    for row in select_by_id(rows, MATH_IDS, "unique_id"):
        cases.append(
            {
                "id": "math500:{}".format(row["unique_id"]),
                "input": (
                    "Solve the following mathematics problem. Return only the "
                    "final answer, without derivation.\n\n{}".format(row["problem"])
                ),
                "target": row["answer"],
                "group": "math500",
                "scorer": "math",
                "source": "HuggingFaceH4/MATH-500",
                "source_id": row["unique_id"],
                "split": "test",
                "subject": row["subject"],
                "level": row["level"],
            }
        )
    return cases


def hotpot_cases(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    for row in select_by_id(rows, HOTPOT_IDS, "id"):
        passages = []
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"]
        ):
            passages.append(
                "[{}]\n{}".format(title, " ".join(sentences).strip())
            )
        cases.append(
            {
                "id": "hotpotqa:{}".format(row["id"]),
                "input": (
                    "Answer the question using the provided passages. Return "
                    "only a concise answer.\n\nPassages:\n{}\n\nQuestion: {}"
                ).format("\n\n".join(passages), row["question"]),
                "target": row["answer"],
                "group": "hotpotqa",
                "scorer": "hotpot_f1",
                "source": "hotpotqa/hotpot_qa",
                "source_id": row["id"],
                "split": "validation",
                "question_type": row["type"],
                "level": row["level"],
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/admission_v01.jsonl")
    )
    parser.add_argument("--math-cache", type=Path)
    parser.add_argument("--hotpot-cache", type=Path)
    args = parser.parse_args()

    cases = math_cases(fetch_rows(MATH_URL, args.math_cache))
    cases.extend(hotpot_cases(fetch_rows(HOTPOT_URL, args.hotpot_cache)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("Wrote {} pinned cases to {}".format(len(cases), args.output))


if __name__ == "__main__":
    main()
