#!/usr/bin/env python3
"""Build fixed, disjoint search/validation/test sets for DE pool ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
SOURCE_WINDOW = 500
PAGE_SIZE = 100
SELECTION_SEED = 20260812
SPLIT_COUNTS = {
    "search": 32,
    "validation": 64,
    "test": 128,
}

SOURCES = {
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "config": "default",
        "split": "test",
        "id_field": "unique_id",
    },
    "hotpotqa": {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "id_field": "id",
    },
}


def fetch_source(
    source: Mapping[str, str], cache_dir: Path
) -> List[Mapping[str, Any]]:
    """Fetch a fixed source window in cacheable datasets-server pages."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Mapping[str, Any]] = []
    for offset in range(0, SOURCE_WINDOW, PAGE_SIZE):
        cache_name = "{}-{}-{:04d}.json".format(
            source["dataset"].replace("/", "--"), source["split"], offset
        )
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            query = urllib.parse.urlencode(
                {
                    "dataset": source["dataset"],
                    "config": source["config"],
                    "split": source["split"],
                    "offset": offset,
                    "length": PAGE_SIZE,
                }
            )
            request = urllib.request.Request(
                "{}?{}".format(DATASETS_SERVER, query),
                headers={"User-Agent": "AgentMerge-benchmark-builder/0.2"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        rows.extend(entry["row"] for entry in payload["rows"])
    if len(rows) < SOURCE_WINDOW:
        raise RuntimeError(
            "Source {} returned only {} rows".format(
                source["dataset"], len(rows)
            )
        )
    return rows[:SOURCE_WINDOW]


def admission_ids(path: Path) -> Mapping[str, set]:
    excluded = {name: set() for name in SOURCES}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            source = str(item.get("source", ""))
            if source == SOURCES["math500"]["dataset"]:
                excluded["math500"].add(str(item["source_id"]))
            elif source == SOURCES["hotpotqa"]["dataset"]:
                excluded["hotpotqa"].add(str(item["source_id"]))
    return excluded


def select_ids(
    source_name: str,
    rows: Sequence[Mapping[str, Any]],
    id_field: str,
    excluded: Iterable[str],
) -> Mapping[str, Tuple[str, ...]]:
    """Stable hash ordering prevents dependence on RNG/library versions."""

    excluded_set = set(excluded)
    identifiers = sorted(
        {
            str(row[id_field])
            for row in rows
            if str(row[id_field]) not in excluded_set
        },
        key=lambda identifier: (
            hashlib.sha256(
                "{}:{}:{}".format(
                    SELECTION_SEED, source_name, identifier
                ).encode("utf-8")
            ).hexdigest(),
            identifier,
        ),
    )
    required = sum(SPLIT_COUNTS.values())
    if len(identifiers) < required:
        raise RuntimeError(
            "{} has {} eligible IDs, {} required".format(
                source_name, len(identifiers), required
            )
        )
    cursor = 0
    selected = {}
    for split_name, count in SPLIT_COUNTS.items():
        selected[split_name] = tuple(identifiers[cursor : cursor + count])
        cursor += count
    return selected


def math_case(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "id": "math500:{}".format(row["unique_id"]),
        "input": (
            "Solve the following mathematics problem. Return only the final "
            "answer, without derivation.\n\n{}"
        ).format(row["problem"]),
        "target": row["answer"],
        "group": "math500",
        "scorer": "math",
        "source": SOURCES["math500"]["dataset"],
        "source_id": row["unique_id"],
        "split": SOURCES["math500"]["split"],
        "subject": row["subject"],
        "level": row["level"],
    }


def hotpot_case(row: Mapping[str, Any]) -> Mapping[str, Any]:
    passages = []
    for title, sentences in zip(
        row["context"]["title"], row["context"]["sentences"]
    ):
        passages.append("[{}]\n{}".format(title, " ".join(sentences).strip()))
    return {
        "id": "hotpotqa:{}".format(row["id"]),
        "input": (
            "Answer the question using the provided passages. Return only a "
            "concise answer.\n\nPassages:\n{}\n\nQuestion: {}"
        ).format("\n\n".join(passages), row["question"]),
        "target": row["answer"],
        "group": "hotpotqa",
        "scorer": "hotpot_f1",
        "source": SOURCES["hotpotqa"]["dataset"],
        "source_id": row["id"],
        "split": SOURCES["hotpotqa"]["split"],
        "question_type": row["type"],
        "level": row["level"],
    }


def write_jsonl(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def build(output_dir: Path, cache_dir: Path, admission_path: Path) -> None:
    excluded = admission_ids(admission_path)
    source_rows = {
        name: fetch_source(source, cache_dir)
        for name, source in SOURCES.items()
    }
    selected = {
        name: select_ids(
            name,
            source_rows[name],
            source["id_field"],
            excluded[name],
        )
        for name, source in SOURCES.items()
    }
    indexed = {
        name: {
            str(row[source["id_field"]]): row
            for row in source_rows[name]
        }
        for name, source in SOURCES.items()
    }
    converters = {"math500": math_case, "hotpotqa": hotpot_case}
    for split_name in SPLIT_COUNTS:
        cases = []
        for source_name in ("math500", "hotpotqa"):
            cases.extend(
                converters[source_name](indexed[source_name][identifier])
                for identifier in selected[source_name][split_name]
            )
        write_jsonl(
            output_dir / "research_v01_{}.jsonl".format(split_name), cases
        )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "selection_seed": SELECTION_SEED,
        "source_window": SOURCE_WINDOW,
        "admission_exclusion": str(admission_path),
        "sources": SOURCES,
        "splits": {
            split_name: {
                "total_count": 2 * count,
                "per_source_count": count,
                "ids": {
                    source_name: list(selected[source_name][split_name])
                    for source_name in SOURCES
                },
            }
            for split_name, count in SPLIT_COUNTS.items()
        },
    }
    (output_dir / "research_v01_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/.cache/research_v01")
    )
    parser.add_argument(
        "--admission", type=Path, default=Path("data/admission_v01.jsonl")
    )
    args = parser.parse_args()
    build(args.output_dir, args.cache_dir, args.admission)
    for split_name, per_source in SPLIT_COUNTS.items():
        print(
            "research_v01_{}.jsonl: {} cases".format(
                split_name, 2 * per_source
            )
        )


if __name__ == "__main__":
    main()
