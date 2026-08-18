import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED = {"search": 64, "validation": 128, "test": 256}
EXPECTED_V02 = {"search": 224, "validation": 320, "test": 576}
EXPECTED_V03 = {"search": 384, "validation": 648, "test": 1152}
V02_GROUPS = {
    "math_olympiad",
    "qa_hotpot",
    "qa_musique",
    "code_cruxeval",
    "logic_deduction",
    "planning_path",
    "game_sudoku",
    "game_sokoban",
}
V03_GROUPS = {
    "nli_mnli",
    "nli_qnli",
    "nli_snli",
    "nli_rte",
    "nli_sick",
    "nli_scitail",
    "math_gsm8k",
    "math_math",
    "code_humaneval",
    "code_mbpp",
    "tool_bfcl",
}


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_research_splits_have_fixed_balanced_sizes_and_required_fields():
    for split_name, expected_count in EXPECTED.items():
        rows = _load_jsonl(
            DATA / "research_v01_{}.jsonl".format(split_name)
        )
        assert len(rows) == expected_count
        assert len({row["id"] for row in rows}) == expected_count
        groups = {"math500": 0, "hotpotqa": 0}
        for row in rows:
            groups[row["group"]] += 1
            assert row["input"]
            assert row["target"]
            assert row["scorer"] in ("math", "hotpot_f1")
            assert row["source_id"]
        assert groups == {
            "math500": expected_count // 2,
            "hotpotqa": expected_count // 2,
        }


def test_research_splits_and_admission_are_pairwise_disjoint():
    split_ids = {
        split_name: {
            row["id"]
            for row in _load_jsonl(
                DATA / "research_v01_{}.jsonl".format(split_name)
            )
        }
        for split_name in EXPECTED
    }
    assert split_ids["search"].isdisjoint(split_ids["validation"])
    assert split_ids["search"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    admission = {
        row["id"] for row in _load_jsonl(DATA / "admission_v01.jsonl")
    }
    assert admission.isdisjoint(set().union(*split_ids.values()))


def test_research_manifest_exactly_matches_materialized_source_ids():
    manifest = json.loads(
        (DATA / "research_v01_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selection_seed"] == 20260812
    assert manifest["source_window"] == 500
    for split_name, expected_count in EXPECTED.items():
        rows = _load_jsonl(
            DATA / "research_v01_{}.jsonl".format(split_name)
        )
        actual = {
            "math500": {
                row["source_id"]
                for row in rows
                if row["group"] == "math500"
            },
            "hotpotqa": {
                row["source_id"]
                for row in rows
                if row["group"] == "hotpotqa"
            },
        }
        assert manifest["splits"][split_name]["total_count"] == expected_count
        for source_name in actual:
            assert actual[source_name] == set(
                manifest["splits"][split_name]["ids"][source_name]
            )


def test_research_v02_is_balanced_multidomain_and_pairwise_disjoint():
    split_ids = {}
    for split_name, expected_count in EXPECTED_V02.items():
        rows = _load_jsonl(
            DATA / "research_v02_{}.jsonl".format(split_name)
        )
        assert len(rows) == expected_count
        assert len({row["id"] for row in rows}) == expected_count
        groups = {}
        for row in rows:
            groups[row["group"]] = groups.get(row["group"], 0) + 1
            assert row["input"]
            assert row["target"]
            assert row["source_id"]
        assert set(groups) == V02_GROUPS
        assert set(groups.values()) == {expected_count // len(V02_GROUPS)}
        prefix_groups = {}
        for row in rows[:64]:
            prefix_groups[row["group"]] = prefix_groups.get(row["group"], 0) + 1
        assert set(prefix_groups) == V02_GROUPS
        assert set(prefix_groups.values()) == {8}
        split_ids[split_name] = {row["id"] for row in rows}
    assert split_ids["search"].isdisjoint(split_ids["validation"])
    assert split_ids["search"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])


def test_research_v02_materializes_declared_hardness_filters():
    rows = [
        row
        for split_name in EXPECTED_V02
        for row in _load_jsonl(
            DATA / "research_v02_{}.jsonl".format(split_name)
        )
    ]
    assert all(
        row["difficulty"] == "Competition"
        for row in rows
        if row["group"] == "math_olympiad"
    )
    assert all(
        row["hop_count"] in (3, 4)
        for row in rows
        if row["group"] == "qa_musique"
    )
    assert all(
        row["object_count"] == 7
        for row in rows
        if row["group"] == "logic_deduction"
    )
    assert all(
        row["shortest_path_length"] >= 18 and row["turn_count"] >= 5
        for row in rows
        if row["group"] == "planning_path"
    )
    assert all(
        row["clue_count"] == 6
        for row in rows
        if row["group"] == "game_sudoku"
    )
    assert all(
        row["box_count"] == 2 and row["shortest_solution_length"] >= 8
        for row in rows
        if row["group"] == "game_sokoban"
    )


def test_research_v02_manifest_matches_all_materialized_ids():
    manifest = json.loads(
        (DATA / "research_v02_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selection_seed"] == 20260814
    assert manifest["task_count"] == len(V02_GROUPS)
    for split_name, expected_count in EXPECTED_V02.items():
        rows = _load_jsonl(
            DATA / "research_v02_{}.jsonl".format(split_name)
        )
        assert manifest["splits"][split_name]["total_count"] == expected_count
        for group in V02_GROUPS:
            assert set(manifest["splits"][split_name]["ids"][group]) == {
                row["id"] for row in rows if row["group"] == group
            }


def test_research_v03_is_domain_balanced_and_pairwise_disjoint():
    split_ids = {}
    for split_name, expected_count in EXPECTED_V03.items():
        rows = _load_jsonl(
            DATA / "research_v03_{}.jsonl".format(split_name)
        )
        assert len(rows) == expected_count
        assert len({row["id"] for row in rows}) == expected_count
        assert {row["group"] for row in rows} == V03_GROUPS
        domains = {}
        for row in rows:
            domains[row["domain"]] = domains.get(row["domain"], 0) + 1
            assert row["input"]
            assert row["target"]
            assert row["source_id"]
        assert set(domains) == {
            "language_reasoning", "mathematics", "code", "tool_use"
        }
        assert set(domains.values()) == {expected_count // 4}
        assert [row["domain"] for row in rows[:4]] == [
            "language_reasoning", "mathematics", "code", "tool_use"
        ]
        split_ids[split_name] = {row["id"] for row in rows}
    assert split_ids["search"].isdisjoint(split_ids["validation"])
    assert split_ids["search"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])


def test_research_v03_manifest_and_protocol_match_materialized_data():
    manifest = json.loads(
        (DATA / "research_v03_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selection_seed"] == 20260815
    assert set(manifest["objectives"]) == {
        "language_reasoning", "mathematics", "code", "tool_use"
    }
    for split_name, expected_count in EXPECTED_V03.items():
        rows = _load_jsonl(
            DATA / "research_v03_{}.jsonl".format(split_name)
        )
        assert manifest["splits"][split_name]["total_count"] == expected_count
        for group in V03_GROUPS:
            assert set(manifest["splits"][split_name]["ids"][group]) == {
                row["id"] for row in rows if row["group"] == group
            }
    all_rows = [
        row
        for split_name in EXPECTED_V03
        for row in _load_jsonl(
            DATA / "research_v03_{}.jsonl".format(split_name)
        )
    ]
    assert {row["group"] for row in all_rows if row["group"].startswith("nli_")} == {
        "nli_mnli", "nli_qnli", "nli_snli", "nli_rte", "nli_sick", "nli_scitail"
    }
    assert {row["bfcl_category"] for row in all_rows if row["group"] == "tool_bfcl"} == {
        "simple_python", "live_simple"
    }


def test_research_v03_mbpp_exposes_interface_but_keeps_hidden_tests():
    rows = [
        row
        for split_name in EXPECTED_V03
        for row in _load_jsonl(
            DATA / "research_v03_{}.jsonl".format(split_name)
        )
        if row["group"] == "code_mbpp"
    ]
    assert rows
    for row in rows:
        target = json.loads(row["target"])
        entry_point = target["entry_point"]
        public_tests = target["public_tests"]
        assert len(public_tests) == 1
        assert len(target["tests"]) >= 2
        assert public_tests[0] == target["tests"][0]
        assert "Required function: {}".format(entry_point) in row["input"]
        assert public_tests[0] in row["input"]
        assert all(test not in row["input"] for test in target["tests"][1:])
