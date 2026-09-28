"""Semantic invariants and independent boundary cases for the new research protocol."""
import json
import random
from collections import defaultdict

import pytest

from data import generate, strict_encode
from metrics import summarize
from rules import DEV_FAMILIES, TEST_FAMILIES, TRAIN_FAMILIES, make_group, oracle, semantic_state


@pytest.mark.parametrize("family,state,expected", [
    ("threshold", {"amount": 10}, 1), ("threshold", {"amount": 9}, 0),
    ("membership", {"category": "amber"}, 1), ("membership", {"category": "fir"}, 0),
    ("conjunction", {"amount": 10, "age": 50}, 1),
    ("conjunction", {"amount": 10, "age": 51}, 0),
    ("disjunction", {"amount": 9, "age": 51}, 0),
    ("disjunction", {"amount": 9, "age": 50}, 1),
    ("veto", {"amount": 99, "flag": True}, 0),
    ("veto", {"amount": 10, "flag": False}, 1),
    ("exclusive_or", {"amount": 10, "age": 50}, 0),
    ("exclusive_or", {"amount": 9, "age": 50}, 1),
    ("bands", {"amount": 9}, 0), ("bands", {"amount": 10}, 1),
    ("bands", {"amount": 19}, 1), ("bands", {"amount": 20}, 2),
    ("weighted_bands", {"amount": 5, "age": 0}, 1),
    ("weighted_bands", {"amount": 5, "age": 10}, 2),
])
def test_oracle_boundaries(family, state, expected):
    facts = {"amount": 0, "age": 99, "flag": False, "category": "fir", **state}
    policy = {"cutoff": 10, "age_limit": 50, "allowed": ["amber"], "low": 10, "high": 20, "weight": 2}
    assert oracle(family, facts, policy) == expected


def test_group_semantics_and_split_isolation():
    splits = generate(20260928)
    state_owners, group_owners = {}, {}
    for split, rows in splits.items():
        groups = defaultdict(list)
        for row in rows:
            assert row["group_id"] not in group_owners or group_owners[row["group_id"]] == split
            group_owners[row["group_id"]] = split
            key = semantic_state(row)
            assert key not in state_owners or state_owners[key] == split
            state_owners[key] = split
            assert row["semantic_values"][row["label_index"]] == oracle(row["family"], row["state"], row["rule"])
            groups[row["group_id"]].append(row)
        for rows_in_group in groups.values():
            assert len(rows_in_group) == 4
            base, policy, fact, irrelevant = rows_in_group
            values = [r["semantic_values"][r["label_index"]] for r in rows_in_group]
            assert values[0] != values[1] and values[0] != values[2] and values[0] == values[3]
            assert base["state"] == policy["state"] and base["question"] != policy["question"]
            assert base["question"] == fact["question"] == irrelevant["question"]
        families = {r["family"] for r in rows}
        if split == "train":
            assert families == set(TRAIN_FAMILIES)
        if split == "dev_new":
            assert families == set(DEV_FAMILIES)
        if split == "locked_test":
            assert families == set(TEST_FAMILIES)


def test_native_encoding_preserves_criteria_and_order():
    import os
    from transformers import AutoTokenizer
    model_dir = os.environ.get("LAYA_BASE_DIR")
    if not model_dir:
        pytest.skip("set LAYA_BASE_DIR for pinned-tokenizer integration test")
    tok = AutoTokenizer.from_pretrained(model_dir + "/tokenizer", local_files_only=True)
    seen_types = set()
    for family in TRAIN_FAMILIES + DEV_FAMILIES + TEST_FAMILIES:
        rows = make_group(family, "audit", 0, random.Random(7))
        for row in rows:
            seen_types.add(row["question"]["type"])
            encoded = strict_encode(row, tok)
            order = list(reversed(range(len(row["target"]))))
            reversed_row = strict_encode(row, tok, order=order)
            assert reversed_row["target"] == [row["target"][i] for i in order]
            assert len(encoded["markers"]) == len(row["target"])
        broken = json.loads(json.dumps(rows[0]))
        broken["question"]["instructions"] += " long instruction" * 250
        with pytest.raises(ValueError, match="truncated"):
            strict_encode(broken, tok)


def test_pair_metric_does_not_reward_merely_changing_answers():
    rows = []
    for variant, label, prediction in [("base", 0, 1), ("policy", 1, 0), ("fact", 1, 0), ("irrelevant", 0, 1)]:
        rows.append({"id": variant, "group_id": "one", "family": "threshold", "variant": variant,
                     "label_index": label, "prediction": prediction, "semantic_values": [0, 1],
                     "probabilities": [.9, .1] if prediction == 0 else [.1, .9],
                     "logits": [2., 0.] if prediction == 0 else [0., 2.]})
    result = summarize(rows)["macro"]
    assert result["counterfactual_both_correct"] == 0.
    assert result["irrelevant_semantic_consistency"] == 1.
    assert result["irrelevant_both_correct"] == 0.
