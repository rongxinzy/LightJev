import json
import pytest
from lightjev.schema import load_records, validate_record


def sample():
    return dict(id="x", group_id="g", state="example", question="which?", kind="choice", candidates=["a", "b"], target=[.2, .8])


def test_valid_and_inference():
    row = sample()
    assert validate_record(row) == row
    del row["target"]
    assert "target" not in validate_record(row)


@pytest.mark.parametrize("patch", [
    {"target": [float("nan"), 1]}, {"target": [.2, .9]},
    {"target": [True, False]}, {"target": [0, 0]},
    {"target": [-.1, 1.1]}, {"target": [1]},
    {"candidates": ["a", " a "]}, {"candidates": ["a"]},
    {"id": " "}, {"group_id": 1}, {"state": ""}, {"kind": "unknown"},
    {"kind": "boolean"}, {"candidates": ["a", ""]},
])
def test_reject_invalid(patch):
    with pytest.raises(ValueError):
        validate_record(sample() | patch)


def test_boolean_and_score():
    assert validate_record(sample() | {"kind": "boolean", "candidates": ["false", "true"]})
    assert validate_record(sample() | {"kind": "score"})


def test_load_duplicate_and_line_number(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(sample()) + "\n" + json.dumps(sample()))
    with pytest.raises(ValueError, match=r":2: duplicate id"):
        load_records(path)
    path.write_text("\n")
    with pytest.raises(ValueError, match="no records"):
        load_records(path)
