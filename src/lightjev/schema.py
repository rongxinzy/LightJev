"""Strict, small JSONL contract for supervised decision records."""
from __future__ import annotations

import json
import math
from pathlib import Path


def validate_record(row: dict) -> dict:
    """Return a normalized copy; reject ambiguous labels and invalid distributions.

    Score candidates are ordered from low to high. Their ordinal indices, not
    numbers appearing in their text, define the score scale.
    """
    if not isinstance(row, dict):
        raise ValueError("record must be a JSON object")
    normalized = dict(row)
    for key in ("id", "group_id", "state", "question"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string")
        normalized[key] = value.strip()
    kind = row.get("kind")
    if kind not in ("choice", "boolean", "score"):
        raise ValueError("kind must be choice, boolean, or score")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 255:
        raise ValueError("candidates must contain between 2 and 255 strings")
    if any(not isinstance(item, str) or not item.strip() for item in candidates):
        raise ValueError("each candidate must be a nonempty string")
    candidates = [item.strip() for item in candidates]
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique after stripping whitespace")
    if kind == "boolean" and candidates != ["false", "true"]:
        raise ValueError("boolean candidates must be ['false', 'true'] in that order")
    normalized["candidates"] = candidates
    if "target" not in row:
        return normalized
    target = row["target"]
    if not isinstance(target, list) or len(target) != len(candidates):
        raise ValueError("target must have one probability per candidate")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) for p in target):
        raise ValueError("target probabilities must be numbers")
    target = [float(p) for p in target]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in target):
        raise ValueError("target probabilities must be finite and in [0, 1]")
    if not math.isclose(sum(target), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("target probabilities must sum to 1")
    total = sum(target)
    normalized.update(candidates=candidates, target=[p / total for p in target])
    return normalized


def load_records(path: str | Path) -> list[dict]:
    """Read JSONL with useful line errors and unique record IDs."""
    records, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = validate_record(json.loads(line))
                if record["id"] in seen:
                    raise ValueError(f"duplicate id: {record['id']}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            seen.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records
