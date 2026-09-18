#!/usr/bin/env python3
"""Prepare pinned NanoJev stage1 programmatic gold, never teacher targets.

Before any model evaluation, remove exact normalized input duplicates from later
splits, using precedence train, dev, calibration, test, ood. Conflicting labels
or cross-split source groups/state IDs are errors. Source cache is not publishable.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lightjev.schema import validate_record

REVISION = "87061eb91e8fc687e9b046454afdcc5551e3eff7"
SHA256 = "7294765b80e751fc5aee7aba906b28a8ea80d6f147253d3f6c3e4f491de0b2d7"
URL = f"https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/resolve/{REVISION}/stage1/all.jsonl"
SPLITS = ("train", "dev", "calibration", "test", "ood")

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())

def prepare(source: Path, output: Path):
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256, "source SHA256 mismatch"
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    converted = {s: [] for s in SPLITS}
    group_splits, state_splits, ids = {}, {}, set()
    licenses, sources, families, removed_fields = Counter(), Counter(), Counter(), Counter()
    excluded_questions, target_kinds = [], Counter()
    for row in rows:
        metadata = row["metadata"]
        assert metadata["license"] == "CC0-1.0", "unapproved source license"
        assert metadata["source"] == "self_authored_programmatic", "unapproved source provenance"
        licenses[metadata["license"]] += 1
        sources[metadata["source"]] += 1
        split = row["split"]
        assert split in SPLITS
        group, state_id = metadata["source_group_id"], row["state_id"]
        assert isinstance(group, str) and group.strip()
        assert isinstance(state_id, str) and state_id.strip()
        for key, assignments in ((group, group_splits), (state_id, state_splits)):
            assert assignments.setdefault(key, split) == split, f"cross-split source group/state: {key}"
        families[row["family_id"]] += 1
        removed_fields.update(set(row) - {"id", "state_id", "family_id", "split", "state", "questions", "gold_probs", "gold_probs_kind", "metadata"})
        for qid, q in row["questions"].items():
            target_kind = row.get("gold_probs_kind", "missing")
            if isinstance(target_kind, dict):
                target_kind = target_kind[qid]
            if target_kind not in {"deterministic_truth", "programmatic_conditional_distribution"}:
                excluded_questions.append({"id": f"{row['id']}:{qid}", "target_kind": target_kind, "split": split})
                continue
            target_kinds[target_kind] += 1
            kind = q["type"]
            if kind == "boolean":
                assert q.get("criteria") is None, "boolean criteria requires explicit semantic conversion"
                keys = candidates = ["false", "true"]
            elif kind == "choice":
                assert isinstance(q["criteria"], dict)
                keys = list(q["criteria"])
                candidates = list(q["criteria"].values())
            elif kind == "score":
                assert isinstance(q["criteria"], list)
                candidates = q["criteria"]
                keys = [str(i) for i in range(len(candidates))]
            else:
                raise ValueError(f"unknown question type {kind}")
            assert all(isinstance(c, str) and c.strip() for c in candidates)
            if len(set(c.strip() for c in candidates)) != len(candidates):
                candidates = [f"{key}: {desc}" for key, desc in zip(keys, candidates)]
            gold = row["gold_probs"][qid]
            assert set(gold) == set(keys), "gold and candidate keys differ"
            result = validate_record({
                "id": f"{row['id']}:{qid}", "group_id": group,
                "state": row["state"] if isinstance(row["state"], str) else canonical(row["state"]),
                "question": q["instructions"], "kind": kind,
                "candidates": candidates, "target": [gold[k] for k in keys],
                "provenance": {"source_id": row["id"], "state_id": state_id,
                               "family_id": row["family_id"], "question_id": qid,
                               "license": metadata["license"], "source": metadata["source"],
                               "target_kind": target_kind, "split": split},
            })
            if target_kind == "deterministic_truth":
                assert all(p in (0.0, 1.0) for p in result["target"]), "deterministic target is not one-hot"
            assert result["id"] not in ids
            ids.add(result["id"])
            converted[split].append(result)
    seen, removed, counts_before = {}, [], {s: len(v) for s, v in converted.items()}
    # Normalize whitespace/Unicode, preserve candidate order and all prompt fields.
    for split in SPLITS:
        retained = []
        for row in converted[split]:
            signature = canonical([normalize(row["state"]), normalize(row["question"]),
                                   row["kind"], [normalize(c) for c in row["candidates"]]])
            if signature in seen:
                first_split, first_id, first_target = seen[signature]
                assert first_target == row["target"], f"conflicting targets: {first_id}, {row['id']}"
                if first_split != split:
                    removed.append({"id": row["id"], "split": split, "duplicate_of": first_id,
                                    "reason": "exact_normalized_model_input_in_earlier_split"})
                    continue
            else:
                seen[signature] = (split, row["id"], row["target"])
            retained.append(row)
        converted[split] = retained
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, records in converted.items():
        assert records, f"empty split {split}"
        payload = "".join(canonical(r) + "\n" for r in records).encode()
        (output / f"{split}.jsonl").write_bytes(payload)
        files[f"{split}.jsonl"] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "records": len(records),
            "kinds": dict(Counter(r["kind"] for r in records)),
            "families": dict(Counter(r["provenance"]["family_id"] for r in records))}
    manifest = {"source_url": URL, "source_revision": REVISION, "source_sha256": SHA256,
        "source_records": len(rows), "target_kind_counts": dict(target_kinds), "excluded_questions": excluded_questions, "license_counts": dict(licenses), "source_counts": dict(sources),
        "source_family_counts": dict(families), "removed_root_field_counts": dict(removed_fields),
        "transformation": "Whitelist only state, question instructions, ordered candidate descriptions and programmatic gold_probs; all teacher fields excluded.",
        "license": "CC0-1.0 per source record metadata; upstream repository has no blanket license declaration",
        "attribution": "C-Tianyu / TianyuCodings, NanoJev-Data; modified into one-question-per-record LightJev format",
        "audit": {"cross_split_source_groups": 0, "cross_split_state_ids": 0, "conflicting_normalized_inputs": 0,
                  "source_groups": len(group_splits), "source_state_ids": len(state_splits)},
        "deduplication_policy": "Fixed before evaluation: normalized exact input collisions retain earliest split by train/dev/calibration/test/ood precedence; conflicting targets fail; within-split repeats retained.",
        "counts_before_deduplication": counts_before, "removed_records": removed, "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/stage1-source/all.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/stage1-data")
    args = parser.parse_args()
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        assert hashlib.sha256(payload).hexdigest() == SHA256, "download SHA256 mismatch"
        args.source.write_bytes(payload)
    manifest = prepare(args.source, args.output)
    print(json.dumps({"source_records": manifest["source_records"], "licenses": manifest["license_counts"],
        "removed_records": len(manifest["removed_records"]), "audit": manifest["audit"],
        "splits": {k: v["records"] for k, v in manifest["files"].items()}}, indent=2))

if __name__ == "__main__":
    main()
