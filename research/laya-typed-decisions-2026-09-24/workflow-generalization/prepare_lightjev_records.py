#!/usr/bin/env python3
"""Convert fold case JSONL into LightJev candidate-scoring records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def make_records(case):
    out = []
    for qid, question in case["questions"].items():
        kind = question["type"]
        criteria = question.get("criteria", {})
        gold = case["gold"][qid]
        if kind == "choice":
            labels = list(criteria)
            candidates = [f"{label}: {criteria[label]}" for label in labels]
            model_kind = "choice"
        elif kind == "noul":
            labels, candidates, model_kind = ["false", "true"], ["false", "true"], "boolean"
        elif kind == "score":
            labels = [str(i) for i in range(len(criteria))]
            descriptions = ([criteria.get(str(i), criteria.get(i)) for i in range(len(criteria))]
                            if isinstance(criteria, dict) else list(criteria))
            candidates = [f"{i}: {descriptions[i]}" for i in range(len(criteria))]
            model_kind = "score"
        else:
            raise ValueError(f"unknown question type {kind}")
        target = [float(gold["probabilities"][label]) for label in labels]
        total = sum(target)
        if abs(total - 1.0) > 0.02 or any(p < 0 for p in target):
            raise ValueError(f"invalid target {case['id']}:{qid}: {target}")
        target = [p / total for p in target]
        label = str(gold["label"]).lower()
        if label not in labels:
            raise ValueError(f"gold label not in candidate set: {case['id']}:{qid}:{label}")
        record = {
            "id": f"{case['id']}:{qid}",
            "group_id": case["id"],
            "state": json.dumps(case["state"], ensure_ascii=False, separators=(",", ":")),
            "question": question["instructions"],
            "kind": model_kind,
            "candidates": candidates,
            "target": target,
            "metadata": {"case_id": case["id"], "workflow": case["workflow"],
                         "qid": qid, "original_kind": kind, "labels": labels,
                         "label_index": labels.index(label),
                         "gold_expected_score": float(gold["score"]) if kind == "score" else None},
        }
        out.append(record)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "calibration", "test"):
        source = args.cases_dir / f"cases_{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        records = [r for case in rows for r in make_records(case)]
        out = args.output_dir / f"records_{split}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records))
        print(json.dumps({"split": split, "cases": len(rows), "decisions": len(records), "path": str(out)}), flush=True)

if __name__ == "__main__":
    main()
