#!/usr/bin/env python3
"""Build four case-disjoint leave-one-workflow-out folds from pinned train parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from prepare_data import encode_case, parse_cases, write_jsonl
from transformers import AutoTokenizer

TRAIN_PER_SOURCE_WORKFLOW = 272
DEV_PER_SOURCE_WORKFLOW = 14
CAL_PER_SOURCE_WORKFLOW = 14


def write_fold(cases, heldout, tok, args, out_root):
    workflows = sorted({row["workflow"] for row in cases})
    if len(workflows) != 4:
        raise ValueError(f"expected exactly 4 workflows, got {workflows}")
    if heldout not in workflows:
        raise ValueError(f"unknown held-out workflow {heldout}")
    partitions = {name: [] for name in ("train", "dev", "calibration", "test")}
    for workflow in workflows:
        group = sorted((row for row in cases if row["workflow"] == workflow), key=lambda r: r["id"])
        if len(group) != 300:
            raise ValueError(f"expected 300 train cases for {workflow}, got {len(group)}")
        if workflow == heldout:
            partitions["test"] = group
            continue
        seed_offset = sum(workflow.encode("utf-8"))
        random.Random(args.seed + seed_offset).shuffle(group)
        n_train, n_dev = TRAIN_PER_SOURCE_WORKFLOW, DEV_PER_SOURCE_WORKFLOW
        n_cal = CAL_PER_SOURCE_WORKFLOW
        partitions["train"].extend(group[:n_train])
        partitions["dev"].extend(group[n_train:n_train + n_dev])
        partitions["calibration"].extend(group[n_train + n_dev:n_train + n_dev + n_cal])
    id_sets = {name: {r["id"] for r in rows} for name, rows in partitions.items()}
    state_sets = {name: {json.dumps(r["state"], sort_keys=True, ensure_ascii=False) for r in rows}
                  for name, rows in partitions.items()}
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if id_sets[left] & id_sets[right] or state_sets[left] & state_sets[right]:
                raise ValueError(f"case/state overlap between {left} and {right}")
    if sum(map(len, partitions.values())) != len(cases):
        raise ValueError("fold does not account for every source case exactly once")

    fold_dir = out_root / heldout
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "leave-one-workflow-out on the pinned official train split",
        "seed": args.seed,
        "heldout_workflow": heldout,
        "training_workflows": [w for w in workflows if w != heldout],
        "case_counts": {name: len(rows) for name, rows in partitions.items()},
        "workflow_case_counts": {name: dict(Counter(r["workflow"] for r in rows))
                                  for name, rows in partitions.items()},
        "question_counts": {},
        "question_type_counts": {},
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "case_and_state_disjoint": True,
        "test_source": "all cases from the held-out workflow in the public train split; public test split is not used",
    }
    for name, rows in partitions.items():
        write_jsonl(fold_dir / f"cases_{name}.jsonl", rows)
        items = [item for case in rows for item in encode_case(case, tok, args.max_len, args.head_max_len)]
        write_jsonl(fold_dir / f"items_{name}.jsonl", items)
        manifest["question_counts"][name] = len(items)
        manifest["question_type_counts"][name] = dict(Counter(item["kind"] for item in items))
    if manifest["question_counts"]["train"] % 8:
        raise ValueError("train question count must divide evenly across eight DDP ranks")
    (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"fold": heldout, **manifest}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    args = ap.parse_args()
    cases = parse_cases(args.train_parquet)
    tok = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer")
    workflows = sorted({row["workflow"] for row in cases})
    if len(cases) != 1200 or len(workflows) != 4:
        raise ValueError(f"expected 1200 cases across 4 workflows, got {len(cases)}, {workflows}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for heldout in workflows:
        write_fold(cases, heldout, tok, args, args.output_dir)
    (args.output_dir / "source_train_parquet.sha256").write_text(
        hashlib.sha256(args.train_parquet.read_bytes()).hexdigest() + "  train.parquet\n")


if __name__ == "__main__":
    main()
