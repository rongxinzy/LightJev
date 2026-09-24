#!/usr/bin/env python3
"""Pin-agnostic preprocessing for the public LocalLLaMA/typed-decisions parquet files."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from laya.common import QTYPES, build_sequence


def parse_cases(path: Path) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    cases = []
    for row in rows:
        cases.append({
            "id": row["id"], "workflow": row["workflow"],
            "state": json.loads(row["state"]),
            "questions": json.loads(row["questions"]),
            "gold": json.loads(row["gold"]),
        })
    return cases


def split_train_cases(cases: list[dict], seed: int) -> dict[str, list[dict]]:
    by_workflow: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_workflow[case["workflow"]].append(case)
    out = {key: [] for key in ("train", "dev", "calibration")}
    for workflow, group in sorted(by_workflow.items()):
        group = sorted(group, key=lambda x: x["id"])
        random.Random(seed + sum(map(ord, workflow))).shuffle(group)
        n = len(group)
        n_dev = max(1, round(n * 0.10))
        n_cal = max(1, round(n * 0.10))
        out["dev"].extend(group[:n_dev])
        out["calibration"].extend(group[n_dev:n_dev + n_cal])
        out["train"].extend(group[n_dev + n_cal:])
    return out


def encode_case(case: dict, tok, max_len: int, head_max_len: int) -> list[dict]:
    items = []
    if len(case["questions"]) != 5:
        raise ValueError(f"expected 5 questions in {case['id']}")
    for qid, q in case["questions"].items():
        typ = q["type"]
        if typ not in QTYPES:
            raise ValueError(f"unknown question type {typ} in {case['id']}:{qid}")
        criteria = q.get("criteria", {})
        gold = case["gold"][qid]
        if typ == "choice":
            labels = list(criteria.keys())
            target = [float(gold["probabilities"][label]) for label in labels]
            gold_label = str(gold["label"])
            label_index = labels.index(gold_label)
        elif typ == "noul":
            labels = ["false", "true"]
            target = [float(gold["probabilities"][label]) for label in labels]
            label_index = labels.index(str(gold["label"]).lower())
        else:
            labels = [str(i) for i in range(len(criteria))]
            target = [float(gold["probabilities"][label]) for label in labels]
            label_index = int(gold["label"])
        total = sum(target)
        if not target or abs(total - 1.0) > 0.02 or any(x < 0 for x in target):
            raise ValueError(f"invalid gold distribution in {case['id']}:{qid}: {target}")
        target = [x / total for x in target]
        q_for_laya = {"t": typ, "ins": q["instructions"], "crit": criteria}
        ids, markers = build_sequence(tok, case["state"], q_for_laya,
                                      max_len=max_len, head_max_len=head_max_len)
        if len(markers) != len(target) or not ids or len(ids) > max_len:
            raise ValueError(f"sequence/target mismatch in {case['id']}:{qid}")
        items.append({
            "case_id": case["id"], "workflow": case["workflow"], "qid": qid,
            "kind": typ, "labels": labels, "ids": ids, "markers": markers,
            "qtype": QTYPES[typ], "target": target, "label_index": label_index,
        })
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--test-parquet", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer")
    raw_train = parse_cases(args.train_parquet)
    raw_test = parse_cases(args.test_parquet)
    partitions = split_train_cases(raw_train, args.seed)
    partitions["test"] = raw_test

    id_sets = {name: {x["id"] for x in cases} for name, cases in partitions.items()}
    state_sets = {name: {json.dumps(x["state"], sort_keys=True) for x in cases}
                  for name, cases in partitions.items()}
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if id_sets[left] & id_sets[right] or state_sets[left] & state_sets[right]:
                raise ValueError(f"case/state leakage between {left} and {right}")

    manifest = {"seed": args.seed, "case_counts": {}, "question_counts": {},
                "workflow_case_counts": {}, "max_len": args.max_len,
                "head_max_len": args.head_max_len, "type_counts": {}}
    for name, cases in partitions.items():
        write_jsonl(args.output_dir / f"cases_{name}.jsonl", cases)
        items = [item for case in cases
                 for item in encode_case(case, tok, args.max_len, args.head_max_len)]
        write_jsonl(args.output_dir / f"items_{name}.jsonl", items)
        manifest["case_counts"][name] = len(cases)
        manifest["question_counts"][name] = len(items)
        manifest["workflow_case_counts"][name] = dict(Counter(x["workflow"] for x in cases))
        manifest["type_counts"][name] = dict(Counter(x["kind"] for x in items))
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
