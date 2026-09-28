"""Generate frozen contrast groups and reject all native-tokenizer truncation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from rules import TRAIN_FAMILIES, DEV_FAMILIES, TEST_FAMILIES, make_group, semantic_state


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def strict_encode(row, tokenizer, max_len=512, head_max_len=192, order=None):
    from laya.common import QTYPES, build_sequence, render_options, serialize_state
    q = row["question"]
    native = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    options = render_options(native)
    order = list(range(len(options))) if order is None else list(order)
    if sorted(order) != list(range(len(options))):
        raise ValueError("candidate order is not a permutation")
    texts = [q["instructions"], serialize_state(row["state"]), *options]
    if any(tokenizer.mask_token in value for value in texts):
        raise ValueError("literal mask token in task text")
    head = tokenizer.encode(f"{q['type']} question: {q['instructions']}", add_special_tokens=False)
    blocks = [tokenizer.encode(" " + options[i], add_special_tokens=False) for i in order]
    if any(len(block) > 48 for block in blocks):
        raise ValueError(f"{row['id']}: option would be truncated")
    if len(head) + sum(len(block) + 1 for block in blocks) > head_max_len:
        raise ValueError(f"{row['id']}: head would be truncated")
    reference = [tokenizer.cls_token_id, *head, tokenizer.sep_token_id]
    positions = []
    for block in blocks:
        positions.append(len(reference))
        reference.extend([tokenizer.mask_token_id, *block])
    reference.append(tokenizer.sep_token_id)
    reference.extend(tokenizer.encode(serialize_state(row["state"]), add_special_tokens=False))
    reference.append(tokenizer.sep_token_id)
    if len(reference) > max_len:
        raise ValueError(f"{row['id']}: state would be truncated")
    ids, markers = build_sequence(tokenizer, row["state"], native, max_len, head_max_len, order)
    if ids != reference or markers != positions or len(markers) != len(row["target"]):
        raise ValueError(f"{row['id']}: native sequence differs from complete reference")
    return {**row, "ids": ids, "markers": markers, "qtype": QTYPES[q["type"]],
            "order": order, "target": [row["target"][i] for i in order]}


def generate(seed):
    plans = {"train": (TRAIN_FAMILIES, 96), "dev_seen": (TRAIN_FAMILIES, 16),
             "dev_new": (DEV_FAMILIES, 32), "calibration": (TRAIN_FAMILIES + DEV_FAMILIES, 12),
             "locked_test": (TEST_FAMILIES, 48)}
    splits, seen_states, seen_inputs = {}, {}, {}
    for split, (families, count) in plans.items():
        rows = []
        for family in families:
            rng_seed = int.from_bytes(hashlib.sha256(f"{seed}/{split}/{family}".encode()).digest()[:8], "big")
            rng = random.Random(rng_seed)
            for index in range(count):
                for _ in range(1000):
                    group = make_group(family, split, index, rng)
                    states = {semantic_state(r) for r in group}
                    inputs = {json.dumps([r["state"], r["question"]], sort_keys=True) for r in group}
                    # Family and all its variants have one split; additionally disallow exact
                    # state reuse across splits even when reference identifiers differ.
                    if all(seen_states.get(s, split) == split for s in states) and all(
                            seen_inputs.get(s, split) == split for s in inputs):
                        break
                else:
                    raise RuntimeError("unable to find a disjoint contrast group")
                seen_states.update({s: split for s in states})
                seen_inputs.update({s: split for s in inputs})
                rows.extend(group)
        splits[split] = rows
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer", local_files_only=True)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("data output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    splits = generate(args.seed)
    manifest = {"generator_seed": args.seed, "scope": "synthetic rule-mechanism pilot",
                "train_families": TRAIN_FAMILIES, "dev_only_families": DEV_FAMILIES,
                "locked_test_families": TEST_FAMILIES, "max_len": 512, "head_max_len": 192,
                "source_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("rules.py"))},
                "splits": {}}
    snapshots = {}
    for split, rows in splits.items():
        encoded = [strict_encode(row, tokenizer) for row in rows]
        path = args.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in encoded))
        manifest["splits"][split] = {"rows": len(rows), "groups": len(rows) // 4,
                                     "sha256": digest(path), "max_tokens": max(len(r["ids"]) for r in encoded),
                                     "family_counts": dict(Counter(r["family"] for r in rows)),
                                     "label_counts": dict(Counter(f"{r['question']['type']}:{r['label_index']}" for r in rows))}
        if split == "train":
            for row in encoded:
                snapshots.setdefault(row["question"]["type"], {"id": row["id"], "question": row["question"],
                    "decoded": tokenizer.decode(row["ids"]), "markers": row["markers"], "target": row["target"]})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "token_snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
