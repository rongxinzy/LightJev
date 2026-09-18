#!/usr/bin/env python3
"""Evaluate a release checkpoint once loaded; fit temperature on calibration only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lightjev.inference import load_checkpoint
from lightjev.metrics import evaluate, fit_temperature, temperature_scale
from lightjev.schema import load_records
from lightjev.training import _batch, check_disjoint

SPLITS = ("dev", "calibration", "test", "ood")


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _prediction(row, probabilities, temperature):
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    result = {"id": row["id"], "kind": row["kind"], "candidates": row["candidates"],
              "probabilities": probabilities, "temperature": temperature,
              "selected": row["candidates"][index], "selected_index": index}
    if row["kind"] == "score":
        result["expectation"] = sum(i * p for i, p in enumerate(probabilities))
    return result


def _summarize(records, predictions):
    def subset(rows):
        ids = {row["id"] for row in rows}
        return evaluate(rows, [p for p in predictions if p["id"] in ids]) if rows else None
    hard = [r for r in records if sum(p == 1 for p in r["target"]) == 1]
    soft = [r for r in records if r.get("provenance", {}).get("target_kind") == "programmatic_conditional_distribution"]
    families = sorted({r.get("provenance", {}).get("family_id", "unspecified") for r in records})
    return {"all": subset(records), "hard": subset(hard), "soft": subset(soft),
            "families": {family: subset([r for r in records if r.get("provenance", {}).get("family_id", "unspecified") == family])
                         for family in families}}


def run(checkpoint, data_dir, output_dir, device="cpu", batch_size=8):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    data_dir, output_dir, checkpoint = Path(data_dir), Path(output_dir), Path(checkpoint)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output directory must be new or empty")
    records = {split: load_records(data_dir / f"{split}.jsonl") for split in SPLITS}
    for i, split in enumerate(SPLITS):
        if any("target" not in r for r in records[split]):
            raise ValueError(f"{split} requires labeled records")
        for previous in SPLITS[:i]:
            check_disjoint(records[previous], records[split])
    # When available, independently reject accidental train/evaluation overlap.
    if (data_dir / "train.jsonl").exists():
        train = load_records(data_dir / "train.jsonl")
        for split in SPLITS:
            check_disjoint(train, records[split])
    started = time.perf_counter()
    model, tokenizer, config = load_checkpoint(checkpoint, device=device)
    model.float().eval()
    raw = {}
    with torch.inference_mode():
        for split, rows in records.items():
            predictions = []
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                logits = model(**_batch(batch, tokenizer, config["max_length"], device))
                if len(logits) != len(batch):
                    raise ValueError("model returned incorrect question count")
                for row, scores in zip(batch, logits):
                    probabilities = scores.float().softmax(-1)
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("nonfinite probabilities")
                    predictions.append(_prediction(row, probabilities.cpu().tolist(), 1.0))
            raw[split] = predictions
    calibration = fit_temperature(records["calibration"], raw["calibration"])
    temperature = calibration["temperature"]
    calibrated = {split: [_prediction(row, temperature_scale(pred["probabilities"], temperature), temperature)
                           for row, pred in zip(records[split], raw[split])] for split in SPLITS}
    report = {"checkpoint_manifest": config, "device": device, "dtype": "float32", "batch_size": batch_size,
              "calibration": calibration, "temperature_fit_split": "calibration",
              "heldout_splits": ["test", "ood"], "calibration_guaranteed": False,
              "subset_definitions": {"hard": "one-hot targets", "soft": "provenance.target_kind == programmatic_conditional_distribution",
                                     "note": "Hard and soft subsets can overlap if a conditional distribution is degenerate."},
              "data_sha256": {s: hashlib.sha256((data_dir / f"{s}.jsonl").read_bytes()).hexdigest() for s in SPLITS},
              "splits": {s: {"role": "heldout" if s in ("test", "ood") else "calibration_fit" if s == "calibration" else "checkpoint_selection",
                             "raw": _summarize(records[s], raw[s]), "calibrated": _summarize(records[s], calibrated[s])} for s in SPLITS},
              "elapsed_seconds": time.perf_counter() - started,
              "note": "One global temperature fitted only on calibration; dev selected checkpoint, calibration fit metrics are in-sample. Probability scaling preserves exact zero support."}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write(output_dir / f"{split}.predictions.json", raw[split])
        _write(output_dir / f"{split}.calibrated.predictions.json", calibrated[split])
    _write(output_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        report = run(args.checkpoint, args.data_dir, args.output_dir, args.device, args.batch_size)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(Path(args.output_dir) / "report.json"), "temperature": report["calibration"]["temperature"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
