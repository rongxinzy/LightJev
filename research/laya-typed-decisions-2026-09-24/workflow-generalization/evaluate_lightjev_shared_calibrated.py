#!/usr/bin/env python3
"""Calibrate and evaluate shared-context checkpoints on held-out workflows."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import torch
from lightjev.inference import load_checkpoint
from evaluate_benchmark import score_predictions
from evaluate_lightjev_calibrated import fit_temperature, mean_nll, probabilities
from shared_context import infer_probabilities


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def predict(model, tokenizer, rows, max_length, device, batch_size, variants):
    probs = infer_probabilities(model, tokenizer, rows, max_length, device, batch_size, variants)
    result = []
    for row, p in zip(rows, probs):
        meta = row["metadata"]
        p = p.double().tolist()
        result.append({"case_id": meta["case_id"], "workflow": meta["workflow"],
                       "qid": meta["qid"], "kind": meta["original_kind"],
                       "labels": meta["labels"], "target": row["target"],
                       "label_index": meta["label_index"],
                       "gold_expected_score": meta["gold_expected_score"],
                       "logits": [math.log(max(x, 1e-30)) for x in p], "probabilities_raw": p})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--permutations", type=int, default=4)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    model, tokenizer, manifest = load_checkpoint(args.checkpoint, device="cuda:0")
    max_length = args.max_length or manifest["max_length"]
    calibration = predict(model, tokenizer, read_rows(args.calibration), max_length,
                          "cuda:0", args.batch_size, args.permutations)
    test = predict(model, tokenizer, read_rows(args.test), max_length,
                   "cuda:0", args.batch_size, args.permutations)
    kinds = sorted({r["kind"] for r in calibration})
    temperatures = {k: fit_temperature([r for r in calibration if r["kind"] == k]) for k in kinds}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in ("raw", "calibrated"):
        out = []
        for row in test:
            temp = temperatures[row["kind"]] if mode == "calibrated" else 1.0
            # Log of the permutation-mean distribution reproduces its raw probabilities.
            row_logits = row["logits"]
            result = {k: v for k, v in row.items() if k not in {"logits", "probabilities_raw"}}
            result["probabilities"] = probabilities(row_logits, temp)
            result["temperature"] = temp
            out.append(result)
        pred = args.output_dir / f"{mode}_predictions.jsonl"
        pred.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        metrics = score_predictions(out, [])
        metrics["latency_p50_ms_per_case"] = None
        metrics["latency_p95_ms_per_case"] = None
        metrics["decisions_per_second"] = None
        report = {"label": mode, "checkpoint_id": args.checkpoint.name,
                  "architecture": "shared-context-causal-listwise",
                  "candidate_order_permutations_max": args.permutations,
                  "calibration_temperatures_by_type": temperatures,
                  "calibration_nll_by_type": {k: mean_nll([r for r in calibration if r["kind"] == k], temperatures[k]) for k in kinds},
                  "max_length": max_length, "metrics": metrics,
                  "prediction_path": pred.name}
        (args.output_dir / f"{mode}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        reports[mode] = metrics
    (args.output_dir / "evaluation.json").write_text(json.dumps({"temperatures": temperatures,
                      "reports": reports, "candidate_order_permutations_max": args.permutations}, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), "temperatures": temperatures,
                      "accuracy_raw": reports["raw"]["accuracy"],
                      "accuracy_calibrated": reports["calibrated"]["accuracy"]}), flush=True)


if __name__ == "__main__": main()
