#!/usr/bin/env python3
"""Fit per-type temperatures on source workflows and score held-out records."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from lightjev.inference import load_checkpoint
from lightjev.training import _batch
from evaluate_benchmark import score_predictions


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def infer(model, tokenizer, records, max_length, device, batch_size):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            block = records[start:start + batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**_batch(block, tokenizer, max_length, device))
            for record, scores in zip(block, logits):
                meta = record["metadata"]
                rows.append({
                    "case_id": meta["case_id"], "workflow": meta["workflow"],
                    "qid": meta["qid"], "kind": meta["original_kind"],
                    "labels": meta["labels"], "target": record["target"],
                    "label_index": meta["label_index"],
                    "gold_expected_score": meta["gold_expected_score"],
                    "logits": scores.float().cpu().tolist(),
                })
    return rows


def mean_nll(rows, temperature):
    total = 0.0
    for row in rows:
        scaled = [x / temperature for x in row["logits"]]
        peak = max(scaled)
        log_z = peak + math.log(sum(math.exp(x - peak) for x in scaled))
        total += sum(t * (log_z - x) for t, x in zip(row["target"], scaled))
    return total / len(rows)


def fit_temperature(rows):
    # Golden-section minimization in log-temperature over [0.5, 5].
    lo, hi = math.log(0.5), math.log(5.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = mean_nll(rows, math.exp(x1)), mean_nll(rows, math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo); f1 = mean_nll(rows, math.exp(x1))
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo); f2 = mean_nll(rows, math.exp(x2))
    return math.exp((lo + hi) / 2.0)


def probabilities(logits, temperature):
    values = [x / temperature for x in logits]
    peak = max(values)
    exps = [math.exp(x - peak) for x in values]
    total = sum(exps)
    return [x / total for x in exps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=None)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = "cuda:0"
    model, tokenizer, manifest = load_checkpoint(args.checkpoint, device=device)
    max_length = args.max_length or manifest["max_length"]
    calibration = infer(model, tokenizer, read_rows(args.calibration), max_length, device, args.batch_size)
    test = infer(model, tokenizer, read_rows(args.test), max_length, device, args.batch_size)
    kinds = sorted({row["kind"] for row in calibration})
    temperatures = {kind: fit_temperature([r for r in calibration if r["kind"] == kind]) for kind in kinds}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for mode in ("raw", "calibrated"):
        output = []
        for row in test:
            temp = temperatures[row["kind"]] if mode == "calibrated" else 1.0
            result = {k: v for k, v in row.items() if k != "logits"}
            result["probabilities"] = probabilities(row["logits"], temp)
            result["temperature"] = temp
            output.append(result)
        pred = args.output_dir / f"{mode}_predictions.jsonl"
        pred.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
        metrics = score_predictions(output, [])
        metrics["latency_p50_ms_per_case"] = None
        metrics["latency_p95_ms_per_case"] = None
        metrics["decisions_per_second"] = None
        report = {"label": mode, "checkpoint": str(args.checkpoint),
                  "mode": "generalist", "calibration_source": str(args.calibration),
                  "calibration_temperatures_by_type": temperatures,
                  "calibration_nll_by_type": {k: mean_nll([r for r in calibration if r["kind"] == k], temperatures[k]) for k in kinds},
                  "max_length": max_length, "metrics": metrics,
                  "prediction_path": str(pred)}
        (args.output_dir / f"{mode}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        reports[mode] = report["metrics"]
    (args.output_dir / "evaluation.json").write_text(json.dumps({"temperatures": temperatures, "reports": reports}, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), "temperatures": temperatures,
                      "accuracy_raw": reports["raw"]["accuracy"],
                      "accuracy_calibrated": reports["calibrated"]["accuracy"]}), flush=True)

if __name__ == "__main__": main()
