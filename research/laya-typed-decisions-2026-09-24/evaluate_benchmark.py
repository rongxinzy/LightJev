#!/usr/bin/env python3
"""Official Laya-style end-to-end evaluation and full-distribution benchmark report."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def read_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(values)] * len(values)
    return [max(0.0, x) / total for x in values]


def ece(confs: list[float], corrects: list[float], bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(confs)
    total = 0.0
    for i in range(bins):
        selected = [(c, a) for c, a in zip(confs, corrects)
                    if (c >= edges[i] if i == 0 else c > edges[i]) and c <= edges[i + 1]]
        if selected:
            total += len(selected) / n * abs(
                sum(x[0] for x in selected) / len(selected)
                - sum(x[1] for x in selected) / len(selected))
    return total


def score_predictions(rows: list[dict], latencies: list[float]) -> dict:
    hard, confs, corrects, scores = [], [], [], []
    dist_metrics = defaultdict(list)
    by_kind, by_workflow = defaultdict(list), defaultdict(list)
    for row in rows:
        qtype = row["kind"]
        p = normalize(row["probabilities"])
        y = normalize(row["target"])
        idx = max(range(len(p)), key=p.__getitem__)
        hit = float(idx == row["label_index"])
        hard.append(hit)
        confs.append(max(p))
        corrects.append(hit)
        by_kind[qtype].append(hit)
        by_workflow[row["workflow"]].append(hit)
        eps = 1e-12
        ce = -sum(t * math.log(max(eps, v)) for t, v in zip(y, p))
        kl = sum(t * math.log(max(eps, t) / max(eps, v)) for t, v in zip(y, p) if t > 0)
        brier = sum((v - t) ** 2 for t, v in zip(y, p))
        soft = sum(v * t for v, t in zip(p, y))
        tv = 0.5 * sum(abs(v - t) for v, t in zip(p, y))
        for name, value in (("target_cross_entropy", ce), ("kl_from_gold", kl),
                            ("brier", brier), ("soft_accuracy", soft), ("total_variation", tv)):
            dist_metrics[name].append(value)
        if qtype in ("choice", "noul"):
            scores.append((ce, kl, brier, soft, tv))
        if qtype == "score":
            pred_expected = sum(i * value for i, value in enumerate(p))
            abs_error = abs(pred_expected - row["gold_expected_score"])
            row["predicted_expected_score"] = pred_expected
            row["score_absolute_error"] = abs_error

    scores_out = {
        "cases": len({row["case_id"] for row in rows}), "decisions": len(rows),
        "accuracy": sum(hard) / max(1, len(hard)),
        "soft_accuracy_all_types": statistics.fmean(dist_metrics["soft_accuracy"]),
        "kl_from_gold_all_types": statistics.fmean(dist_metrics["kl_from_gold"]),
        "brier_all_types": statistics.fmean(dist_metrics["brier"]),
        "target_cross_entropy_all_types": statistics.fmean(dist_metrics["target_cross_entropy"]),
        "total_variation_all_types": statistics.fmean(dist_metrics["total_variation"]),
        "ece_15_bins": ece(confs, corrects),
        "latency_p50_ms_per_case": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms_per_case": float(np.percentile(latencies, 95)) if latencies else None,
        "decisions_per_second": len(rows) / max(1e-6, sum(latencies) / 1000.0),
        "by_type_accuracy": {k: statistics.fmean(v) for k, v in sorted(by_kind.items())},
        "by_workflow_accuracy": {k: statistics.fmean(v) for k, v in sorted(by_workflow.items())},
        "laya_notebook_metrics": {
            "accuracy": statistics.fmean(hard),
            "soft_accuracy_choice_noul": statistics.fmean(x[3] for x in scores),
            "brier_choice_noul": statistics.fmean(x[2] for x in scores),
            "kl_choice_noul": statistics.fmean(x[1] for x in scores),
            "ece_15_bins": ece(confs, corrects),
            "score_mae": statistics.fmean(r["score_absolute_error"] for r in rows if r["kind"] == "score"),
            "score_within_one": statistics.fmean(
                float(r["score_absolute_error"] <= 1.0) for r in rows if r["kind"] == "score"),
        },
    }
    return scores_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=("specialist", "generalist"), required=True)
    args = ap.parse_args()
    distributed = "RANK" in __import__("os").environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world, local_rank = 0, 1, 0
    device = f"cuda:{local_rank}"
    cases = read_cases(args.cases)
    mine = cases[rank::world]
    import laya
    agent = laya.Agent(str(args.model_dir), device=device)
    if mine:
        agent.predict(mine[0]["state"], mine[0]["questions"])  # warm-up, excluded from latency
    predictions, latencies = [], []
    for case in mine:
        start = time.perf_counter()
        result = agent.predict(case["state"], case["questions"])
        latencies.append((time.perf_counter() - start) * 1000.0)
        answers = result["answers"]
        for qid, question in case["questions"].items():
            typ, gold = question["type"], case["gold"][qid]
            if typ == "choice":
                labels = list(question["criteria"])
                probs = normalize([float(answers[qid]["probabilities"].get(k, 0.0)) for k in labels])
                target = normalize([float(gold["probabilities"][k]) for k in labels])
                label_index = labels.index(str(gold["label"]))
            elif typ == "noul":
                labels = ["false", "true"]
                p_true = float(answers[qid]["noul"])
                probs = [1.0 - p_true, p_true]
                target = [float(gold["probabilities"]["false"]), float(gold["probabilities"]["true"])]
                label_index = 1 if str(gold["label"]).lower() == "true" else 0
            else:
                n = len(question["criteria"])
                labels = [str(i) for i in range(n)]
                probs = normalize([float(answers[qid]["probabilities"].get(k, 0.0)) for k in labels])
                target = normalize([float(gold["probabilities"][k]) for k in labels])
                label_index = int(gold["label"])
            predictions.append({
                "case_id": case["id"], "workflow": case["workflow"], "qid": qid,
                "kind": typ, "labels": labels, "probabilities": probs, "target": target,
                "label_index": label_index,
                "predicted_expected_score": sum(i * value for i, value in enumerate(probs)) if typ == "score" else None,
                "gold_expected_score": float(gold["score"]) if typ == "score" else None,
            })
    gathered = [None] * world if rank == 0 else None
    if distributed:
        dist.gather_object((predictions, latencies), gathered, dst=0)
    else:
        gathered = [(predictions, latencies)]
    if rank == 0:
        all_predictions = [row for pred, _ in gathered for row in pred]
        all_latencies = [x for _, lat in gathered for x in lat]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = args.output_dir / f"{args.label}_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in all_predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {"label": args.label, "mode": args.mode,
                   "model_dir": str(args.model_dir),
                   "metrics": score_predictions(all_predictions, all_latencies),
                   "prediction_path": str(pred_path)}
        out = args.output_dir / f"{args.label}_report.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
