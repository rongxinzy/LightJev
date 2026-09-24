#!/usr/bin/env python3
"""8-GPU zero-shot LightJev evaluation on LocalLLaMA/typed-decisions test cases."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from lightjev.inference import load_checkpoint
from lightjev.training import _batch
from evaluate_benchmark import score_predictions


def make_record(case, qid, q, gold):
    typ = q["type"]
    criteria = q.get("criteria", {})
    if typ == "choice":
        labels = list(criteria.keys())
        candidates = [f"{label}: {criteria[label]}" for label in labels]
        kind = "choice"
    elif typ == "noul":
        labels, candidates, kind = ["false", "true"], ["false", "true"], "boolean"
    else:
        labels = [str(i) for i in range(len(criteria))]
        candidates = [f"{i}: {desc}" for i, desc in enumerate(criteria)]
        kind = "score"
    target = [float(gold["probabilities"][label]) for label in labels]
    total = sum(target)
    target = [x / total for x in target]
    label = str(gold["label"]).lower()
    return {
        "record": {"id": f"{case['id']}:{qid}", "group_id": case["id"],
                   "state": json.dumps(case["state"], ensure_ascii=False, separators=(",", ":")),
                   "question": q["instructions"],
                   "kind": kind, "candidates": candidates},
        "metadata": {"case_id": case["id"], "workflow": case["workflow"], "qid": qid,
                     "kind": "noul" if typ == "noul" else typ,
                     "labels": labels, "target": target,
                     "label_index": labels.index(label),
                     "gold_expected_score": float(gold["score"]) if typ == "score" else None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--label", default="lightjev_zero_shot")
    ap.add_argument("--batch-cases", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=None,
                    help="explicit non-truncating input limit; use > checkpoint training window only with that caveat reported")
    args = ap.parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world, local_rank = 0, 1, 0
    device = f"cuda:{local_rank}"
    config = json.loads((args.checkpoint / "manifest.json").read_text())
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=device)
    max_length = config["max_length"] if args.max_length is None else args.max_length
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    mine = cases[rank::world]
    rows, latencies = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(mine), args.batch_cases):
            block = mine[start:start + args.batch_cases]
            packaged = [make_record(case, qid, question, case["gold"][qid])
                        for case in block for qid, question in case["questions"].items()]
            records = [entry["record"] for entry in packaged]
            began = time.perf_counter()
            batch = _batch(records, tokenizer, max_length, device)
            logits = model(**batch)
            latency = (time.perf_counter() - began) * 1000
            # Attribute equal wall-time shares to cases in this batch for descriptive throughput only.
            latencies.extend([latency / len(block)] * len(block))
            for entry, distribution in zip(packaged, logits):
                meta = entry["metadata"]
                probs = distribution.float().softmax(-1).cpu().tolist()
                expected = (sum(i * p for i, p in enumerate(probs))
                            if meta["kind"] == "score" else None)
                rows.append({"case_id": meta["case_id"], "workflow": meta["workflow"],
                             "qid": meta["qid"], "kind": meta["kind"],
                             "labels": meta["labels"],
                             "probabilities": probs, "target": meta["target"],
                             "label_index": meta["label_index"],
                             "predicted_expected_score": expected,
                             "gold_expected_score": meta["gold_expected_score"]})
    gathered = [None] * world if rank == 0 else None
    if distributed:
        dist.gather_object((rows, latencies), gathered, dst=0)
    else:
        gathered = [(rows, latencies)]
    if rank == 0:
        all_rows = [x for part, _ in gathered for x in part]
        all_latencies = [x for _, part in gathered for x in part]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = args.output_dir / f"{args.label}_predictions.jsonl"
        pred_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_rows), encoding="utf-8")
        summary = {"label": args.label, "mode": "generalist",
                   "mode_details": "zero-shot on these four benchmark workflows; criteria descriptions are part of candidate text",
                   "checkpoint": str(args.checkpoint),
                   "max_length": max_length,
                   "latency_note": "8-rank evaluation batched two cases at a time; latency is descriptive and not comparable to sequential Laya Agent or leaderboard client timings",
                   "metrics": score_predictions(all_rows, all_latencies), "prediction_path": str(pred_path)}
        (args.output_dir / f"{args.label}_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__": main()
