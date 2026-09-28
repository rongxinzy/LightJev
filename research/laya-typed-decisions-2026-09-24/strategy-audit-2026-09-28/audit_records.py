#!/usr/bin/env python3
"""Recompute a read-only research audit from tracked training data and artifacts."""
from __future__ import annotations

import collections
import importlib.util
import json
import statistics
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
WF = STUDY / "workflow-generalization"
ARTIFACTS = WF / "artifacts"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rps(prediction, target, order):
    cumulative_prediction = cumulative_target = result = 0.0
    for i in order:
        cumulative_prediction += prediction[i]
        cumulative_target += target[i]
        result += (cumulative_prediction - cumulative_target) ** 2
    return result / (len(order) - 1)


def audit_data():
    spec = importlib.util.spec_from_file_location("records_converter", WF / "prepare_lightjev_records.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    cases = [row for split in ("train", "dev", "calibration")
             for row in read_rows(STUDY / "data" / f"cases_{split}.jsonl")]
    templates = collections.defaultdict(set)
    counts = collections.Counter()
    affected = set()
    margins = []
    for case in cases:
        records = {r["metadata"]["qid"]: r for r in converter.make_records(case)}
        for qid, q in case["questions"].items():
            counts["decisions"] += 1
            templates[case["workflow"]].add(json.dumps(q, sort_keys=True))
            gold = case["gold"][qid]
            probabilities = gold["probabilities"]
            ranked = sorted(probabilities.values(), reverse=True)
            margins.append(ranked[0] - ranked[1])
            label = str(gold["label"]).lower()
            chosen = max(probabilities, key=probabilities.get)
            if chosen != label:
                counts["argmax_ties" if ranked[0] - probabilities[label] < 1e-6
                       else "non_tie_label_disagreement"] += 1
            if q["type"] == "noul" and q.get("criteria"):
                counts["noul_with_criteria"] += 1
                record = records[qid]
                rendered = record["question"] + "\n" + "\n".join(record["candidates"])
                if not any(description in rendered for description in q["criteria"].values()):
                    counts["noul_criteria_dropped"] += 1
                    affected.add((case["workflow"], qid))
    return {
        "source": "official train partition only; train/dev/calibration combined for descriptive audit",
        "cases": len(cases), "counts": dict(counts),
        "question_templates_by_workflow": {k: len(v) for k, v in templates.items()},
        "affected_noul_templates": sorted(affected),
        "fraction_decisions_with_dropped_criteria": counts["noul_criteria_dropped"] / counts["decisions"],
        "fraction_target_margin_below_0_1": sum(x < .1 for x in margins) / len(margins),
    }


def ece(rows):
    bins = [[] for _ in range(15)]
    for row in rows:
        prediction = max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)
        confidence = row["probabilities"][prediction]
        bins[min(int(confidence * 15), 14)].append((confidence, prediction == row["label_index"]))
    return sum(abs(sum(c - a for c, a in bucket)) for bucket in bins) / len(rows)


def audit_predictions(arm):
    root = ARTIFACTS / "results" / "lightjev_objective_ab" / arm
    buckets = collections.defaultdict(list)
    distributions = collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
    calibrated, run_eces = [], []
    for path in sorted(root.glob("*/seed*/raw_predictions.jsonl")):
        for row in read_rows(path):
            prediction = max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)
            is_correct = prediction == row["label_index"]
            targets = sorted(row["target"], reverse=True)
            buckets[row["kind"]].append(is_correct)
            buckets["margin_below_0.1" if targets[0] - targets[1] < .1 else "margin_at_least_0.1"].append(is_correct)
            key = (row["workflow"], row["qid"])
            distributions[key][0][row["labels"][prediction]] += 1
            distributions[key][1][row["labels"][row["label_index"]]] += 1
        calibrated.extend(read_rows(path.with_name("calibrated_predictions.jsonl")))
        report = json.loads(path.with_name("calibrated_report.json").read_text())
        run_eces.append(report["metrics"]["ece_15_bins"])
    collapse = []
    for (workflow, qid), (predicted, gold) in distributions.items():
        n = sum(predicted.values())
        label, count = predicted.most_common(1)[0]
        if count / n > .9:
            collapse.append({"workflow": workflow, "qid": qid, "predicted_label": label,
                             "predicted_fraction": count / n, "gold_fraction": gold[label] / n,
                             "observations_across_four_seeds": n, "unique_cases": n // 4})
    return {"accuracy_slices": {k: {"n_repeated_across_seeds": len(v), "accuracy": statistics.mean(v)}
                                for k, v in buckets.items()},
            "prediction_collapse_over_90_percent": collapse,
            "calibrated_ece_mean_of_16_runs": statistics.mean(run_eces),
            "calibrated_ece_pooled_24000_decisions": ece(calibrated)}


def audit_training():
    result = {}
    for arm, path in [("shared_context", ARTIFACTS / "lightjev_shared_xworkflow/runs"),
                      ("ce", ARTIFACTS / "lightjev_objective_ab/runs/ce"),
                      ("rlcd", ARTIFACTS / "lightjev_objective_ab/runs/rlcd")]:
        manifests = [json.loads(p.read_text()) for p in path.glob("*/seed*/manifest.json")]
        result[arm] = {"runs": len(manifests),
                       "best_step_counts": dict(collections.Counter(d["best_step"] for d in manifests)),
                       "sum_single_gpu_run_hours": sum(d["elapsed_seconds"] for d in manifests) / 3600}
    return result


def main():
    target, near, far = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    evidence = {
        "data": audit_data(),
        "rps_counterexample": {"canonical_order": [0, 1, 2], "presentation_order": [0, 2, 1],
            "canonical_near_error": rps(near, target, [0, 1, 2]),
            "canonical_far_error": rps(far, target, [0, 1, 2]),
            "shuffled_near_error": rps(near, target, [0, 2, 1]),
            "shuffled_far_error": rps(far, target, [0, 2, 1])},
        "predictions": {arm: audit_predictions(arm) for arm in ("ce", "rlcd")},
        "training": audit_training(),
        "scope": "Existing records only. No training, checkpoint edits, or fresh public-test inference.",
    }
    output = Path(__file__).with_name("evidence.json")
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
