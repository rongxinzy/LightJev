#!/usr/bin/env python3
"""Paired, case-clustered comparison of LightJev folds against Laya arms."""
from __future__ import annotations
import argparse, json, random, statistics
from collections import defaultdict
from pathlib import Path

WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")
SEEDS = (31, 47)
ARMS = {
    "laya_base": "laya_base_raw",
    "laya_rlcd": "rlcd_raw",
    "laya_soft_ce": "soft_ce_raw",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def accuracy_by_case(rows):
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(float(max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__) == row["label_index"]))
    if any(len(values) != 5 for values in by_case.values()):
        raise ValueError("expected exactly five typed decisions per case")
    return {case: statistics.fmean(values) for case, values in by_case.items()}


def bootstrap_diff(lightjev, laya, n_boot, seed):
    rng = random.Random(seed)
    per_workflow = []
    for wf in WORKFLOWS:
        lkeys = sorted(lightjev[wf])
        bkeys = sorted(laya[wf])
        if lkeys != bkeys or len(lkeys) != 300:
            raise ValueError(f"unpaired cases for {wf}: LightJev={len(lkeys)} Laya={len(bkeys)}")
        per_workflow.append([lightjev[wf][k] - laya[wf][k] for k in lkeys])
    point = statistics.fmean(statistics.fmean(v) for v in per_workflow)
    draws = []
    for _ in range(n_boot):
        wf_diffs = []
        for values in per_workflow:
            wf_diffs.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
        draws.append(statistics.fmean(wf_diffs))
    draws.sort()
    lo = draws[int(0.025 * (n_boot - 1))]
    hi = draws[int(0.975 * (n_boot - 1))]
    return {"difference": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "bootstrap_probability_lightjev_better": sum(value > 0 for value in draws) / n_boot,
            "resamples": n_boot, "unit": "case-clustered, stratified by held-out workflow"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, required=True,
                   help="workflow-generalization directory containing artifacts/ and results/")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()
    base = args.experiment_root
    all_metrics = {}
    per_workflow_metrics = {}
    metric_keys = ("accuracy", "soft_accuracy_all_types", "kl_from_gold_all_types", "brier_all_types", "target_cross_entropy_all_types", "total_variation_all_types", "ece_15_bins")
    lightjev_case_acc = {wf: {} for wf in WORKFLOWS}
    laya_case_acc = {arm: {wf: {} for wf in WORKFLOWS} for arm in ARMS}
    for wf in WORKFLOWS:
        fold = {}
        seed_rows = []
        for seed in SEEDS:
            result_dir = base / "artifacts" / "results" / "lightjev_xworkflow" / wf / f"seed{seed}"
            for mode in ("raw", "calibrated"):
                rows = read_jsonl(result_dir / f"{mode}_predictions.jsonl")
                if len(rows) != 1500:
                    raise ValueError(f"expected 1500 rows: {result_dir}/{mode}")
                report = json.loads((result_dir / f"{mode}_report.json").read_text())
                fold[f"lightjev_seed{seed}_{mode}"] = report["metrics"]
                if mode == "calibrated":
                    seed_rows.extend(rows)
                    lightjev_case_acc[wf][seed] = accuracy_by_case(rows)
        for arm, label in ARMS.items():
            rows = read_jsonl(base / "artifacts" / "results" / wf / f"{label}_predictions.jsonl")
            if len(rows) != 1500:
                raise ValueError(f"expected 1500 Laya rows: {wf}/{label}")
            report_path = base / "artifacts" / "results" / wf / f"{label}_report.json"
            fold[arm] = json.loads(report_path.read_text())["metrics"]
            laya_case_acc[arm][wf] = accuracy_by_case(rows)
        fold["lightjev_mean_two_seeds_calibrated"] = {
            key: statistics.fmean(fold[f"lightjev_seed{s}_calibrated"][key] for s in SEEDS)
            for key in metric_keys
        }
        per_workflow_metrics[wf] = fold
    # Two seed predictions are repeated measurements for the same cases. Accuracy
    # and proper scores here are their arithmetic mean, not a probability ensemble.
    summary = {}
    for arm in ARMS:
        summary[arm] = {"accuracy": statistics.fmean(per_workflow_metrics[w][arm]["accuracy"] for w in WORKFLOWS)}
    lightjev_acc = statistics.fmean(per_workflow_metrics[w]["lightjev_mean_two_seeds_calibrated"]["accuracy"] for w in WORKFLOWS)
    lightjev_metrics = {key: statistics.fmean(per_workflow_metrics[wf][f"lightjev_seed{seed}_calibrated"][key]
                                                    for wf in WORKFLOWS for seed in SEEDS)
                        for key in metric_keys}
    lightjev_metrics.update(cases_per_seed=1200, decisions_per_seed=6000,
                            cases_scored_once=1200, seed_repetitions=2)
    comparisons = {}
    for arm in ARMS:
        means = {wf: {case: statistics.fmean([lightjev_case_acc[wf][seed][case] for seed in SEEDS])
                      for case in lightjev_case_acc[wf][SEEDS[0]]} for wf in WORKFLOWS}
        comparisons[f"lightjev_vs_{arm}"] = bootstrap_diff(means, laya_case_acc[arm], args.bootstrap, 20260925)
    all_metrics = {
        "protocol": "four workflow-held-out folds; LightJev two seeds per fold; calibration on source workflows only",
        "note": "Exploratory. No official public test split used. LightJev was trained/evaluated up to 640 tokens although the released checkpoint was trained to 256; see report caveat.",
        "decisions_per_seed": 6000,
        "lightjev_two_seed_calibrated_mean": lightjev_metrics,
        "lightjev_two_seed_accuracy": lightjev_acc,
        "laya_accuracy_by_arm": summary,
        "paired_case_bootstrap": comparisons,
        "per_workflow": per_workflow_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False) + "\n")
    lines = ["# LightJev versus Laya: workflow-held-out comparison", "",
             "Four folds, two LightJev seeds per fold. Every held-out case is scored once per model/seed; two-seed LightJev metrics are the arithmetic mean of runs, not a probability ensemble.", "",
             "Exploratory: prior work inspected these workflow families and public-test results. The official public test split was not used here. LightJev inputs are run at 640 tokens, above its released 256-token training window.", "",
             "| Model | Accuracy |", "|---|---:|",
             f"| LightJev mean across seeds | {lightjev_metrics['accuracy']:.3f} |"]
    for arm in ARMS:
        lines.append(f"| {arm} | {summary[arm]['accuracy']:.3f} |")
    lines += ["", "## Paired case-cluster bootstrap", "",
              "Intervals resample cases within each held-out workflow (10,000 draws by default); the five questions on one state stay clustered.", "",
              "| Comparison | Difference | 95% CI | P(LightJev better) |", "|---|---:|---:|---:|"]
    for name, result in comparisons.items():
        lines.append(f"| {name} | {result['difference']:+.3f} | [{result['ci95_low']:+.3f}, {result['ci95_high']:+.3f}] | {result['bootstrap_probability_lightjev_better']:.3f} |")
    lines += ["", "## Accuracy by held-out workflow", "", "| Workflow | LightJev (2-seed mean) | Laya RLCD | Laya soft-CE |", "|---|---:|---:|---:|"]
    for wf in WORKFLOWS:
        row = per_workflow_metrics[wf]
        lines.append(f"| {wf} | {row['lightjev_mean_two_seeds_calibrated']['accuracy']:.3f} | {row['laya_rlcd']['accuracy']:.3f} | {row['laya_soft_ce']['accuracy']:.3f} |")
    lines += ["", "Full distribution metrics and per-seed reports are in `comparison.json`.", ""]
    (args.output_dir / "comparison.md").write_text("\n".join(lines))
    print(json.dumps({"lightjev_accuracy": lightjev_metrics["accuracy"], "laya": summary,
                      "paired_case_bootstrap": comparisons}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
