#!/usr/bin/env python3
"""Paired case/seed comparison for the supervised CE and RLCD objective arms."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from compare_lightjev_to_laya import WORKFLOWS, accuracy_by_case, read_jsonl


def bootstrap(differences, seeds, n_boot, include_seed):
    rng = random.Random(20260929 if include_seed else 20260928)
    seed_draws = []
    for _ in range(n_boot):
        sampled_seeds = ([seeds[rng.randrange(len(seeds))] for _ in seeds]
                         if include_seed else list(seeds))
        fold_means = []
        for wf in WORKFLOWS:
            case_keys = sorted(differences[wf][seeds[0]])
            sample = []
            for _ in case_keys:
                case = case_keys[rng.randrange(len(case_keys))]
                sample.append(statistics.fmean(differences[wf][s][case] for s in sampled_seeds))
            fold_means.append(statistics.fmean(sample))
        seed_draws.append(statistics.fmean(fold_means))
    seed_draws.sort()
    return {
        "ci95_low": seed_draws[int(0.025 * (n_boot - 1))],
        "ci95_high": seed_draws[int(0.975 * (n_boot - 1))],
        "probability_rlcd_better": sum(x > 0 for x in seed_draws) / n_boot,
        "resamples": n_boot,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="31,47,73,97")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique seeds are required")
    root = args.experiment_root
    reports = {arm: {} for arm in ("ce", "rlcd")}
    case_acc = {arm: {wf: {} for wf in WORKFLOWS} for arm in reports}
    per_seed_macro = {arm: {} for arm in reports}
    metric_keys = ("accuracy", "soft_accuracy_all_types", "target_cross_entropy_all_types",
                   "brier_all_types", "ece_15_bins")
    by_seed_wf = {arm: {seed: [] for seed in seeds} for arm in reports}
    for arm in reports:
        for seed in seeds:
            for wf in WORKFLOWS:
                d = root / "artifacts" / "results" / "lightjev_objective_ab" / arm / wf / f"seed{seed}"
                rows = read_jsonl(d / "calibrated_predictions.jsonl")
                if len(rows) != 1500:
                    raise ValueError(f"expected 1,500 calibrated rows at {d}")
                case_acc[arm][wf][seed] = accuracy_by_case(rows)
                report = json.loads((d / "calibrated_report.json").read_text())
                reports[arm][(wf, seed)] = report["metrics"]
                by_seed_wf[arm][seed].append(report["metrics"]["accuracy"])
    diffs = {wf: {seed: {} for seed in seeds} for wf in WORKFLOWS}
    for wf in WORKFLOWS:
        for seed in seeds:
            a, b = case_acc["rlcd"][wf][seed], case_acc["ce"][wf][seed]
            if sorted(a) != sorted(b) or len(a) != 300:
                raise ValueError(f"unpaired cases for {wf}, seed {seed}")
            diffs[wf][seed] = {case: a[case] - b[case] for case in a}
    arm_metrics = {
        arm: {metric: statistics.fmean(reports[arm][(wf, seed)][metric]
                                       for wf in WORKFLOWS for seed in seeds)
              for metric in metric_keys}
        for arm in reports
    }
    fold_metrics = {
        arm: {wf: {metric: statistics.fmean(reports[arm][(wf, seed)][metric] for seed in seeds)
                   for metric in metric_keys} for wf in WORKFLOWS}
        for arm in reports
    }
    seed_accuracy = {
        arm: {str(seed): statistics.fmean(by_seed_wf[arm][seed]) for seed in seeds}
        for arm in reports
    }
    point = statistics.fmean(
        statistics.fmean(statistics.fmean(diffs[wf][seed].values()) for seed in seeds)
        for wf in WORKFLOWS)
    result = {
        "protocol": f"paired four-workflow comparison; matched seeds {list(seeds)}; calibrated probabilities; 1,200 held-out cases",
        "note": "Exploratory; same public training split and four folds as prior study; official public test split unused.",
        "ce_metrics": arm_metrics["ce"], "rlcd_metrics": arm_metrics["rlcd"],
        "difference_rlcd_minus_ce": point,
        "case_bootstrap": bootstrap(diffs, seeds, args.bootstrap, include_seed=False),
        "seed_and_case_bootstrap": bootstrap(diffs, seeds, args.bootstrap, include_seed=True),
        "seed_macro_accuracy": seed_accuracy,
        "per_workflow": {wf: {arm: fold_metrics[arm][wf] for arm in reports} for wf in WORKFLOWS},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "objective_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Paired CE versus RLCD objective comparison", "",
             f"Four held-out workflows and {len(seeds)} matched seeds. Each seed scores the same 300 case clusters per fold.",
             "Case-only intervals condition on the observed seeds. Seed-and-case intervals resample matched training seeds and case clusters within workflow.", "",
             "| Arm | Accuracy | Soft accuracy | Target CE ↓ | Brier ↓ | ECE ↓ |", "|---|---:|---:|---:|---:|---:|"]
    for arm, label in (("ce", "CE"), ("rlcd", "CE + RLCD")):
        m = arm_metrics[arm]
        lines.append(f"| {label} | {m['accuracy']:.4f} | {m['soft_accuracy_all_types']:.4f} | {m['target_cross_entropy_all_types']:.4f} | {m['brier_all_types']:.4f} | {m['ece_15_bins']:.4f} |")
    lines += ["", f"RLCD minus CE accuracy: {point:+.4f} ({point*100:+.2f} percentage points).",
              f"Case bootstrap 95% CI: [{result['case_bootstrap']['ci95_low']:+.4f}, {result['case_bootstrap']['ci95_high']:+.4f}]; P(RLCD > CE)={result['case_bootstrap']['probability_rlcd_better']:.3f}.",
              f"Seed-and-case bootstrap 95% CI: [{result['seed_and_case_bootstrap']['ci95_low']:+.4f}, {result['seed_and_case_bootstrap']['ci95_high']:+.4f}]; P(RLCD > CE)={result['seed_and_case_bootstrap']['probability_rlcd_better']:.3f}.", "",
              "| Workflow | CE accuracy | CE+RLCD accuracy | Difference |", "|---|---:|---:|---:|"]
    for wf in WORKFLOWS:
        ce = fold_metrics["ce"][wf]["accuracy"]
        rlcd = fold_metrics["rlcd"][wf]["accuracy"]
        lines.append(f"| {wf} | {ce:.4f} | {rlcd:.4f} | {rlcd-ce:+.4f} |")
    lines += ["", "## Macro accuracy by training seed", "", "| Seed | CE | CE + RLCD |", "|---:|---:|---:|"]
    for seed in seeds:
        lines.append(f"| {seed} | {seed_accuracy['ce'][str(seed)]:.4f} | {seed_accuracy['rlcd'][str(seed)]:.4f} |")
    lines += ["", "All metrics are measured on the same held-out decisions after each arm's source-only per-type temperature fit.", ""]
    (args.output_dir / "objective_comparison.md").write_text("\n".join(lines))
    print(json.dumps({"difference": point, "case_bootstrap": result["case_bootstrap"],
                      "seed_and_case_bootstrap": result["seed_and_case_bootstrap"],
                      "metrics": arm_metrics}, indent=2))


if __name__ == "__main__":
    main()
