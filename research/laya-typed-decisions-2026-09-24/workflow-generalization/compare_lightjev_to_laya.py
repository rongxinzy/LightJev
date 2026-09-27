#!/usr/bin/env python3
"""Paired, case-clustered comparison of LightJev folds against Laya arms."""
from __future__ import annotations
import argparse, json, random, statistics
from collections import defaultdict
from pathlib import Path

WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")
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


def bootstrap_diff(lightjev, laya, n_boot, seed, seed_count=2):
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
            "resamples": n_boot,
            "unit": f"case-clustered and stratified by workflow; conditional on {seed_count} observed LightJev seeds"}


def bootstrap_seed_and_case_diff(lightjev_by_seed, laya, n_boot, seed):
    """Resample LightJev training seeds and held-out case clusters together."""
    rng = random.Random(seed)
    seeds = sorted(next(iter(lightjev_by_seed.values())))
    workflows = []
    point_diffs = []
    for wf in WORKFLOWS:
        by_seed = lightjev_by_seed[wf]
        cases = sorted(by_seed[seeds[0]])
        if any(sorted(by_seed[s]) != cases for s in seeds) or sorted(laya[wf]) != cases:
            raise ValueError(f"unpaired cases or seeds for {wf}")
        point_diffs.append(statistics.fmean(
            statistics.fmean(by_seed[s][case] for s in seeds) - laya[wf][case]
            for case in cases))
        workflows.append((wf, cases))
    draws = []
    for _ in range(n_boot):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        wf_diffs = []
        for wf, cases in workflows:
            sampled = []
            for _ in cases:
                case = cases[rng.randrange(len(cases))]
                model_accuracy = statistics.fmean(lightjev_by_seed[wf][s][case] for s in sampled_seeds)
                sampled.append(model_accuracy - laya[wf][case])
            wf_diffs.append(statistics.fmean(sampled))
        draws.append(statistics.fmean(wf_diffs))
    draws.sort()
    return {"difference": statistics.fmean(point_diffs),
            "ci95_low": draws[int(0.025 * (n_boot - 1))],
            "ci95_high": draws[int(0.975 * (n_boot - 1))],
            "bootstrap_probability_lightjev_better": sum(value > 0 for value in draws) / n_boot,
            "resamples": n_boot,
            "unit": "training-seed and case-cluster bootstrap, stratified by workflow; Laya seed uncertainty not sampled"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, required=True,
                   help="workflow-generalization directory containing artifacts/ and results/")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--lightjev-subdir", default="lightjev_xworkflow",
                   help="results/artifacts subdirectory containing LightJev fold outputs")
    p.add_argument("--independent-baseline-subdir", default=None,
                   help="optional independent-candidate LightJev arm for a paired architecture contrast")
    p.add_argument("--seeds", default="31,47",
                   help="comma-separated LightJev seeds to aggregate (default: 31,47)")
    args = p.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer seeds")
    base = args.experiment_root
    independent_seeds = tuple(seed for seed in seeds if args.independent_baseline_subdir and all(
        (base / "artifacts" / "results" / args.independent_baseline_subdir / wf / f"seed{seed}" / "calibrated_predictions.jsonl").is_file()
        for wf in WORKFLOWS))
    if args.independent_baseline_subdir and not independent_seeds:
        raise ValueError("no selected seed has an independent-candidate baseline in every workflow")
    all_metrics = {}
    per_workflow_metrics = {}
    metric_keys = ("accuracy", "soft_accuracy_all_types", "kl_from_gold_all_types", "brier_all_types", "target_cross_entropy_all_types", "total_variation_all_types", "ece_15_bins")
    lightjev_case_acc = {wf: {} for wf in WORKFLOWS}
    independent_case_acc = {wf: {} for wf in WORKFLOWS}
    laya_case_acc = {arm: {wf: {} for wf in WORKFLOWS} for arm in ARMS}
    for wf in WORKFLOWS:
        fold = {}
        seed_rows = []
        independent_fold_metrics = []
        for seed in seeds:
            result_dir = base / "artifacts" / "results" / args.lightjev_subdir / wf / f"seed{seed}"
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
        if args.independent_baseline_subdir:
            by_seed = []
            for seed in independent_seeds:
                baseline_dir = base / "artifacts" / "results" / args.independent_baseline_subdir / wf / f"seed{seed}"
                rows = read_jsonl(baseline_dir / "calibrated_predictions.jsonl")
                if len(rows) != 1500:
                    raise ValueError(f"expected 1500 independent baseline rows: {baseline_dir}")
                by_seed.append(accuracy_by_case(rows))
                report = json.loads((baseline_dir / "calibrated_report.json").read_text())
                independent_fold_metrics.append(report["metrics"])
            if any(sorted(rows) != sorted(by_seed[0]) for rows in by_seed[1:]):
                raise ValueError(f"unpaired independent baseline cases for {wf}")
            independent_case_acc[wf] = {
                case: statistics.fmean([seed_rows[case] for seed_rows in by_seed])
                for case in by_seed[0]
            }
            fold["independent_candidate_mean_seeds_calibrated"] = {
                key: statistics.fmean(metrics[key] for metrics in independent_fold_metrics)
                for key in metric_keys
            }
        fold["lightjev_mean_seeds_calibrated"] = {
            key: statistics.fmean(fold[f"lightjev_seed{s}_calibrated"][key] for s in seeds)
            for key in metric_keys
        }
        per_workflow_metrics[wf] = fold
    # Seed predictions are repeated measurements for the same cases. Accuracy
    # and proper scores are arithmetic means, not probability ensembles.
    summary = {}
    for arm in ARMS:
        summary[arm] = {key: statistics.fmean(per_workflow_metrics[w][arm][key] for w in WORKFLOWS)
                        for key in metric_keys}
    lightjev_acc = statistics.fmean(per_workflow_metrics[w]["lightjev_mean_seeds_calibrated"]["accuracy"] for w in WORKFLOWS)
    independent_lightjev_acc = (statistics.fmean(per_workflow_metrics[w]["independent_candidate_mean_seeds_calibrated"]["accuracy"]
                                                   for w in WORKFLOWS)
                                if args.independent_baseline_subdir else None)
    independent_metrics = ({key: statistics.fmean(per_workflow_metrics[w]["independent_candidate_mean_seeds_calibrated"][key]
                                                   for w in WORKFLOWS)
                            for key in metric_keys} if args.independent_baseline_subdir else None)
    lightjev_metrics = {key: statistics.fmean(per_workflow_metrics[wf][f"lightjev_seed{seed}_calibrated"][key]
                                                    for wf in WORKFLOWS for seed in seeds)
                        for key in metric_keys}
    lightjev_metrics.update(cases_per_seed=1200, decisions_per_seed=6000,
                            cases_scored_once=1200, seed_repetitions=len(seeds))
    seed_macro_accuracy = {
        str(seed): statistics.fmean(
            per_workflow_metrics[wf][f"lightjev_seed{seed}_calibrated"]["accuracy"]
            for wf in WORKFLOWS)
        for seed in seeds
    }
    comparisons = {}
    for arm in ARMS:
        means = {wf: {case: statistics.fmean([lightjev_case_acc[wf][seed][case] for seed in seeds])
                      for case in lightjev_case_acc[wf][seeds[0]]} for wf in WORKFLOWS}
        comparisons[f"lightjev_vs_{arm}"] = bootstrap_diff(means, laya_case_acc[arm], args.bootstrap, 20260925, len(seeds))
    seed_case_comparisons = {
        f"lightjev_vs_{arm}": bootstrap_seed_and_case_diff(
            lightjev_case_acc, laya_case_acc[arm], args.bootstrap, 20260928)
        for arm in ARMS
    }
    if args.independent_baseline_subdir:
        shared_means = {wf: {case: statistics.fmean([lightjev_case_acc[wf][seed][case] for seed in independent_seeds])
                             for case in lightjev_case_acc[wf][independent_seeds[0]]} for wf in WORKFLOWS}
        comparisons["shared_context_vs_independent_candidate"] = bootstrap_diff(
            shared_means, independent_case_acc, args.bootstrap, 20260927, len(independent_seeds))
    all_metrics = {
        "protocol": f"four workflow-held-out folds; LightJev seeds {list(seeds)} per fold; calibration on source workflows only",
        "note": f"Exploratory. No official public test split used. Result arm: {args.lightjev_subdir}; see its report for architecture and context-window caveats.",
        "decisions_per_seed": 6000,
        "lightjev_seeds": list(seeds),
        "independent_baseline_matched_seeds": list(independent_seeds),
        "lightjev_mean_seed_calibrated_metrics": lightjev_metrics,
        "lightjev_mean_seed_accuracy": lightjev_acc,
        "lightjev_seed_macro_accuracy": seed_macro_accuracy,
        "independent_candidate_lightjev_mean_seed_accuracy": independent_lightjev_acc,
        "independent_candidate_lightjev_mean_seed_metrics": independent_metrics,
        "laya_accuracy_by_arm": summary,
        "paired_case_bootstrap": comparisons,
        "seed_and_case_bootstrap": seed_case_comparisons,
        "per_workflow": per_workflow_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False) + "\n")
    lines = [f"# LightJev versus Laya: workflow-held-out comparison ({args.lightjev_subdir})", "",
             f"Four folds, {len(seeds)} LightJev seeds per fold. Every held-out case is scored once per model/seed; LightJev mean metrics are the arithmetic mean of runs, not a probability ensemble.", "",
             f"Exploratory: prior work inspected these workflow families and public-test results. The official public test split was not used here. LightJev inputs are run at 640 tokens, above its released 256-token training window. Case-bootstrap intervals condition on the {len(seeds)} selected training seeds and do not estimate seed uncertainty.", "",
             "| Model | Accuracy | Target CE ↓ | Brier ↓ | ECE ↓ |", "|---|---:|---:|---:|---:|",
             f"| LightJev mean across seeds | {lightjev_metrics['accuracy']:.3f} | {lightjev_metrics['target_cross_entropy_all_types']:.3f} | {lightjev_metrics['brier_all_types']:.3f} | {lightjev_metrics['ece_15_bins']:.3f} |"]
    if independent_lightjev_acc is not None:
        lines.append(f"| Independent-candidate LightJev | {independent_lightjev_acc:.3f} | {independent_metrics['target_cross_entropy_all_types']:.3f} | {independent_metrics['brier_all_types']:.3f} | {independent_metrics['ece_15_bins']:.3f} |")
    for arm in ARMS:
        lines.append(f"| {arm} | {summary[arm]['accuracy']:.3f} | {summary[arm]['target_cross_entropy_all_types']:.3f} | {summary[arm]['brier_all_types']:.3f} | {summary[arm]['ece_15_bins']:.3f} |")
    lines += ["", "## Paired case-cluster bootstrap", "",
              "Intervals resample cases within each held-out workflow (10,000 draws by default); the five questions on one state stay clustered.", "",
              "| Comparison | Difference | 95% CI | P(LightJev better) |", "|---|---:|---:|---:|"]
    for name, result in comparisons.items():
        lines.append(f"| {name} | {result['difference']:+.3f} | [{result['ci95_low']:+.3f}, {result['ci95_high']:+.3f}] | {result['bootstrap_probability_lightjev_better']:.3f} |")
    lines += ["", "## Seed and case bootstrap", "",
              "This resamples both observed LightJev training seeds and held-out case clusters. Laya's own seed uncertainty is not sampled.", "",
              "| Comparison | Difference | 95% CI | P(LightJev better) |", "|---|---:|---:|---:|"]
    for name, result in seed_case_comparisons.items():
        lines.append(f"| {name} | {result['difference']:+.3f} | [{result['ci95_low']:+.3f}, {result['ci95_high']:+.3f}] | {result['bootstrap_probability_lightjev_better']:.3f} |")
    lines += ["", "## Accuracy by held-out workflow", "", "| Workflow | LightJev (seed mean) | Laya RLCD | Laya soft-CE |", "|---|---:|---:|---:|"]
    for wf in WORKFLOWS:
        row = per_workflow_metrics[wf]
        lines.append(f"| {wf} | {row['lightjev_mean_seeds_calibrated']['accuracy']:.3f} | {row['laya_rlcd']['accuracy']:.3f} | {row['laya_soft_ce']['accuracy']:.3f} |")
    lines += ["", "## Accuracy by training seed", "", "| Seed | Macro accuracy |", "|---:|---:|"]
    for seed, accuracy in seed_macro_accuracy.items():
        lines.append(f"| {seed} | {accuracy:.3f} |")
    lines += ["", "Full distribution metrics and per-seed reports are in `comparison.json`.", ""]
    (args.output_dir / "comparison.md").write_text("\n".join(lines))
    print(json.dumps({"lightjev_accuracy": lightjev_metrics["accuracy"], "laya": summary,
                      "paired_case_bootstrap": comparisons}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
