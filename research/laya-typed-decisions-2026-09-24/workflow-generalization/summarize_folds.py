#!/usr/bin/env python3
"""Aggregate complete per-fold predictions into workflow-macro test metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from evaluate_benchmark import score_predictions

WORKFLOWS = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]
LABELS = ["laya_base_calibrated", "laya_base_raw", "rlcd_calibrated", "rlcd_raw",
          "soft_ce_calibrated", "soft_ce_raw", "lightjev_zero_shot"]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    rows_by_model={}
    fold_reports={}
    for label in LABELS:
        rows=[]
        fold_reports[label]={}
        for workflow in WORKFLOWS:
            path=args.results_dir/workflow/f"{label}_predictions.jsonl"
            report=args.results_dir/workflow/f"{label}_report.json"
            if not path.is_file() or not report.is_file():
                raise FileNotFoundError(f"missing completed result for {workflow}/{label}")
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
            fold_reports[label][workflow]=json.loads(report.read_text(encoding="utf-8"))["metrics"]
        if len({(r["case_id"], r["qid"]) for r in rows}) != len(rows):
            raise ValueError(f"duplicate case-question rows for {label}")
        rows_by_model[label]=score_predictions(rows, [])
    result={
        "protocol":"4-fold leave-one-workflow-out over the pinned public train split",
        "note":"Exploratory: prior aggregate results on these workflow families were inspected. The official public test split was not used here.",
        "cases_per_model":1200,
        "decisions_per_model":6000,
        "fold_reports":fold_reports,
        "workflow_macro_metrics":rows_by_model,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Workflow-held-out generalization results","",
           "Four-fold leave-one-workflow-out evaluation on the pinned public train split. Each row below tests all 300 cases from one workflow excluded from training, development, and calibration.","",
           "Exploratory result: this workflow family and dataset have been inspected in prior runs. The official public test split was not used in this experiment.","",
           "Accuracy and probability metrics are weighted by decisions across all 1,200 held-out cases (6,000 decisions); per-workflow rows show the held-out fold scores.","",
           "| Model | Accuracy | Soft acc. | KL | Brier | NLL | ECE |","|---|---:|---:|---:|---:|---:|---:|"]
    for label, metrics in rows_by_model.items():
        lines.append(f"| {label} | {metrics['accuracy']:.3f} | {metrics['soft_accuracy_all_types']:.3f} | {metrics['kl_from_gold_all_types']:.3f} | {metrics['brier_all_types']:.3f} | {metrics['target_cross_entropy_all_types']:.3f} | {metrics['ece_15_bins']:.3f} |")
    lines += ["","## Accuracy by held-out workflow","","| Model | " + " | ".join(WORKFLOWS) + " |","|---|" + "---:|"*len(WORKFLOWS)]
    for label, metrics in rows_by_model.items():
        bywf=metrics['by_workflow_accuracy']
        lines.append("| "+label+" | "+" | ".join(f"{bywf[w]:.3f}" for w in WORKFLOWS)+" |")
    lines += ["","## Accuracy and distribution metrics by question type","","Full per-type metrics are in `summary.json` under `workflow_macro_metrics.*.by_type_metrics`.",""]
    args.output.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"models":list(rows_by_model),"accuracy_by_model":{k:v['accuracy'] for k,v in rows_by_model.items()}},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
