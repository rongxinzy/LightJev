"""Apply the frozen continuation gate; never runs inference or opens the locked test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(base, selected):
    bm = {s: base[s]["normal"]["macro"] for s in ("dev_seen", "dev_new")}
    sm = {s: selected[s]["normal"]["macro"] for s in ("dev_seen", "dev_new")}
    gain = sm["dev_new"]["counterfactual_both_correct"] - bm["dev_new"]["counterfactual_both_correct"]
    seen_delta = sm["dev_seen"]["accuracy"] - bm["dev_seen"]["accuracy"]
    stable_delta = min(sm[s]["irrelevant_semantic_consistency"] - bm[s]["irrelevant_semantic_consistency"]
                       for s in ("dev_seen", "dev_new"))
    checks = {"new_family_pair_gain_at_least_5pp": gain >= .05 - 1e-12,
              "seen_accuracy_drop_at_most_2pp": seen_delta >= -.02 - 1e-12,
              "irrelevant_consistency_drop_at_most_2pp_each_split": stable_delta >= -.02 - 1e-12}
    return {"base": bm, "selected": sm, "delta": {"new_family_pair_both_correct": gain,
            "seen_family_accuracy": seen_delta, "worst_split_irrelevant_consistency": stable_delta},
            "gate_checks": checks, "promising_for_second_seed": all(checks.values()),
            "interpretation": "single-seed synthetic mechanism pilot; development selection; no significance claim",
            "locked_test_evaluated": False, "automatic_additional_training": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads((args.artifacts / "results/base_diagnostics/report.json").read_text())
    selected = json.loads((args.artifacts / "results/selected_diagnostics/report.json").read_text())
    result = compare(base, selected)
    result["selection"] = json.loads((args.artifacts / "runs/ce_seed31/selection.json").read_text())
    result["completion"] = json.loads((args.artifacts / "runs/ce_seed31/complete.json").read_text())
    result["smoke"] = json.loads((args.artifacts / "runs/smoke/complete.json").read_text())
    if result["completion"]["status"] != "complete" or result["smoke"]["status"] != "complete":
        raise ValueError("cannot summarize an incomplete pilot")
    (args.artifacts / "pilot_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
