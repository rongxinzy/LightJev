"""Per-family and paired metrics; rows within a contrast group are correlated."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    result = {"rows": len(rows), "families": {}}
    for family, members in sorted(families.items()):
        groups = defaultdict(dict)
        bins = [[] for _ in range(15)]
        for row in members:
            groups[row["group_id"]][row["variant"]] = row
            confidence = max(row["probabilities"])
            bins[min(14, int(confidence * 15))].append((confidence, row["prediction"] == row["label_index"]))
        pair = {}
        for variant in ("policy", "fact", "irrelevant"):
            complete = [g for g in groups.values() if "base" in g and variant in g]
            if complete:
                pair[f"{variant}_both_correct"] = mean(
                    all(g[v]["prediction"] == g[v]["label_index"] for v in ("base", variant)) for g in complete)
                pair[f"{variant}_semantic_consistency"] = mean(
                    g["base"]["semantic_values"][g["base"]["prediction"]] ==
                    g[variant]["semantic_values"][g[variant]["prediction"]] for g in complete)
        # Stable log-softmax from logits, avoiding underflow-biased NLL.
        nlls = []
        for row in members:
            z, label = row["logits"], row["label_index"]
            top = max(z)
            nlls.append(top + math.log(sum(math.exp(x - top) for x in z)) - z[label])
        values = {"rows": len(members), "groups": len(groups),
                  "accuracy": mean(r["prediction"] == r["label_index"] for r in members),
                  "nll": mean(nlls),
                  "brier": mean(sum((p - float(i == r["label_index"])) ** 2
                                    for i, p in enumerate(r["probabilities"])) for r in members),
                  "ece_15": sum(abs(sum(c - a for c, a in bucket)) for bucket in bins) / len(members),
                  "predicted_semantic_counts": dict(Counter(str(r["semantic_values"][r["prediction"]]) for r in members)),
                  "gold_semantic_counts": dict(Counter(str(r["semantic_values"][r["label_index"]]) for r in members)), **pair}
        result["families"][family] = values
    numeric = ("accuracy", "nll", "brier", "ece_15", "policy_both_correct", "fact_both_correct",
               "irrelevant_both_correct", "irrelevant_semantic_consistency")
    result["macro"] = {key: mean(v[key] for v in result["families"].values() if key in v)
                       for key in numeric if any(key in v for v in result["families"].values())}
    if "policy_both_correct" in result["macro"]:
        result["macro"]["counterfactual_both_correct"] = mean(
            result["macro"][key] for key in ("policy_both_correct", "fact_both_correct"))
    return result
