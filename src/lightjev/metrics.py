"""Probability metrics; calibration is measured, never inferred from softmax."""
import math


def _vector(values):
    p = [float(v) for v in values]
    if len(p) < 2 or any(not math.isfinite(v) or v < 0 or v > 1 for v in p):
        raise ValueError("Probabilities must be finite values in [0,1]")
    if abs(sum(p) - 1) > 1e-5:
        raise ValueError("Probabilities must sum to one")
    return p


def temperature_scale(probabilities, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    p = _vector(probabilities)
    z = [math.log(v) / temperature if v else -math.inf for v in p]
    shift = max(z)
    e = [math.exp(v - shift) for v in z]
    return [v / sum(e) for v in e]


def aligned(records, predictions):
    expected = {r["id"]: r for r in records}
    actual = {p["id"]: p for p in predictions}
    if len(expected) != len(records) or len(actual) != len(predictions):
        raise ValueError("Duplicate record or prediction IDs")
    if set(expected) != set(actual) or not records:
        raise ValueError("Prediction IDs must match the nonempty evaluation set exactly")
    pairs = []
    for row in records:
        pred = actual[row["id"]]
        if "target" not in row:
            raise ValueError("Evaluation requires targets")
        if pred["candidates"] != row["candidates"]:
            raise ValueError("Prediction candidate order does not match the target")
        p, q = _vector(pred["probabilities"]), _vector(row["target"])
        if len(p) != len(q) or len(p) != len(row["candidates"]):
            raise ValueError("Candidate and distribution lengths differ")
        pairs.append((p, q))
    return pairs


def evaluate(records, predictions):
    pairs = aligned(records, predictions)
    n = len(pairs)
    ce = sum(-sum(y * math.log(max(x, 1e-12)) for x, y in zip(p, q)) for p, q in pairs) / n
    l2 = sum(sum((x - y) ** 2 for x, y in zip(p, q)) for p, q in pairs) / n
    hard = all(sum(v == 1 for v in q) == 1 for _, q in pairs)
    report = {"questions": n, "target_cross_entropy": ce, "target_squared_l2": l2,
              "target_kind": "one_hot" if hard else "distribution_or_mixed",
              "log_floor": 1e-12, "calibration_provenance": "not_verified",
              "calibration_guaranteed": False}
    if hard:
        scored = [(max(p), int(max(range(len(p)), key=p.__getitem__) == q.index(1))) for p, q in pairs]
        report["accuracy"] = sum(correct for _, correct in scored) / n
        report["vector_brier"] = l2
        bins = []
        for b in range(10):
            bucket = [(c, ok) for c, ok in scored if min(int(c * 10), 9) == b]
            if bucket:
                bins.append({"lower": b / 10, "upper": (b + 1) / 10, "count": len(bucket),
                             "mean_probability": sum(c for c, _ in bucket) / len(bucket),
                             "accuracy": sum(ok for _, ok in bucket) / len(bucket)})
        report["top_label_ece_10_bins"] = sum(b["count"] * abs(b["mean_probability"] - b["accuracy"]) for b in bins) / n
        report["reliability_bins"] = bins
        report["selective_risk"] = []
        for threshold in (0, .5, .7, .9, .95):
            selected = [ok for p, ok in scored if p >= threshold]
            report["selective_risk"].append({"threshold": threshold, "coverage": len(selected) / n,
                                            "error_rate": 1 - sum(selected) / len(selected) if selected else None})
    return report


def fit_temperature(records, predictions):
    """Grid-fit on a dedicated calibration split; do not use the test set here."""
    pairs = aligned(records, predictions)
    if any(pred.get("temperature", 1.0) != 1.0 for pred in predictions):
        raise ValueError("Temperature fitting requires predictions generated at temperature 1")
    grid = [math.exp(i / 10) for i in range(-23, 24)]
    def loss(t):
        return sum(-sum(y * math.log(max(x, 1e-12)) for x, y in zip(temperature_scale(p, t), q)) for p, q in pairs) / len(pairs)
    t = min(grid, key=loss)
    return {"temperature": t, "calibration_questions": len(pairs), "before_ce": loss(1),
            "after_ce": loss(t), "method": "bounded_log_grid", "grid_min": grid[0],
            "grid_max": grid[-1], "calibration_guaranteed": False,
            "note": "Validate on a separate test split. Zero probability support is preserved."}
