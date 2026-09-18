import pytest
from lightjev.metrics import evaluate, fit_temperature, temperature_scale


def fixture():
    rows = [{"id": "a", "candidates": ["no", "yes"], "target": [0, 1]},
            {"id": "b", "candidates": ["no", "yes"], "target": [1, 0]}]
    predictions = [{"id": "a", "candidates": ["no", "yes"], "probabilities": [.2, .8]},
                   {"id": "b", "candidates": ["no", "yes"], "probabilities": [.4, .6]}]
    return rows, predictions


def test_hand_computed_metrics_and_empty_coverage():
    report = evaluate(*fixture())
    assert report["accuracy"] == .5
    assert report["vector_brier"] == pytest.approx(.4)
    assert report["top_label_ece_10_bins"] == pytest.approx(.4)
    assert report["selective_risk"][-1]["error_rate"] is None


def test_alignment_and_soft_targets():
    rows, predictions = fixture()
    assert evaluate(rows, predictions[::-1])["accuracy"] == .5
    predictions[0]["candidates"] = ["yes", "no"]
    with pytest.raises(ValueError):
        evaluate(rows, predictions)
    rows, predictions = fixture()
    rows[0]["target"] = [.2, .8]
    assert "accuracy" not in evaluate(rows, predictions)


def test_temperature_fit_and_zero_support():
    result = fit_temperature(*fixture())
    assert result["after_ce"] <= result["before_ce"]
    assert temperature_scale([0, 1], 2) == [0, 1]
    with pytest.raises(ValueError):
        temperature_scale([.5, .5], 0)


def test_duplicate_predictions_rejected():
    rows, pred = fixture()
    with pytest.raises(ValueError):
        evaluate(rows, pred + pred[:1])


def test_temperature_fit_rejects_already_scaled_predictions():
    rows, pred = fixture()
    pred[0]["temperature"] = 2.0
    with pytest.raises(ValueError, match="temperature 1"):
        fit_temperature(rows, pred)
