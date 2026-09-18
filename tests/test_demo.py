import json
from lightjev.demo import run_demo


def test_offline_training_checkpoint_prediction_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    report = run_demo(tmp_path / 'demo', steps=2)
    assert report['test_metrics']['questions'] == 12
    assert report['pretrained'] is False
    predictions = json.loads((tmp_path / 'demo' / 'predictions.json').read_text())
    assert len(predictions) == 12
    for pred in predictions:
        assert abs(sum(pred['probabilities']) - 1) < 1e-5
        assert pred['selected'] in pred['candidates']
