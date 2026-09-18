import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("evaluate_release", Path(__file__).resolve().parents[1] / "scripts/evaluate_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def setup_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split in release.SPLITS:
        rows = [dict(id=f"{split}-{i}", group_id=f"{split}-{i}", state="s", question="q", kind="choice",
                     candidates=["a", "b"], target=[1, 0] if i == 0 else [.3, .7],
                     provenance=dict(family_id="alpha" if i == 0 else "beta", target_kind="deterministic_truth" if i == 0 else "programmatic_conditional_distribution")) for i in range(2)]
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return data


def test_once_loaded_batches_subsets_and_calibration_only(tmp_path, monkeypatch):
    data = setup_data(tmp_path)
    calls = []
    class Model:
        def float(self):
            return self
        def eval(self):
            return self
        def __call__(self, rows):
            calls.append(len(rows))
            return [torch.tensor([1., 0.]) for _ in rows]
    loads = []
    def load(*args, **kwargs):
        loads.append(1)
        return Model(), None, {"max_length": 100}
    monkeypatch.setattr(release, "load_checkpoint", load)
    monkeypatch.setattr(release, "_batch", lambda rows, *args: {"rows": rows})
    real_fit = release.fit_temperature
    def fit(rows, predictions):
        assert all(r["id"].startswith("calibration-") for r in rows)
        return real_fit(rows, predictions)
    monkeypatch.setattr(release, "fit_temperature", fit)
    report = release.run("unused", data, tmp_path / "out", batch_size=2)
    assert len(loads) == 1
    assert calls == [2, 2, 2, 2]
    assert report["splits"]["test"]["raw"]["hard"]["questions"] == 1
    assert report["splits"]["test"]["raw"]["soft"]["questions"] == 1
    assert set(report["splits"]["test"]["raw"]["families"]) == {"alpha", "beta"}
    assert len(list((tmp_path / "out").glob("*.predictions.json"))) == 8
    assert report["temperature_fit_split"] == "calibration"
    with pytest.raises(ValueError, match="new or empty"):
        release.run("unused", data, tmp_path / "out")


def test_reject_cross_split_leakage_before_loading(tmp_path, monkeypatch):
    data = setup_data(tmp_path)
    path = data / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["group_id"] = "calibration-0"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(release, "load_checkpoint", lambda *a, **k: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError, match="group_id overlap"):
        release.run("unused", data, tmp_path / "out")


def test_real_offline_checkpoint_roundtrip(tmp_path, monkeypatch):
    from lightjev.demo import run_demo
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    run_demo(tmp_path / "demo", steps=1)
    data = setup_data(tmp_path)
    report = release.run(tmp_path / "demo/checkpoint", data, tmp_path / "evaluation", batch_size=2)
    assert report["dtype"] == "float32"
    assert report["splits"]["ood"]["role"] == "heldout"
    assert report["splits"]["test"]["raw"]["all"]["questions"] == 2
    saved = json.loads((tmp_path / "evaluation/test.calibrated.predictions.json").read_text())
    assert all(sum(p["probabilities"]) == pytest.approx(1) for p in saved)
