"""Development diagnostics only. The locked test is deliberately unavailable here."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from data import digest, load_rows, strict_encode
from metrics import summarize
from model_io import load_model, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("evaluation output already exists")
    manifest = json.loads((args.data / "manifest.json").read_text())
    model, tokenizer, _ = load_model(args.model, args.device)
    reports, predictions, diagnostic_predictions = {}, [], []
    for split in ("dev_seen", "dev_new"):
        path = args.data / f"{split}.jsonl"
        if digest(path) != manifest["splits"][split]["sha256"]:
            raise ValueError("data does not match frozen manifest")
        rows = load_rows(path)
        if any(r["split"] != split for r in rows):
            raise ValueError("unexpected split contents")
        normal = predict(model, rows, tokenizer.pad_token_id, args.device, args.batch_size)
        reports[split] = {"normal": summarize(normal)}
        predictions.extend(normal)
        if args.diagnostics:
            for mode in ("reverse_options", "no_state", "no_criteria"):
                changed = []
                for row in rows:
                    record = copy.deepcopy(row)
                    order = None
                    if mode == "reverse_options":
                        order = list(reversed(range(len(row["target"]))))
                    elif mode == "no_state":
                        record["state"] = {}
                    else:
                        criteria = record["question"]["criteria"]
                        record["question"]["criteria"] = (["unspecified"] * len(criteria) if isinstance(criteria, list)
                                                               else {key: "unspecified" for key in criteria})
                    changed.append(strict_encode(record, tokenizer, order=order))
                results = predict(model, changed, tokenizer.pad_token_id, args.device, args.batch_size)
                diagnostic_predictions.extend({**row, "diagnostic": mode} for row in results)
                report = summarize(results)
                report["same_prediction_as_normal"] = sum(a["prediction"] == b["prediction"]
                                                           for a, b in zip(normal, results)) / len(normal)
                reports[split][mode] = report
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    if args.diagnostics:
        (args.output / "diagnostic_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in diagnostic_predictions))
    (args.output / "manifest.json").write_text(json.dumps({"model_sha256": digest(args.model / "model.safetensors"),
        "data_manifest_sha256": digest(args.data / "manifest.json"), "locked_test_read": False,
        "corrupted_input_controls": ["no_state", "no_criteria"] if args.diagnostics else [],
        "semantics_preserving_controls": ["reverse_options"] if args.diagnostics else [],
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in ("evaluate.py", "model_io.py", "data.py", "metrics.py")}}, indent=2) + "\n")
    print(json.dumps({k: v["normal"]["macro"] for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
