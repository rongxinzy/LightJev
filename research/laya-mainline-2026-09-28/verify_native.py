"""Check the exported checkpoint through the native Laya SDK on development examples."""
import argparse
import json
from collections import Counter
from pathlib import Path

import laya

from data import load_rows
from model_io import load_model, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    counts, rows = Counter(), []
    for row in load_rows(args.data / "dev_seen.jsonl"):
        kind = row["question"]["type"]
        if counts[kind] < 3:
            counts[kind] += 1
            rows.append(row)
    assert set(counts) == {"choice", "noul", "score"}
    model, tok, _ = load_model(args.model, "cuda:0")
    reference = predict(model, rows, tok.pad_token_id, "cuda:0", batch_size=1)
    del model
    import torch
    torch.cuda.empty_cache()
    agent = laya.load(str(args.model), device="cuda:0", fast=False)
    results = []
    for row, expected in zip(rows, reference):
        answer = agent.predict(row["state"], {"probe": row["question"]})["answers"]["probe"]
        if row["question"]["type"] == "noul":
            actual = [1. - answer["noul"], answer["noul"]]
        else:
            actual = list(answer["probabilities"].values())
        delta = max(abs(a - b) for a, b in zip(actual, expected["probabilities"]))
        same = max(range(len(actual)), key=actual.__getitem__) == expected["prediction"]
        if delta > .01 or not same:
            raise RuntimeError(f"native SDK mismatch: {row['id']}, delta={delta}")
        results.append({"id": row["id"], "kind": row["question"]["type"],
                        "max_probability_delta": delta, "same_decision": same})
    result = {"status": "pass", "examples": results, "fast_path": False,
              "probability_tolerance": .01, "scope": "nine development examples; SDK rounds probabilities"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
