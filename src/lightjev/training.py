"""Original minimal full-finetuning loop for candidate decisions."""
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import torch


def decision_loss(logits, targets, loss="ce"):
    if loss not in {"ce", "brier"}:
        raise ValueError("loss must be ce or brier")
    if len(logits) != len(targets) or not logits:
        raise ValueError("matching nonempty logits and targets required")
    values = []
    for scores, target in zip(logits, targets):
        target = torch.as_tensor(target, device=scores.device, dtype=scores.dtype)
        if target.shape != scores.shape:
            raise ValueError("target and logit shapes differ")
        values.append(-(target * scores.log_softmax(-1)).sum() if loss == "ce"
                      else ((scores.softmax(-1) - target) ** 2).sum())
    return torch.stack(values).mean()


def check_disjoint(train_records, dev_records):
    for key in ("id", "group_id"):
        overlap = {row[key] for row in train_records} & {row[key] for row in dev_records}
        if overlap:
            raise ValueError(f"train/dev {key} overlap: {sorted(overlap)[:5]}")


def _batch(records, tokenizer, max_length, device):
    from .model import encode_records
    return {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in encode_records(records, tokenizer, max_length).items()}


def _evaluate(model, records, tokenizer, max_length, device, batch_size):
    totals = {"ce": 0., "brier": 0.}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            logits = model(**_batch(rows, tokenizer, max_length, device))
            for name in totals:
                totals[name] += decision_loss(logits, [r["target"] for r in rows], name).item() * len(rows)
    return {name: value / len(records) for name, value in totals.items()}


def train(train_path, dev_path, output_dir, model_name="Qwen/Qwen3-0.6B",
          steps=20, batch_size=2, lr=1e-5, loss="ce", seed=17,
          max_length=512, device="cpu", revision=None):
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer
    import transformers
    import safetensors
    from .model import DecisionModel
    from .schema import load_records
    if steps < 1 or batch_size < 1 or max_length < 1:
        raise ValueError("steps, batch_size and max_length must be positive")
    if not math.isfinite(lr) or lr <= 0 or loss not in {"ce", "brier"}:
        raise ValueError("invalid learning rate or loss")
    train_rows, dev_rows = load_records(train_path), load_records(dev_path)
    if not train_rows or not dev_rows or any(r.get("target") is None for r in train_rows + dev_rows):
        raise ValueError("nonempty labeled train and dev required")
    check_disjoint(train_rows, dev_rows)
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("output directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = random.Random(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    model = DecisionModel(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    config = {"format_version": 1, "base_model": model_name, "requested_revision": revision,
              "resolved_revision": getattr(backbone.config, "_commit_hash", None),
              "steps": steps, "batch_size": batch_size, "lr": lr, "loss": loss,
              "seed": seed, "max_length": max_length, "calibration": "none",
              "determinism": "seeded; unsupported deterministic operations warn",
              "data_sha256": {n: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for n, p in (("train", train_path), ("dev", dev_path))},
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": transformers.__version__, "safetensors": safetensors.__version__}}
    backbone.config.save_pretrained(destination / "backbone")
    tokenizer.save_pretrained(destination / "tokenizer")
    best_value, history = float("inf"), []
    for step in range(1, steps + 1):
        model.train()
        rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        logits = model(**_batch(rows, tokenizer, max_length, device))
        objective = decision_loss(logits, [r["target"] for r in rows], loss)
        if not torch.isfinite(objective):
            raise ValueError("nonfinite training loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        metrics = _evaluate(model, dev_rows, tokenizer, max_length, device, batch_size)
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("nonfinite development metrics")
        history.append({"step": step, "train_loss": objective.item(), "dev": metrics})
        if metrics[loss] < best_value:
            best_value = metrics[loss]
            config.update(best_step=step, best_dev=metrics)
            save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()},
                      str(destination / "model.safetensors"))
    config["history"] = history
    (destination / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    return config
