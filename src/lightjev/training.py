"""Original minimal full-finetuning loop for candidate decisions."""
import hashlib
import json
import math
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from .defaults import DEFAULT_MODEL, resolve_revision


def decision_loss(logits, targets, loss="ce"):
    if loss not in {"ce", "brier"}:
        raise ValueError("loss must be ce or brier")
    if len(logits) != len(targets) or not logits:
        raise ValueError("matching nonempty logits and targets required")
    values = []
    for scores, target in zip(logits, targets):
        scores = scores.float()
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


def _autocast(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def _evaluate(model, records, tokenizer, max_length, device, batch_size, precision="fp32"):
    totals = {"ce": 0., "brier": 0.}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            with _autocast(precision):
                logits = model(**_batch(rows, tokenizer, max_length, device))
            for name in totals:
                totals[name] += decision_loss(logits, [r["target"] for r in rows], name).item() * len(rows)
    return {name: value / len(records) for name, value in totals.items()}


def train(train_path, dev_path, output_dir, model_name=DEFAULT_MODEL,
          steps=20, batch_size=2, lr=1e-5, loss="ce", seed=17,
          max_length=512, device="cpu", revision=None, head_steps=0, head_lr=None,
          head_warmup_lr=1e-3, eval_every=1, grad_accum_steps=1, precision="fp32",
          gradient_checkpointing=False, selection_metric="ce"):
    started = time.monotonic()
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer
    import transformers
    import safetensors
    from .model import DecisionModel
    from .schema import load_records
    if steps < 1 or batch_size < 1 or max_length < 1 or eval_every < 1 or grad_accum_steps < 1 or head_steps < 0:
        raise ValueError("positive steps/batch/length/eval/accum and nonnegative head_steps required")
    head_lr = lr if head_lr is None else head_lr
    if any(not math.isfinite(v) or v <= 0 for v in (head_lr, head_warmup_lr)):
        raise ValueError("head learning rates must be finite and positive")
    if precision not in {"fp32", "bf16"} or selection_metric not in {"ce", "brier"}:
        raise ValueError("invalid precision or selection metric")
    if precision == "bf16" and (torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a supported CUDA device")
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
    effective_revision = resolve_revision(model_name, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=effective_revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(model_name, revision=effective_revision, trust_remote_code=False)
    # Transformers versions may inherit BF16 from the checkpoint configuration.
    # Keep master parameters/Adam moments in FP32; autocast controls compute only.
    backbone.float()
    # Decision training consumes hidden states, not an autoregressive KV cache.
    backbone.config.use_cache = False
    model = DecisionModel(backbone).to(device)
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    config = {"format_version": 1, "base_model": model_name, "requested_revision": revision, "effective_revision": effective_revision,
              "resolved_revision": getattr(backbone.config, "_commit_hash", None),
              "steps": steps, "head_steps": head_steps, "head_lr": head_lr, "head_warmup_lr": head_warmup_lr,
              "eval_every": eval_every, "grad_accum_steps": grad_accum_steps,
              "effective_batch_size": batch_size * grad_accum_steps, "precision": precision,
              "parameter_dtype": "float32", "optimizer_state_dtype": "float32",
              "gradient_checkpointing": gradient_checkpointing, "selection_metric": selection_metric,
              "status": "initializing", "samples_seen": 0,
              "parameter_counts": {"total": sum(p.numel() for p in model.parameters()),
                                   "backbone": sum(p.numel() for p in backbone.parameters()),
                                   "head": sum(p.numel() for p in model.head.parameters())},
              "batch_size": batch_size, "lr": lr, "loss": loss,
              "seed": seed, "max_length": max_length, "calibration": "none",
              "determinism": "seeded; unsupported deterministic operations warn",
              "data_sha256": {n: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for n, p in (("train", train_path), ("dev", dev_path))},
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "transformers": transformers.__version__, "safetensors": safetensors.__version__}}
    backbone.config.save_pretrained(destination / "backbone")
    tokenizer.save_pretrained(destination / "tokenizer")
    best_value, history = float("inf"), []

    def write_manifest():
        config["elapsed_seconds"] = time.monotonic() - started
        config["peak_cuda_memory_bytes"] = (torch.cuda.max_memory_allocated(device)
                                            if torch.device(device).type == "cuda" else None)
        temporary = destination / "manifest.json.tmp"
        temporary.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination / "manifest.json")

    def evaluate_and_save(step):
        nonlocal best_value
        metrics = _evaluate(model, dev_rows, tokenizer, max_length, device, batch_size, precision)
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("nonfinite development metrics")
        if metrics[selection_metric] < best_value:
            best_value = metrics[selection_metric]
            temporary = destination / "model.safetensors.tmp"
            save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(temporary))
            temporary.replace(destination / "model.safetensors")
            config.update(best_step=step, best_dev=metrics)
        return metrics

    write_manifest()
    with (destination / "train_log.jsonl").open("a", encoding="utf-8") as log:
        def record(entry):
            history.append(entry)
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
            config["last_step"] = entry["step"]
            write_manifest()

        initial_metrics = evaluate_and_save(0)
        config.update(initial_dev=initial_metrics, status="training")
        record({"step": 0, "phase": "initial", "dev": initial_metrics, "samples_seen": 0})
        global_step = 0
        for phase, phase_steps in (("head", head_steps), ("full", steps)):
            if not phase_steps:
                continue
            for parameter in backbone.parameters():
                parameter.requires_grad_(phase == "full")
            groups = [{"params": model.head.parameters(), "lr": head_warmup_lr if phase == "head" else head_lr}]
            if phase == "full":
                groups.append({"params": backbone.parameters(), "lr": lr})
            optimizer = torch.optim.AdamW(groups)
            for phase_step in range(1, phase_steps + 1):
                global_step += 1
                step_started = time.monotonic()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                objective_value = 0.
                for _ in range(grad_accum_steps):
                    rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(batch_size)]
                    with _autocast(precision):
                        logits = model(**_batch(rows, tokenizer, max_length, device))
                        objective = decision_loss(logits, [r["target"] for r in rows], loss)
                    if not torch.isfinite(objective):
                        raise ValueError("nonfinite training loss")
                    (objective / grad_accum_steps).backward()
                    objective_value += objective.item() / grad_accum_steps
                    config["samples_seen"] += len(rows)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite gradient norm")
                optimizer.step()
                entry = {"step": global_step, "phase": phase, "phase_step": phase_step,
                         "train_loss": objective_value, "samples_seen": config["samples_seen"],
                         "gradient_norm": norm.item()}
                if global_step % eval_every == 0 or phase_step == phase_steps:
                    entry["dev"] = evaluate_and_save(global_step)
                entry["seconds"] = time.monotonic() - step_started
                record(entry)
            del optimizer
    config.update(status="complete", history=history)
    write_manifest()
    return config
