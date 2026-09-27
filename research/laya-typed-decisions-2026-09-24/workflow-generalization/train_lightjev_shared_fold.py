#!/usr/bin/env python3
"""Fine-tune LightJev with all candidates scored from one shared input."""
from __future__ import annotations

import argparse, hashlib, json, math, random, time
from pathlib import Path

import torch
from safetensors.torch import save_file

from lightjev.inference import load_checkpoint
from lightjev.training import check_disjoint, decision_loss
from lightjev.schema import load_records
from shared_context import encode_joint, infer_probabilities, shared_scores


def batch_to_device(rows, tokenizer, max_length, device, rng=None):
    permutations = []
    for row in rows:
        order = list(range(len(row["candidates"])))
        if rng is not None:
            rng.shuffle(order)
        permutations.append(order)
    batch = encode_joint(rows, tokenizer, max_length, permutations)
    tensors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    targets = [torch.tensor([r["target"][i] for i in order], dtype=torch.float32, device=device)
               for r, order in zip(rows, permutations)]
    return tensors, targets


def evaluate(model, tokenizer, rows, max_length, device, batch_size):
    # Select checkpoints using the same four-order probability averaging used at test time.
    probabilities = infer_probabilities(model, tokenizer, rows, max_length, device,
                                        batch_size=batch_size, variants=4)
    losses = []
    for row, probs in zip(rows, probabilities):
        target = torch.tensor(row["target"], dtype=torch.float32)
        losses.append(-(target * probs.clamp_min(1e-12).log()).sum().item())
    model.train()
    return math.fsum(losses) / len(losses)


def rlcd_ce_loss(logits, targets, kinds, sigma, group_size=4,
                 w_sph=0.75, w_rps=1.0, ce_weight=1.0):
    """Laya-style proper-reward policy gradient anchored by soft-target CE.

    Each decision gets a group of zero-mean Gaussian logit perturbations. The
    group-centered reward is a variance-reduced policy-gradient advantage.
    """
    if sigma <= 0 or group_size < 2 or ce_weight < 0:
        raise ValueError("RLCD requires sigma > 0, group_size >= 2, and ce_weight >= 0")
    if not (len(logits) == len(targets) == len(kinds)):
        raise ValueError("logits, targets, and kinds must have matching lengths")
    ce = decision_loss(logits, targets, "ce")
    policy_losses, mean_rewards = [], []
    for scores, target, kind in zip(logits, targets, kinds):
        scores = scores.float()
        target = target.to(device=scores.device, dtype=torch.float32)
        eps = torch.randn((group_size, scores.numel()), device=scores.device) * sigma
        eps = eps - eps.mean(dim=-1, keepdim=True)
        sampled_logits = scores.detach().unsqueeze(0) + eps
        probs = torch.softmax(sampled_logits, dim=-1)
        log_probs = probs.clamp_min(1e-12).log().clamp_min(-9.21)
        reward = (target.unsqueeze(0) * log_probs).sum(-1)
        reward = reward + w_sph * (target.unsqueeze(0) * probs).sum(-1) / probs.norm(dim=-1).clamp_min(1e-9)
        if kind == "score":
            cdf_probs = probs.cumsum(-1)
            cdf_target = target.cumsum(-1).unsqueeze(0)
            rps = ((cdf_probs - cdf_target).square()).sum(-1) / max(1, scores.numel() - 1)
            reward = reward - w_rps * rps
        advantage = reward - reward.mean()
        advantage = advantage / reward.std(unbiased=False).clamp_min(1e-6)
        log_policy = -((sampled_logits - scores.unsqueeze(0)).square().sum(-1)) / (2 * sigma * sigma)
        policy_losses.append(-(advantage.detach() * log_policy).mean())
        mean_rewards.append(reward.mean().detach())
    policy = torch.stack(policy_losses).mean()
    return ce_weight * ce + policy, {"ce": ce.detach(), "policy": policy.detach(),
                                    "mean_reward": torch.stack(mean_rewards).mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--max-length", type=int, default=640)
    p.add_argument("--eval-every", type=int, default=64)
    p.add_argument("--encoder-lr", type=float, default=2.5e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=12)
    p.add_argument("--objective", choices=("ce", "rlcd"), default="ce")
    p.add_argument("--rlcd-group-size", type=int, default=4)
    p.add_argument("--rlcd-sigma-start", type=float, default=0.4)
    p.add_argument("--rlcd-sigma-end", type=float, default=0.1)
    p.add_argument("--rlcd-ce-weight", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this run requires a BF16-capable CUDA GPU")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory must be new or empty: {args.output}")
    train_rows, dev_rows = load_records(args.train), load_records(args.dev)
    check_disjoint(train_rows, dev_rows)
    if any(r.get("target") is None for r in train_rows + dev_rows):
        raise ValueError("all rows need soft targets")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    started = time.monotonic()
    model, tokenizer, base_manifest = load_checkpoint(args.checkpoint, device="cuda:0")
    model.float(); model.backbone.config.use_cache = False
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(); torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True, exist_ok=True)
    model.backbone.config.save_pretrained(args.output / "backbone")
    tokenizer.save_pretrained(args.output / "tokenizer")
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.encoder_lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ])
    initial_sha = hashlib.sha256((args.checkpoint / "model.safetensors").read_bytes()).hexdigest()
    manifest = {
        "format_version": 1, "architecture": "shared-context-causal-listwise",
        "base_model": "Qwen/Qwen3-0.6B",
        "base_model_revision": base_manifest.get("requested_revision"),
        "initialization_checkpoint": "rongxinzy/LightJev-0.6B-v0.1",
        "initialization_revision": "b3d9a281885e3c315b12a22d3602fc627cdba3dd",
        "initialization_sha256": initial_sha,
        "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum, "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length, "precision": "bf16-autocast-fp32-parameters",
        "objective": args.objective,
        "loss": ("soft-target cross entropy over joint candidate scores" if args.objective == "ce"
                 else "proper-scoring-rule perturbation policy gradient plus soft-target CE"),
        "candidate_ordering": "randomized independently per sampled training decision",
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr, "warmup_steps": args.warmup_steps,
        "rlcd_group_size": args.rlcd_group_size,
        "rlcd_sigma_start": args.rlcd_sigma_start,
        "rlcd_sigma_end": args.rlcd_sigma_end,
        "rlcd_ce_weight": args.rlcd_ce_weight,
        "gradient_checkpointing": args.gradient_checkpointing,
        "selection_metric": "dev NLL of four-order averaged probabilities",
        "train_records": len(train_rows), "dev_records": len(dev_rows),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(args.dev.read_bytes()).hexdigest(), "status": "running",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    best_dev, history = float("inf"), []

    def save_best(step, value):
        nonlocal best_dev
        if value < best_dev:
            best_dev = value
            temp = args.output / "model.safetensors.tmp"
            save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(temp))
            temp.replace(args.output / "model.safetensors")
            manifest.update(best_step=step, best_dev_soft_ce=value)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    init = evaluate(model, tokenizer, dev_rows, args.max_length, "cuda:0", 4)
    save_best(0, init); history.append({"step": 0, "dev_soft_ce": init})
    with (args.output / "train_log.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            tick = time.monotonic(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.
            for _ in range(args.grad_accum):
                rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
                batch, targets = batch_to_device(rows, tokenizer, args.max_length, "cuda:0", rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = shared_scores(model, batch)
                    if args.objective == "rlcd":
                        progress = step / max(1, args.steps)
                        sigma = args.rlcd_sigma_start + (args.rlcd_sigma_end - args.rlcd_sigma_start) * progress
                        kinds = [row["kind"] for row in rows]
                        loss, rlcd_metrics = rlcd_ce_loss(
                            logits, targets, kinds, sigma, args.rlcd_group_size,
                            ce_weight=args.rlcd_ce_weight)
                    else:
                        loss = decision_loss(logits, targets, "ce")
                if not torch.isfinite(loss): raise ValueError("nonfinite loss")
                (loss / args.grad_accum).backward(); loss_sum += loss.item() / args.grad_accum
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm): raise ValueError("nonfinite gradient norm")
            factor = step / max(1, args.warmup_steps) if step <= args.warmup_steps else .5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)))
            for group, base_lr in zip(optimizer.param_groups, (args.encoder_lr, args.head_lr)):
                group["lr"] = base_lr * factor
            optimizer.step()
            row = {"step": step, "train_soft_ce": loss_sum, "gradient_norm": float(norm),
                   "lr_factor": factor, "seconds": time.monotonic() - tick,
                   "gpu_peak_gib": torch.cuda.max_memory_allocated() / 1024**3}
            if args.objective == "rlcd":
                row.update(sigma=sigma, ce_loss=float(rlcd_metrics["ce"]),
                           policy_loss=float(rlcd_metrics["policy"]),
                           mean_reward=float(rlcd_metrics["mean_reward"]))
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(model, tokenizer, dev_rows, args.max_length, "cuda:0", 4)
                row["dev_soft_ce"] = dev; save_best(step, dev)
            log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush(); history.append(row)
    manifest.update(status="complete", history=history, elapsed_seconds=time.monotonic() - started,
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    best_dev_soft_ce=best_dev)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "best_dev_soft_ce": best_dev,
                      "peak_gpu_gib": manifest["peak_gpu_gib"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__": main()
