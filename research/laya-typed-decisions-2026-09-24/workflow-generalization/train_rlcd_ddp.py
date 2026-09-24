#!/usr/bin/env python3
"""Case-split Laya RLCD + soft-CE fine-tuning; torchrun one process per GPU."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward


def read_items(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collate(items: list[dict], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    n, length = len(items), max(len(x["ids"]) for x in items)
    kmax = max(len(x["markers"]) for x in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    pos = torch.zeros((n, kmax), dtype=torch.long)
    mask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, item in enumerate(items):
        li, k = len(item["ids"]), len(item["markers"])
        ids[i, :li] = torch.tensor(item["ids"], dtype=torch.long)
        att[i, :li] = 1
        pos[i, :k] = torch.tensor(item["markers"], dtype=torch.long)
        mask[i, :k] = True
        target[i, :k] = torch.tensor(item["target"], dtype=torch.float32)
    return {
        "input_ids": ids.to(device), "attention_mask": att.to(device),
        "marker_pos": pos.to(device), "marker_mask": mask.to(device),
        "target": target.to(device),
        "qtype": torch.tensor([x["qtype"] for x in items], dtype=torch.long, device=device),
    }


@torch.no_grad()
def evaluate_nll(model, items, pad_id, device, batch_size: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(items), batch_size):
        b = collate(items[start:start + batch_size], pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                              b["marker_mask"], b["qtype"])
        logp = torch.log_softmax(logits.float().masked_fill(~b["marker_mask"], -1e4), -1)
        total += float((-(b["target"] * logp).sum(-1)).sum().item())
        count += len(b["qtype"])
    model.train()
    return total / max(1, count)


def save_weights(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to(device="cpu", dtype=torch.float16).contiguous()
             for k, v in model.state_dict().items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(state, str(tmp))
    os.replace(tmp, path)


def train(args) -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    base_dir, data_dir, out_dir = Path(args.model_dir), Path(args.data_dir), Path(args.output_dir)
    cfg = json.loads((base_dir / "rl_agent_config.json").read_text())
    cfg["max_len"], cfg["head_max_len"] = args.max_len, args.head_max_len
    cfg["gradient_checkpointing"] = True
    tok = AutoTokenizer.from_pretrained(base_dir / "tokenizer")
    model = build_model(cfg, encoder_dir=str(base_dir / "encoder"))
    model.load_state_dict(load_file(str(base_dir / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_items = read_items(data_dir / "items_train.jsonl")
    dev_items = read_items(data_dir / "items_dev.jsonl")
    cal_items = read_items(data_dir / "items_calibration.jsonl")
    if not train_items or not dev_items or not cal_items or len(train_items) % world:
        raise ValueError("empty split or training rows are not evenly divisible by world size")
    local_items = train_items[rank::world]
    micro_chunks = math.ceil(len(local_items) / args.micro_batch)
    updates_per_epoch = math.ceil(micro_chunks / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    if args.max_updates > 0:
        total_updates = min(total_updates, args.max_updates)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=args.min_lr)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "start", "world_size": world,
                          "train_items": len(train_items), "dev_items": len(dev_items),
                          "calibration_items": len(cal_items), "items_per_rank": len(local_items),
                          "effective_batch": world * args.micro_batch * args.grad_accum,
                          "updates_per_epoch": updates_per_epoch, "epochs": args.epochs,
                          "bf16": torch.cuda.is_bf16_supported()}), flush=True)
    best_dev = float("inf")
    start_time = time.time()
    log_path = out_dir / "train_log.jsonl"
    updates_done = 0
    stop_after_epoch = False

    for epoch in range(args.epochs):
        rng = random.Random(args.seed + 1009 * epoch + rank)
        rng.shuffle(local_items)
        chunks = [local_items[i:i + args.micro_batch]
                  for i in range(0, len(local_items), args.micro_batch)]
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
        model.train()
        for group_start in range(0, len(chunks), args.grad_accum):
            group = chunks[group_start:group_start + args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            group_loss, group_reward = 0.0, 0.0
            for mi, chunk in enumerate(group):
                b = collate(chunk, tok.pad_token_id, device)
                sync_ctx = ddp.no_sync() if mi < len(group) - 1 else torch.enable_grad()
                with sync_ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, act = ddp(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                          b["marker_mask"], b["qtype"])
                    logits = logits.float()
                    mask = b["marker_mask"]
                    target = b["target"]
                    k = mask.sum(-1, keepdim=True).float().clamp_min(1.0)
                    eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma
                    eps = eps * mask
                    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
                    z = logits.detach().unsqueeze(0) + eps
                    q = torch.softmax(z.masked_fill(~mask.unsqueeze(0), -1e4), -1)
                    with torch.no_grad():
                        reward = proper_reward(q, target.unsqueeze(0), b["qtype"], mask,
                                               w_sph=args.w_sph, w_rps=args.w_rps)
                        adv = reward - reward.mean(0, keepdim=True)
                        adv = adv / (adv.std() + 1e-6)
                    logp_noise = -(((z - logits.unsqueeze(0)) ** 2) * mask.unsqueeze(0)).sum(-1)
                    logp_noise = logp_noise / (2.0 * sigma * sigma)
                    loss_rl = -(adv * logp_noise).mean()
                    log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
                    loss_ce = -(target * log_probs).sum(-1).mean()
                    loss = (args.rl_weight * loss_rl + args.ce_weight * loss_ce) / len(group) + 0.0 * act.sum()
                    loss.backward()
                group_loss += float(loss.detach().item()) * len(group)
                group_reward += float(reward.mean().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            updates_done += 1
            step = updates_done
            if rank == 0 and (step % args.log_every == 0 or step == 1):
                rec = {"event": "update", "step": step, "epoch": epoch + 1,
                       "loss": group_loss / len(group), "reward": group_reward / len(group),
                       "sigma": sigma, "lr_encoder": scheduler.get_last_lr()[0],
                       "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                       "seconds": time.time() - start_time}
                print(json.dumps(rec), flush=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            if args.max_updates > 0 and updates_done >= args.max_updates:
                stop_after_epoch = True
                break

        dist.barrier()
        if rank == 0:
            dev_nll = evaluate_nll(model, dev_items, tok.pad_token_id, device, args.eval_batch)
            rec = {"event": "epoch_end", "epoch": epoch + 1, "dev_soft_nll": dev_nll,
                   "seconds": time.time() - start_time}
            print(json.dumps(rec), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            save_weights(out_dir / "checkpoints" / f"epoch-{epoch + 1:02d}.safetensors", model)
            if dev_nll < best_dev:
                best_dev = dev_nll
                save_weights(out_dir / "selected.safetensors", model)
                (out_dir / "selected.json").write_text(json.dumps({
                    "selected_by": "minimum case-held-out dev soft-label NLL",
                    "epoch": epoch + 1, "dev_soft_nll": dev_nll, "seed": args.seed,
                    "base_revision": args.base_revision, "dataset_revision": args.dataset_revision,
                    "world_size": world, "effective_batch": world * args.micro_batch * args.grad_accum,
                    "sigma_start": args.sigma_start, "sigma_end": args.sigma_end,
                    "group_size": args.group_size, "ce_weight": args.ce_weight,
                }, indent=2) + "\n")
        dist.barrier()
        if stop_after_epoch:
            break

    if rank == 0:
        model.load_state_dict(load_file(str(out_dir / "selected.safetensors")), strict=True)
        temps = fit_calibration(model, cal_items, tok.pad_token_id, device, args.eval_batch)
        save_weights(out_dir / "model.safetensors", model)
        shutil.copytree(base_dir / "tokenizer", out_dir / "tokenizer", dirs_exist_ok=True)
        shutil.copytree(base_dir / "encoder", out_dir / "encoder", dirs_exist_ok=True)
        cfg.update({"model_name": "laya-typed-decisions-rlcd-8gpu", "fine_tuned": True,
                    "max_len": args.max_len, "head_max_len": args.head_max_len,
                    "temperature": temps, "training": {"method": "Laya RLCD + soft CE",
                    "epochs": args.epochs, "world_size": world, "seed": args.seed}})
        cfg.pop("temperature_by_options", None)
        (out_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (out_dir / "calibration.json").write_text(json.dumps({
            "case_held_out": True, "temperature_by_type": temps,
            "calibration_cases": len({x["case_id"] for x in cal_items}),
            "calibration_questions": len(cal_items),
        }, indent=2) + "\n")
        print(json.dumps({"event": "complete", "best_dev_soft_nll": best_dev,
                          "temperature_by_type": temps, "seconds": time.time() - start_time,
                          "output_dir": str(out_dir)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


@torch.no_grad()
def fit_calibration(model, items, pad_id, device, batch_size: int) -> list[float]:
    import numpy as np
    from scipy.optimize import minimize_scalar

    model.eval()
    collected = {0: [], 1: [], 2: []}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        b = collate(chunk, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                              b["marker_mask"], b["qtype"])
        logits = logits.float().cpu().numpy()
        for i, item in enumerate(chunk):
            k = len(item["target"])
            collected[item["qtype"]].append((logits[i, :k], np.asarray(item["target"])))
    temps = []
    for qtype in range(3):
        pairs = collected[qtype]
        if not pairs:
            temps.append(1.0)
            continue
        def objective(t):
            vals = []
            for z, y in pairs:
                a = z / t
                a = a - np.max(a)
                logp = a - np.log(np.exp(a).sum())
                vals.append(-float(np.sum(y * logp)))
            return float(np.mean(vals))
        opt = minimize_scalar(objective, method="bounded", bounds=(0.1, 10.0),
                              options={"xatol": 1e-4})
        temps.append(float(opt.x))
    return temps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-revision", required=True)
    ap.add_argument("--dataset-revision", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-updates", type=int, default=0,
                    help="optional cap for distributed smoke tests; 0 means the full epoch schedule")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--w-sph", type=float, default=0.75)
    ap.add_argument("--w-rps", type=float, default=1.0)
    ap.add_argument("--rl-weight", type=float, default=1.0,
                    help="set to 0 for the matched soft-CE-only control")
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
