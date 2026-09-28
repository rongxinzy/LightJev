"""Bounded native-Laya CE pilot with development-only selection; launch with torchrun."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import digest, load_rows
from metrics import summarize
from model_io import collate, forward, load_model, predict, save_checkpoint

BASE_SHA = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def train(args):
    dist.init_process_group("nccl")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA device required")
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name in ("train", "dev_seen", "dev_new"):
        if digest(args.data / f"{name}.jsonl") != manifest["splits"][name]["sha256"]:
            raise ValueError(f"data hash mismatch: {name}")
    if digest(args.base / "model.safetensors") != BASE_SHA:
        raise ValueError("starting weights are not the pinned ordinary Laya checkpoint")
    import laya.common
    lock = json.loads(Path(__file__).with_name("runtime_lock.json").read_text())
    for key, path in (("laya_common_sha256", laya.common.__file__),
                      ("tokenizer_sha256", args.base / "tokenizer/tokenizer.json"),
                      ("encoder_config_sha256", args.base / "encoder/config.json")):
        if digest(path) != lock[key]:
            raise ValueError(f"pinned runtime mismatch: {key}")
    train_rows = load_rows(args.data / "train.jsonl")
    dev_sets = {name: load_rows(args.data / f"{name}.jsonl") for name in ("dev_seen", "dev_new")}
    if args.smoke:
        chosen = {}
        for row in train_rows:
            chosen.setdefault(row["question"]["type"], row["group_id"])
        train_rows = [r for r in train_rows if r["group_id"] in chosen.values()]
        if set(chosen) != {"choice", "noul", "score"}:
            raise ValueError("smoke must exercise all three primitives")
        dev_sets = {"smoke_train": train_rows}
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("output must be new or empty; never overwrite a previous run")
        args.output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model, tokenizer, cfg = load_model(args.base, device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    ddp = DDP(model, device_ids=[local], find_unused_parameters=False)
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad],
         "lr": args.encoder_lr},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad],
         "lr": args.head_lr}], weight_decay=.01)
    batch_size = world * args.micro_batch * args.grad_accum
    if not args.smoke and batch_size != 64:
        raise ValueError("the preregistered pilot requires effective batch 64")
    rng = random.Random(args.seed)
    started, best, selected_step = time.monotonic(), float("inf"), 0
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    if rank == 0:
        import laya.common, transformers
        write_json(args.output / "manifest.json", {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "base_sha256": BASE_SHA, "data_manifest_sha256": digest(args.data / "manifest.json"),
            "world_size": world, "effective_batch": batch_size, "torch": torch.__version__,
            "transformers": transformers.__version__, "laya_common_sha256": digest(laya.common.__file__),
            "runtime_lock": lock,
            "code_sha256": {p.name: digest(p) for p in source_paths},
            "selection": "minimum family-macro raw development NLL including step-zero base",
            "locked_test_read": False})
    for step in range(args.steps + 1):
        if step:
            ddp.train()
            indices = rng.choices(range(len(train_rows)), k=batch_size) if args.smoke else rng.sample(range(len(train_rows)), batch_size)
            rate = min(1., step / 8) * .5 * (1. + math.cos(math.pi * max(0, step - 8) / max(1, args.steps - 8)))
            for group, lr in zip(optimizer.param_groups, (args.encoder_lr, args.head_lr)):
                group["lr"] = lr * rate
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(args.grad_accum):
                start = (micro * world + rank) * args.micro_batch
                rows = [train_rows[i] for i in indices[start:start + args.micro_batch]]
                b = collate(rows, tokenizer.pad_token_id, device)
                context = ddp.no_sync() if micro + 1 < args.grad_accum else torch.enable_grad()
                with context:
                    logits = forward(ddp, b, device)
                    loss = -(b["target"] * logits.log_softmax(-1)).sum(-1).mean()
                    (loss / args.grad_accum).backward()
                losses.append(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            average = torch.stack(losses).mean()
            dist.all_reduce(average)
            if rank == 0 and (step == 1 or step % 8 == 0):
                rec = {"event": "update", "step": step, "loss": float(average / world),
                       "seconds": time.monotonic() - started,
                       "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30}
                print(json.dumps(rec), flush=True)
                with (args.output / "history.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")
        if step % args.eval_every == 0 or step == args.steps:
            dist.barrier()
            stop = torch.zeros(1, device=device, dtype=torch.int32)
            if rank == 0:
                reports, all_rows = {}, []
                for name, rows in dev_sets.items():
                    preds = predict(model, rows, tokenizer.pad_token_id, device, args.eval_batch)
                    reports[name] = summarize(preds)
                    all_rows.extend(preds)
                value = summarize(all_rows)["macro"]["nll"]
                write_json(args.output / f"development-step-{step:04d}.json", reports)
                if value < best:
                    best, selected_step = value, step
                    if step:
                        save_checkpoint(args.output / "selected", model, args.base, cfg)
                    (args.output / "selected_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
                    write_json(args.output / "selection.json", {"step": step, "development_macro_nll": best,
                        "checkpoint": "selected" if step else str(args.base), "locked_test_used": False})
                print(json.dumps({"event": "development", "step": step, "macro_nll": value,
                                  "selected_step": selected_step, "seconds": time.monotonic() - started}), flush=True)
                if args.smoke:
                    chosen = load_rows(args.output / "selected_predictions.jsonl")
                    if best < .15 and all(r["prediction"] == r["label_index"] for r in chosen):
                        stop.fill_(1)
            dist.broadcast(stop, src=0)
            dist.barrier()
            if stop.item():
                break
    dist.barrier()
    if rank == 0:
        reference = load_rows(args.output / "selected_predictions.jsonl")
        if args.smoke and (selected_step == 0 or best >= .15 or
                          {r["kind"] for r in reference} != {"choice", "noul", "score"} or
                          any(r["prediction"] != r["label_index"] for r in reference)):
            raise RuntimeError("selected checkpoint failed tiny overfit gate; do not launch the pilot")
        chosen_path = args.output / "selected" if selected_step else args.base
        # Release optimizer/model memory before exact export round-trip verification.
        del ddp, optimizer, model
        torch.cuda.empty_cache()
        reloaded, _, _ = load_model(chosen_path, device)
        current = predict(reloaded, [r for rows in dev_sets.values() for r in rows], tokenizer.pad_token_id,
                          device, args.eval_batch)
        max_delta = max(abs(a - b) for x, y in zip(reference, current)
                        for a, b in zip(x["logits"], y["logits"]))
        if len(reference) != len(current) or max_delta > 1e-5 or any(
                x["prediction"] != y["prediction"] for x, y in zip(reference, current)):
            raise RuntimeError(f"checkpoint round-trip mismatch: {max_delta}")
        write_json(args.output / "complete.json", {"status": "complete", "selected_step": selected_step,
                   "best_macro_nll": best, "reload_max_logit_delta": max_delta,
                   "elapsed_seconds": time.monotonic() - started, "locked_test_read": False})
        print(json.dumps({"event": "complete", "selected_step": selected_step, "best_macro_nll": best}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("base", "data", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=32)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=5e-5)
    p.add_argument("--smoke", action="store_true")
    train(p.parse_args())
