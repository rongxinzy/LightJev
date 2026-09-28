"""Shared native Laya loading, batching, raw prediction and exact checkpoint IO."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model


def load_model(path, device):
    path = Path(path)
    cfg = json.loads((path / "rl_agent_config.json").read_text())
    cfg.update(max_len=512, head_max_len=192, temperature=[1., 1., 1.])
    cfg.pop("temperature_by_options", None)
    model = build_model(cfg, encoder_dir=str(path / "encoder"))
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    model.float().to(device)
    # The inherited escalation head is not a calibrated abstention policy and is unused.
    model.act_head.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    return model, tokenizer, cfg


def collate(rows, pad_id, device):
    n, length, kmax = len(rows), max(len(r["ids"]) for r in rows), max(len(r["markers"]) for r in rows)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((n, kmax), dtype=torch.long)
    mask = torch.zeros((n, kmax), dtype=torch.bool)
    targets = torch.zeros((n, kmax), dtype=torch.float32)
    for i, row in enumerate(rows):
        length_i, ki = len(row["ids"]), len(row["markers"])
        ids[i, :length_i] = torch.tensor(row["ids"])
        attention[i, :length_i] = 1
        positions[i, :ki] = torch.tensor(row["markers"])
        mask[i, :ki] = True
        targets[i, :ki] = torch.tensor(row["target"])
    return {"input_ids": ids.to(device), "attention_mask": attention.to(device),
            "marker_pos": positions.to(device), "marker_mask": mask.to(device),
            "qtype": torch.tensor([r["qtype"] for r in rows], device=device),
            "target": targets.to(device)}


def forward(model, batch, device):
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
    with amp:
        logits, _ = model(**{k: v for k, v in batch.items() if k != "target"})
    return logits.float().masked_fill(~batch["marker_mask"], -1e4)


@torch.no_grad()
def predict(model, rows, pad_id, device, batch_size=8):
    model.eval()
    results = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        logits = forward(model, collate(chunk, pad_id, device), device).cpu()
        for row, scores in zip(chunk, logits):
            scores = scores[:len(row["markers"])].tolist()
            # Restore canonical candidate identities if presentation order was changed.
            order = row.get("order", list(range(len(scores))))
            aligned = [0.] * len(scores)
            for i, canonical in enumerate(order):
                aligned[canonical] = scores[i]
            probs = torch.tensor(aligned).softmax(-1).tolist()
            results.append({"id": row["id"], "group_id": row["group_id"], "family": row["family"],
                            "split": row["split"], "variant": row["variant"],
                            "kind": row["question"]["type"], "label_index": row["label_index"],
                            "semantic_values": row["semantic_values"], "logits": aligned,
                            "probabilities": probs, "prediction": max(range(len(probs)), key=probs.__getitem__)})
    return results


def save_checkpoint(path, model, base, cfg):
    path, base = Path(path), Path(base)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    temp = path / "model.safetensors.tmp"
    save_file(state, str(temp))
    os.replace(temp, path / "model.safetensors")
    for name in ("tokenizer", "encoder"):
        if not (path / name).exists():
            shutil.copytree(base / name, path / name)
    cfg = {**cfg, "temperature": [1., 1., 1.], "model_name": "lightjev-laya-rule-pilot",
           "act_head_status": "inherited, frozen, not validated; do not use for escalation"}
    cfg.pop("temperature_by_options", None)
    (path / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
