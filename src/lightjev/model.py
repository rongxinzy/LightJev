"""Independent candidate scoring on a pretrained text backbone.

Each candidate runs through the backbone with its full context. A batch is not
shared-prefix inference: compute scales with candidate count, including the two
paths used for boolean decisions. The vocabulary generation head is unused.
"""
from __future__ import annotations

import json

import torch
from torch import nn


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        width = backbone.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, input_ids, attention_mask, question_sizes):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have the same 2D shape")
        sizes = list(question_sizes)
        if not sizes or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in sizes):
            raise ValueError("question_sizes must contain integers >= 2")
        if sum(sizes) != input_ids.shape[0]:
            raise ValueError("question_sizes must sum to the candidate batch size")
        active = attention_mask.bool()
        if not active.any(dim=1).all():
            raise ValueError("every candidate must have an active token")
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        # Works with either left or right padding, unlike lengths minus one.
        positions = torch.arange(active.shape[1], device=active.device).expand_as(active)
        last = positions.masked_fill(~active, -1).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        # Heads remain fp32 when a backbone is loaded in reduced precision.
        scores = self.head(pooled.to(self.head[0].weight.dtype)).squeeze(-1)
        return list(scores.split(sizes))


def encode_records(records, tokenizer, max_length: int):
    """Encode complete candidate paths, refusing silent input truncation.

    Targets and identifiers never enter model text. This also supports inference
    records without targets. Training callers should validate records separately.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    texts, sizes = [], []
    for record in records:
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 255:
            raise ValueError("each record requires 2 to 255 candidates")
        for key in ("state", "question"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if any(not isinstance(c, str) or not c.strip() for c in candidates):
            raise ValueError("candidates must be nonempty strings")
        if len(set(c.strip() for c in candidates)) != len(candidates):
            raise ValueError("candidates must be unique")
        sizes.append(len(candidates))
        for candidate in candidates:
            # JSON escaping makes field boundaries explicit even for multiline text.
            payload = {"state": record["state"], "question": record["question"], "candidate": candidate}
            texts.append("Evaluate this candidate for the question.\n" + json.dumps(payload, ensure_ascii=False) + "\nDecision:")
    if not texts:
        raise ValueError("records must not be empty")
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    lengths = mask.sum(dim=1)
    if (lengths > max_length).any():
        longest = int(lengths.max())
        raise ValueError(f"candidate input has {longest} tokens, exceeding max_length={max_length}; shorten input explicitly")
    return {"input_ids": ids.cpu(), "attention_mask": mask.cpu(), "question_sizes": sizes}
