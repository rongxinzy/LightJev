#!/usr/bin/env python3
"""Shared-context, joint candidate scoring helpers for the LightJev ablation."""
from __future__ import annotations

import random
from typing import Sequence

import torch


PREFIX = (
    "You are making one typed decision. Evaluate each candidate given the state and question.\n"
    "State: {state}\nQuestion: {question}\nCandidates:\n"
)


def encode_joint(records, tokenizer, max_length: int, permutations: Sequence[Sequence[int]] | None = None):
    """Encode one shared state/question prefix and candidate blocks per record.

    Candidate blocks are separated and score locations are retained. With no
    permutations, input order is used. The returned mapping for each record is
    the candidate-index order represented by that sequence.
    """
    if not records:
        raise ValueError("records must be nonempty")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer requires pad_token_id or eos_token_id")
    sequences, positions, orders = [], [], []
    for row_index, record in enumerate(records):
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("each record needs at least two candidates")
        order = list(range(len(candidates))) if permutations is None else list(permutations[row_index])
        if sorted(order) != list(range(len(candidates))):
            raise ValueError("each candidate permutation must be a complete index permutation")
        prefix = PREFIX.format(state=record["state"], question=record["question"])
        ids = list(tokenizer.encode(prefix, add_special_tokens=True))
        if not ids:
            raise ValueError("tokenized shared prefix is empty")
        candidate_positions = []
        for index in order:
            block = tokenizer.encode("Option: " + candidates[index] + "\nScore:", add_special_tokens=False)
            if not block:
                raise ValueError("candidate tokenization is empty")
            candidate_positions.append(len(ids) + len(block) - 1)
            ids.extend(block)
        if len(ids) > max_length:
            raise ValueError(f"joint candidate sequence has {len(ids)} tokens, exceeding max_length={max_length}")
        sequences.append(ids)
        positions.append(candidate_positions)
        orders.append(order)
    width = max(map(len, sequences))
    input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(sequences):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "candidate_positions": positions, "candidate_orders": orders}


def shared_scores(model, batch):
    """Score each candidate marker from one backbone pass per decision."""
    output = model.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    hidden = output.last_hidden_state
    scores = []
    for i, positions in enumerate(batch["candidate_positions"]):
        pooled = hidden[i, torch.as_tensor(positions, device=hidden.device)]
        scores.append(model.head(pooled.to(model.head[0].weight.dtype)).squeeze(-1))
    return scores


def fixed_permutations(records, variants: int):
    """Stable pseudo-random orders, replicated independently of process state."""
    result = []
    for row in records:
        n = len(row["candidates"])
        seed_text = str(row.get("id", ""))
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(seed_text))
        rng = random.Random(seed)
        seen, orders = set(), []
        while len(orders) < min(variants, max(1, n)):
            order = list(range(n))
            rng.shuffle(order)
            key = tuple(order)
            if key not in seen:
                seen.add(key)
                orders.append(order)
        result.extend(orders)
    return result


def infer_probabilities(model, tokenizer, records, max_length, device, batch_size=8, variants=4):
    """Average aligned candidate probabilities over deterministic option orders."""
    if not records:
        return []
    if batch_size < 1 or variants < 1:
        raise ValueError("batch_size and variants must be positive")
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            expanded, source_indices = [], []
            output_rows = []
            for local_i, row in enumerate(rows):
                for order in fixed_permutations([row], variants):
                    expanded.append(row)
                    source_indices.append((local_i, order))
            # Candidate counts vary, so group one sequence per record variant;
            # reorder scores back to the original candidate identity afterward.
            for part_start in range(0, len(expanded), batch_size):
                part = expanded[part_start:part_start + batch_size]
                meta = source_indices[part_start:part_start + batch_size]
                perms = [item[1] for item in meta]
                batch = encode_joint(part, tokenizer, max_length, perms)
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                    logits = shared_scores(model, batch)
                for j, scores in enumerate(logits):
                    aligned = torch.empty_like(scores.float())
                    aligned[torch.as_tensor(perms[j], device=aligned.device)] = scores.float()
                    source_index = meta[j][0]
                    output_rows.append((source_index, aligned.softmax(-1).cpu()))
            buckets = [[] for _ in rows]
            for source_index, probs in output_rows:
                if source_index < len(rows):
                    buckets[source_index].append(probs)
            batch_outputs = []
            for bucket in buckets:
                if not bucket:
                    raise RuntimeError("missing candidate permutation prediction")
                batch_outputs.append(torch.stack(bucket).mean(0))
            all_outputs.extend(batch_outputs)
    return all_outputs
