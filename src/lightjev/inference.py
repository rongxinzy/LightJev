"""Offline independent or shared-context scoring for typed candidates."""
import json
import math
from pathlib import Path

import torch


_SHARED_PREFIX = (
    "You are making one typed decision. Evaluate each candidate given the state and question.\n"
    "State: {state}\nQuestion: {question}\nCandidates:\n"
)


def load_checkpoint(path, device="cpu"):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .model import DecisionModel
    path = Path(path)
    config = json.loads((path / "manifest.json").read_text())
    if config.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    backbone_config = AutoConfig.from_pretrained(path / "backbone", local_files_only=True,
                                                 trust_remote_code=False)
    model = DecisionModel(AutoModel.from_config(backbone_config, trust_remote_code=False))
    # Configs can inherit BF16; promote before copying FP32 checkpoint tensors
    # so load_state_dict never rounds trained weights through BF16 storage.
    model.float()
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    return model, tokenizer, config


def _fixed_orders(record, variants):
    count = len(record["candidates"])
    seed_text = str(record.get("id", ""))
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(seed_text))
    import random
    rng = random.Random(seed)
    seen, orders = set(), []
    target = min(variants, count)
    while len(orders) < target:
        order = list(range(count))
        rng.shuffle(order)
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            orders.append(order)
    return orders


def _shared_probabilities(model, tokenizer, records, max_length, device,
                          batch_size=8, permutations=4):
    """Score all candidates in a shared context and average aligned orders."""
    if batch_size < 1 or permutations < 1:
        raise ValueError("batch_size and permutations must be positive")
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            expanded = [(local_i, row, order)
                        for local_i, row in enumerate(rows)
                        for order in _fixed_orders(row, permutations)]
            buckets = [[] for _ in rows]
            for offset in range(0, len(expanded), batch_size):
                part = expanded[offset:offset + batch_size]
                sequences, candidate_positions, orders = [], [], []
                for _, row, order in part:
                    ids = list(tokenizer.encode(
                        _SHARED_PREFIX.format(state=row["state"], question=row["question"]),
                        add_special_tokens=True))
                    if not ids:
                        raise ValueError("tokenized shared prefix is empty")
                    positions = []
                    for index in order:
                        block = tokenizer.encode("Option: " + row["candidates"][index] + "\nScore:",
                                                add_special_tokens=False)
                        if not block:
                            raise ValueError("candidate tokenization is empty")
                        positions.append(len(ids) + len(block) - 1)
                        ids.extend(block)
                    if len(ids) > max_length:
                        raise ValueError(f"joint candidate sequence has {len(ids)} tokens, exceeding max_length={max_length}")
                    sequences.append(ids)
                    candidate_positions.append(positions)
                    orders.append(order)
                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = tokenizer.eos_token_id
                if pad_id is None:
                    raise ValueError("tokenizer requires pad_token_id or eos_token_id")
                width = max(map(len, sequences))
                input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
                attention_mask = torch.zeros_like(input_ids)
                for i, ids in enumerate(sequences):
                    input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[i, :len(ids)] = 1
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                enabled = torch.device(device).type == "cuda"
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                    output = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
                    hidden = output.last_hidden_state
                    logits = []
                    for i, positions in enumerate(candidate_positions):
                        selected = hidden[i, torch.as_tensor(positions, device=hidden.device)]
                        logits.append(model.head(selected.to(model.head[0].weight.dtype)).squeeze(-1).float())
                for (local_i, _, _), order, scores in zip(part, orders, logits):
                    aligned = torch.empty_like(scores)
                    aligned[torch.as_tensor(order, device=scores.device)] = scores
                    buckets[local_i].append(aligned.softmax(-1).cpu())
            result.extend([torch.stack(bucket).mean(0) for bucket in buckets])
    return result


def predict(checkpoint, records, max_length=None, device="cpu", temperature=None,
            permutations=4, batch_size=8):
    from .training import _batch
    from .schema import validate_record
    if temperature is not None:
        values = temperature.values() if isinstance(temperature, dict) else (temperature,)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("temperature must contain finite positive values")
    records = [validate_record(record) for record in records]
    model, tokenizer, config = load_checkpoint(checkpoint, device)
    max_length = config["max_length"] if max_length is None else max_length
    if max_length < 1:
        raise ValueError("max_length must be positive")
    architecture = config.get("architecture", "independent-candidate")
    if architecture == "shared-context-causal-listwise":
        raw_probabilities = _shared_probabilities(model, tokenizer, records, max_length, device,
                                                   batch_size=batch_size, permutations=permutations)
    elif architecture == "independent-candidate":
        raw_probabilities = []
        with torch.no_grad():
            for record in records:
                raw_probabilities.append(model(**_batch([record], tokenizer, max_length, device))[0].float().softmax(-1).cpu())
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    results = []
    calibrated = config.get("temperatures_by_kind", {}) if temperature is None else {}
    for record, raw in zip(records, raw_probabilities):
        if temperature is None:
            record_temperature = calibrated.get(record["kind"], 1.)
        elif isinstance(temperature, dict):
            record_temperature = temperature.get(record["kind"], 1.)
        else:
            record_temperature = temperature
        raw = raw.to(device)
        probabilities = (raw if record_temperature == 1 else
                         raw.clamp_min(1e-30).log().div(record_temperature).softmax(-1))
        if not torch.isfinite(probabilities).all():
            raise ValueError("nonfinite probabilities")
        index = probabilities.argmax().item()
        result = {"id": record["id"], "kind": record["kind"], "candidates": record["candidates"],
                  "probabilities": probabilities.cpu().tolist(), "temperature": record_temperature,
                  "selected": record["candidates"][index], "selected_index": index}
        if record["kind"] == "score":
            # Ordered candidate indices define the ordinal score scale.
            result["expectation"] = sum(i * p for i, p in enumerate(result["probabilities"]))
        results.append(result)
    return results
