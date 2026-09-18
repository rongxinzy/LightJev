"""Offline inference; candidate probabilities are uncalibrated by default."""
import json
import math
from pathlib import Path

import torch


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
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    return model, tokenizer, config


def predict(checkpoint, records, max_length=None, device="cpu", temperature=1.):
    from .training import _batch
    from .schema import validate_record
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    records = [validate_record(record) for record in records]
    model, tokenizer, config = load_checkpoint(checkpoint, device)
    max_length = config["max_length"] if max_length is None else max_length
    if max_length < 1:
        raise ValueError("max_length must be positive")
    results = []
    with torch.no_grad():
        for record in records:
            logits = model(**_batch([record], tokenizer, max_length, device))[0]
            probabilities = (logits / temperature).softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise ValueError("nonfinite probabilities")
            index = probabilities.argmax().item()
            result = {"id": record["id"], "kind": record["kind"], "candidates": record["candidates"],
                      "probabilities": probabilities.cpu().tolist(), "temperature": temperature,
                      "selected": record["candidates"][index], "selected_index": index}
            if record["kind"] == "score":
                # Ordered candidate indices define the ordinal score scale.
                result["expectation"] = sum(i * p for i, p in enumerate(result["probabilities"]))
            results.append(result)
    return results
