import torch
from torch import nn


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        if text.startswith("Option: "):
            candidate = text.removeprefix("Option: ").split("\n", 1)[0]
            return [ord(candidate[0])]
        return [1]


class _Backbone(nn.Module):
    def forward(self, input_ids, attention_mask):
        return type("Output", (), {"last_hidden_state": input_ids.float().unsqueeze(-1)})()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()
        self.head = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Identity())
        with torch.no_grad():
            self.head[0].weight.fill_(1.)


def test_shared_context_prediction_realigns_candidate_permutations(monkeypatch, tmp_path):
    from lightjev.inference import predict

    monkeypatch.setattr("lightjev.inference.load_checkpoint", lambda *_args: (
        _Model(), _Tokenizer(), {
            "format_version": 1,
            "architecture": "shared-context-causal-listwise",
            "max_length": 64,
            "temperatures_by_kind": {"choice": 2.},
        }
    ))
    result = predict(tmp_path, [{
        "id": "case-1:q1", "group_id": "case-1", "state": "state",
        "question": "pick", "kind": "choice", "candidates": ["alpha", "beta"],
    }])[0]

    assert result["selected"] == "beta"
    assert result["temperature"] == 2.
    assert len(result["probabilities"]) == 2
    assert abs(sum(result["probabilities"]) - 1.) < 1e-6


def test_shared_context_rejects_candidate_input_overflow(monkeypatch, tmp_path):
    import pytest
    from lightjev.inference import predict

    monkeypatch.setattr("lightjev.inference.load_checkpoint", lambda *_args: (
        _Model(), _Tokenizer(), {"format_version": 1,
                                 "architecture": "shared-context-causal-listwise",
                                 "max_length": 1}
    ))
    with pytest.raises(ValueError, match="exceeding max_length"):
        predict(tmp_path, [{
            "id": "case-1:q1", "group_id": "case-1", "state": "state",
            "question": "pick", "kind": "choice", "candidates": ["alpha", "beta"],
        }])
