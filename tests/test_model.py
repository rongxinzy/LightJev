from types import SimpleNamespace
import pytest
import torch
from torch import nn
from lightjev.model import DecisionModel, encode_records


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.emb = nn.Embedding(64, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class Tokenizer:
    def __call__(self, texts, **kwargs):
        self.texts = texts
        assert kwargs["truncation"] is False
        lengths = [len(t) for t in texts]
        mask = torch.zeros(len(texts), max(lengths), dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        return {"input_ids": mask.clone(), "attention_mask": mask}


def record():
    return dict(state="a state", question="choose", candidates=["candidate one", "candidate two"], target=[.4, .6], id="secret-id")


def test_padding_pooling_and_gradient():
    torch.manual_seed(0)
    model = DecisionModel(Backbone())
    ids = torch.tensor([[0, 1, 2], [1, 2, 0], [0, 0, 3], [3, 0, 0]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
    scores = model(ids, mask, [2, 2])
    assert len(scores) == 2
    assert torch.allclose(scores[0][0], scores[0][1])
    assert torch.allclose(scores[1][0], scores[1][1])
    sum(s.sum() for s in scores).backward()
    assert model.backbone.emb.weight.grad is not None
    assert model.head[1].weight.grad is not None


def test_encoding_no_targets_and_no_truncation():
    tokenizer = Tokenizer()
    result = encode_records([record()], tokenizer, 1000)
    assert result["question_sizes"] == [2]
    assert "candidate one" in tokenizer.texts[0]
    assert "candidate two" in tokenizer.texts[1]
    assert all("secret-id" not in t and "target" not in t for t in tokenizer.texts)
    assert result["input_ids"].device.type == "cpu"
    with pytest.raises(ValueError, match="exceeding"):
        encode_records([record()], tokenizer, 2)


def test_bad_batch():
    model = DecisionModel(Backbone())
    with pytest.raises(ValueError, match="sum"):
        model(torch.ones(3, 2, dtype=torch.long), torch.ones(3, 2), [2])
    with pytest.raises(ValueError, match="active token"):
        model(torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2), [2])
