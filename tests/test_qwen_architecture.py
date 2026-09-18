"""Offline architecture compatibility, not pretrained Qwen capability tests."""
import torch
from transformers import AutoModel, Qwen3Config
from lightjev.model import DecisionModel
from lightjev.training import decision_loss


def test_qwen3_forward_gradient_and_candidate_permutation():
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, pad_token_id=0,
                         max_position_embeddings=128, use_cache=False)
    model = DecisionModel(AutoModel.from_config(config))
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 4], [6, 7, 8, 9]])
    mask = torch.ones_like(ids)
    model.eval()
    logits = model(ids, mask, [3])[0]
    perm = torch.tensor([2, 0, 1])
    reordered = model(ids[perm], mask[perm], [3])[0]
    assert torch.allclose(reordered, logits[perm], atol=1e-6)
    loss = decision_loss([logits], [[0., 1., 0.]])
    loss.backward()
    grad = model.backbone.get_input_embeddings().weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
