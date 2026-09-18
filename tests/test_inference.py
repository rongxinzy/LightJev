"""Checkpoint loading must preserve trained parameter precision across HF versions."""
import torch


def test_bf16_constructor_preserves_fp32_checkpoint_exactly(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from transformers import AutoModel
    from lightjev.demo import run_demo
    from lightjev.inference import load_checkpoint

    root = tmp_path / 'demo'
    run_demo(root, steps=1)
    checkpoint = root / 'checkpoint'
    expected = load_file(str(checkpoint / 'model.safetensors'))
    # Prove this fixture detects a lossy BF16 round trip, not merely a dtype change.
    assert any(not torch.equal(v, v.bfloat16().float()) for v in expected.values()
               if v.is_floating_point())
    original_constructor = AutoModel.from_config
    observed = []

    def construct_bf16(*args, **kwargs):
        backbone = original_constructor(*args, **kwargs).bfloat16()
        observed.append({p.dtype for p in backbone.parameters()})
        return backbone

    monkeypatch.setattr(AutoModel, 'from_config', construct_bf16)
    restored, _, _ = load_checkpoint(checkpoint)
    assert observed == [{torch.bfloat16}]
    actual = restored.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype, key
        assert torch.equal(actual[key], value), key
