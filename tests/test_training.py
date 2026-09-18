import math

import pytest
import torch

from lightjev.training import check_disjoint, decision_loss
from lightjev.inference import predict


def test_soft_ce_and_equal_question_weight():
    logits = [torch.zeros(2, requires_grad=True), torch.zeros(4, requires_grad=True)]
    loss = decision_loss(logits, [[0.25, 0.75], [0, 0, 0, 1]])
    assert loss.item() == pytest.approx((math.log(2) + math.log(4)) / 2)
    loss.backward()
    assert logits[0].grad is not None


def test_brier_zero_at_target():
    assert decision_loss([torch.tensor([0., 0.])], [[.5, .5]], "brier").item() == 0


def test_group_leakage_rejected():
    with pytest.raises(ValueError, match="group_id overlap"):
        check_disjoint([{"id": "a", "group_id": "x"}], [{"id": "b", "group_id": "x"}])


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_before_loading(temperature):
    with pytest.raises(ValueError, match="temperature"):
        predict("does-not-exist", [], temperature=temperature)


def test_initial_checkpoint_warmup_accumulation_and_logs(tmp_path):
    import json
    from lightjev.demo import run_demo
    from lightjev.training import train
    root = tmp_path / 'demo'
    run_demo(root, steps=1)
    output = tmp_path / 'enhanced'
    summary = train(root / 'train.jsonl', root / 'dev.jsonl', output,
                    model_name=str(root / 'tiny-backbone'), steps=2, head_steps=1,
                    batch_size=2, grad_accum_steps=2, eval_every=10,
                    head_lr=2e-4, gradient_checkpointing=True, max_length=128)
    entries = [json.loads(line) for line in (output / 'train_log.jsonl').read_text().splitlines()]
    assert [e['step'] for e in entries] == [0, 1, 2, 3]
    assert [e['phase'] for e in entries] == ['initial', 'head', 'full', 'full']
    assert 'dev' in entries[1] and 'dev' not in entries[2] and 'dev' in entries[3]
    assert summary['samples_seen'] == 12
    assert summary['best_dev']['ce'] <= summary['initial_dev']['ce']
    assert summary['status'] == 'complete'
    assert summary['parameter_counts']['head'] > 0
    assert json.loads((output / 'manifest.json').read_text())['best_step'] == summary['best_step']


def test_bf16_cpu_rejected(tmp_path):
    from lightjev.training import train
    with pytest.raises(ValueError, match='bf16 requires'):
        train('unused', 'unused', tmp_path, precision='bf16', device='cpu')


def test_bf16_pretrained_backbone_uses_fp32_parameters_and_adam_state(tmp_path, monkeypatch):
    """A loader returning BF16 must not create BF16 Adam master parameters."""
    from safetensors.torch import load_file
    from transformers import AutoModel
    from lightjev.demo import run_demo
    from lightjev.training import train

    root = tmp_path / 'bf16-demo'
    run_demo(root, steps=1)
    original_load = AutoModel.from_pretrained
    original_adam = torch.optim.AdamW
    loaded_dtypes, optimizer_dtypes = [], []

    def load_bf16(*args, **kwargs):
        # Force the HF5 config-inherited dtype behavior also on older HF versions.
        backbone = original_load(*args, **kwargs).to(dtype=torch.bfloat16)
        loaded_dtypes.append({p.dtype for p in backbone.parameters()})
        return backbone

    class InspectAdam(original_adam):
        def step(self, *args, **kwargs):
            assert {p.dtype for group in self.param_groups for p in group['params']} == {torch.float32}
            result = super().step(*args, **kwargs)
            moment_dtypes = {state[key].dtype for state in self.state.values()
                             for key in ('exp_avg', 'exp_avg_sq') if key in state}
            optimizer_dtypes.append(moment_dtypes)
            return result

    monkeypatch.setattr(AutoModel, 'from_pretrained', load_bf16)
    monkeypatch.setattr(torch.optim, 'AdamW', InspectAdam)
    output = tmp_path / 'fp32-checkpoint'
    summary = train(root / 'train.jsonl', root / 'dev.jsonl', output,
                    model_name=str(root / 'tiny-backbone'), steps=1, max_length=128)
    assert loaded_dtypes == [{torch.bfloat16}]
    assert optimizer_dtypes == [{torch.float32}]
    weights = load_file(str(output / 'model.safetensors'))
    assert {tensor.dtype for tensor in weights.values() if tensor.is_floating_point()} == {torch.float32}
    assert summary['parameter_dtype'] == summary['optimizer_state_dtype'] == 'float32'
