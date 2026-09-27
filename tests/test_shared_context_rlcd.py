import math
import sys
from pathlib import Path

import pytest
import torch


WORKFLOW_DIR = (Path(__file__).parents[1] /
                "research/laya-typed-decisions-2026-09-24/workflow-generalization")
sys.path.insert(0, str(WORKFLOW_DIR))
from train_lightjev_shared_fold import rlcd_ce_loss  # noqa: E402


def test_rlcd_objective_is_finite_and_differentiable_for_each_primitive():
    logits = [torch.tensor([0.1, -0.2, 0.4], requires_grad=True),
              torch.tensor([-0.3, 0.2], requires_grad=True)]
    targets = [torch.tensor([0.1, 0.2, 0.7]), torch.tensor([0.25, 0.75])]
    torch.manual_seed(17)
    loss, metrics = rlcd_ce_loss(logits, targets, ["score", "noul"], sigma=0.2)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["mean_reward"])
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in logits)


def test_rlcd_ce_component_matches_soft_target_cross_entropy():
    logits = [torch.tensor([0.0, 0.0], requires_grad=True)]
    target = [torch.tensor([0.25, 0.75])]
    _, metrics = rlcd_ce_loss(logits, target, ["choice"], sigma=0.2)
    assert metrics["ce"].item() == pytest.approx(-0.25 * math.log(0.5) - 0.75 * math.log(0.5))


@pytest.mark.parametrize("sigma,group_size", [(0.0, 4), (0.2, 1)])
def test_rlcd_objective_rejects_invalid_sampling_parameters(sigma, group_size):
    with pytest.raises(ValueError, match="RLCD requires"):
        rlcd_ce_loss([torch.zeros(2)], [torch.tensor([0.5, 0.5])],
                     ["choice"], sigma, group_size)
