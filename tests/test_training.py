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
