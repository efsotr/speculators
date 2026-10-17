"""CPU checks for predicted-token hidden-state supervision."""

from types import SimpleNamespace

import pyarrow  # noqa: F401  # Initialize Arrow before Torch native libraries.
import pytest
import torch

from speculators.models.dflash2 import DFlash2DraftModel
from speculators.models.dflash2.hidden_loss import predicted_hidden_mse


def _loss(hidden, teacher, indices, aligned, mask, docs, *, sample=False):
    return predicted_hidden_mse(
        hidden,
        teacher,
        torch.tensor(indices),
        torch.tensor([aligned]),
        torch.tensor([mask]),
        torch.tensor([docs]),
        block_size=2,
        sample_from_anchor=sample,
    )


@pytest.mark.parametrize("sample,expected", [(False, 5.5), (True, 11.0)])
def test_slots_match_predicted_positions_with_hand_computed_mse(sample, expected):
    # Anchor a=1: bonus-anchor slots use positions 1,2; sampling uses 2,3.0
    teacher = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
    # Actual squared channel means: position 1=2.5, position 2=6.5,
    # position 3=12.5; slot weights 1,3 yield 5.5 or 11 respectively.
    actual = _loss(
        torch.zeros(1, 2, 2), teacher, [1, 2], [1, 3], [1] * 4, [0] * 4,
        sample=sample,
    )
    assert actual.item() == pytest.approx(expected)


def test_bonus_anchor_slot_zero_inherits_aligned_mask():
    teacher = torch.tensor([[[100.0], [2.0], [3.0]]])
    actual = _loss(
        torch.zeros(1, 2, 1), teacher, [0, 1], [0, 1], [1] * 3, [0] * 3
    )
    assert actual.item() == 4.0


def test_sampling_uses_supervision_mask_at_predicted_target_position():
    teacher = torch.tensor([[[100.0], [200.0], [3.0]]])
    actual = _loss(
        torch.zeros(1, 2, 1), teacher, [0, 1], [1, 1], [1, 0, 1], [0] * 3,
        sample=True,
    )
    assert actual.item() == 9.0


@pytest.mark.parametrize("sample", [False, True])
def test_targets_cannot_cross_document_or_sequence_end(sample):
    teacher = torch.tensor([[[1.0], [2.0], [100.0], [200.0]]])
    # First block starts at doc0 position1; position2 belongs to doc1.
    # Second block ends at sequence boundary. Sampling shifts its final target
    # beyond the sequence, which must be excluded instead of clamped into loss.
    actual = _loss(
        torch.zeros(1, 4, 1), teacher, [1, 2, 2, 3], [1] * 4,
        [1] * 4, [0, 0, 1, 1], sample=sample,
    )
    assert actual.item() == pytest.approx(40000.0 if sample else 50004.0 / 3)


@pytest.mark.parametrize("aligned,mask", [([0, 0], [1, 1]), ([1, 1], [0, 0])])
def test_padded_or_zero_supervision_returns_finite_zero(aligned, mask):
    teacher = torch.full((1, 2, 2), float("nan"))
    hidden = torch.zeros(1, 2, 2, requires_grad=True)
    actual = _loss(hidden, teacher, [0, 1], aligned, mask, [0, 0])
    assert actual.item() == 0.0
    actual.backward()
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))


def test_teacher_detached_and_only_supervised_student_slots_receive_gradients():
    teacher = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]], requires_grad=True)
    hidden = torch.zeros(1, 2, 2, requires_grad=True)
    actual = _loss(hidden, teacher, [0, 1], [1, 0], [1, 1], [0, 0])
    assert actual.item() == 10.0
    actual.backward()
    assert teacher.grad is None
    torch.testing.assert_close(hidden.grad, torch.tensor([[[-2.0, -4.0], [0.0, 0.0]]]))


@pytest.mark.parametrize("alpha", [0.0, 0.75])
def test_trainer_kwargs_forward_mse_weight_to_training_and_validation(alpha):
    for kwargs in DFlash2DraftModel.get_trainer_kwargs(
        loss_fn="ce", loss_implementation="eager", hidden_state_mse_alpha=alpha
    ):
        assert kwargs["hidden_state_mse_alpha"] == alpha


@pytest.mark.parametrize("alpha", [None, 0.5])
def test_forward_adds_normalized_hidden_mse_and_updates_loss_metric(monkeypatch, alpha):
    hidden = torch.zeros(1, 2, 2, requires_grad=True)
    unary = torch.zeros(1, 2, 4)
    targets = torch.ones(1, 2, 4)
    aligned = torch.ones(1, 2)
    indices = torch.tensor([0, 1])
    backbone_calls = []

    def backbone(*args, **kwargs):
        backbone_calls.append(args)
        return hidden, unary, targets, aligned, indices

    model = SimpleNamespace(
        _backbone_forward=backbone,
        _predecessor_ids=lambda *args: torch.zeros(1, 2, dtype=torch.long),
        candidate_selector=SimpleNamespace(
            top_k=1,
            score_candidates=lambda unary, hidden, predecessor, ids: unary.gather(-1, ids),
        ),
        block_size=2,
        config=SimpleNamespace(sample_from_anchor=False),
        verifier_norm=lambda teacher: teacher * 2,
    )

    def base_metrics(**kwargs):
        loss = hidden.sum() * 0 + 2
        return loss, {"loss_sum": loss.detach(), "loss_total": torch.tensor(1.0)}

    monkeypatch.setattr("speculators.models.dflash2.core.compute_metrics", base_metrics)
    forward = getattr(DFlash2DraftModel.forward, "__wrapped__", DFlash2DraftModel.forward)
    kwargs = {} if alpha is None else {"hidden_state_mse_alpha": alpha}
    _, loss, metrics = forward(
        model,
        hidden_states=torch.zeros(1, 2, 4),
        input_ids=torch.zeros(1, 2, dtype=torch.long),
        loss_mask=torch.ones(1, 2),
        verifier_last_hidden_states=torch.ones(1, 2, 2),
        document_ids=torch.zeros(1, 2, dtype=torch.long),
        **kwargs,
    )
    # Norm doubles teacher values: per-token MSE=4, added to base loss=2.0
    expected = 2.0 if alpha is None else 2.0 + alpha * 4.0
    assert loss.item() == expected
    assert metrics["loss_sum"].item() == expected
    assert len(backbone_calls) == 1
    if alpha is None:
        assert "hidden_state_mse_sum" not in metrics
    else:
        assert metrics["hidden_state_mse_sum"].item() == 4.0
        assert metrics["hidden_state_mse_total"].item() == 1.0
