"""Small numerical and compilation checks for D-PACE development weights."""

import math

import pytest
import torch
import yaml

from speculators.losses import resolve_loss_config
from speculators.losses.eager import ce_loss, kl_div_loss, tv_loss
from speculators.losses.utils import (
    dpace_dev_loss_decay,
    dpace_dev_token_probs,
    dpace_loss_decay,
)
from speculators.models.dflash2.core import DFlash2DraftModel
from speculators.models.dflash2.metrics import (
    compute_metrics,
    selector_training_candidates,
)
from speculators.train.config import TrainConfig

pytestmark = pytest.mark.smoke

MODES = ("q", "p", "min_q_p", "p_over_q", "q_over_p")


def _reference(q, p, mask, block_size, alpha, mode):  # noqa: PLR0917
    """Scalar loops deliberately independent of the tensor implementation."""
    out = []
    tiny = torch.finfo(torch.float32).tiny
    for offset in range(0, q.numel(), block_size):
        prefix = 1.0
        prefixes = []
        for index in range(offset, offset + block_size):
            qi, pi = q.flatten()[index].item(), p.flatten()[index].item()
            probability = {
                "q": qi,
                "p": pi,
                "min_q_p": min(qi, pi),
                "p_over_q": min(1.0, pi / max(qi, tiny)),
                "q_over_p": min(1.0, qi / max(pi, tiny)),
            }[mode]
            active = mask.flatten()[index].item()
            prefix *= (1 - alpha) * probability + alpha if active else 1.0
            prefixes.append(prefix * active)
        out.extend(
            sum(prefixes[index:]) * mask.flatten()[offset + index].item()
            for index in range(block_size)
        )
    return torch.tensor(out).reshape_as(mask)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_weights_match_scalar_reference(mode, alpha):
    q = torch.tensor([[0.1, 0.8, 0.3, 0.9, 0.6, 0.2, 0.4, 0.7]])
    p = torch.tensor([[0.6, 0.2, 0.5, 0.7, 0.1, 0.8, 0.4, 0.9]])
    mask = torch.tensor([[0.0, 1, 1, 0, 0, 1, 1, 1]])
    result = dpace_dev_loss_decay(
        torch.arange(8).reshape(1, -1), mask, 4, alpha, q, p, accept_prob=mode
    )
    torch.testing.assert_close(result, _reference(q, p, mask, 4, alpha, mode))


@pytest.mark.parametrize("mode", MODES)
def test_weights_are_detached_and_zero_masks_are_finite(mode):
    q = torch.tensor([[0.0, 1e-40, 1.0, 0.3]], requires_grad=True)
    p = torch.tensor([[1.0, 0.5, 0.0, 0.7]], requires_grad=True)
    mask = torch.ones_like(q)
    weights = dpace_dev_loss_decay(None, mask, 4, 0.5, q, p, accept_prob=mode)
    assert not weights.requires_grad
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, _reference(q, p, mask, 4, 0.5, mode))
    empty = dpace_dev_loss_decay(
        None, torch.zeros_like(mask), 4, 0.5, q, p, accept_prob=mode
    )
    torch.testing.assert_close(empty, torch.zeros_like(empty))


@pytest.mark.parametrize("mode", MODES)
def test_zero_probabilities_and_all_masked_blocks(mode):
    q = torch.zeros(1, 8)
    p = torch.zeros_like(q)
    mask = torch.tensor([[0.0, 1, 1, 1, 0, 0, 0, 0]])
    actual = dpace_dev_loss_decay(None, mask, 4, 0.5, q, p, accept_prob=mode)
    torch.testing.assert_close(actual, _reference(q, p, mask, 4, 0.5, mode))
    assert torch.isfinite(actual).all()


def test_q_weights_reproduce_original_dpace_ce():
    torch.manual_seed(71)
    logits = torch.randn(1, 8, 7)
    targets = torch.randn_like(logits)
    mask = torch.tensor([[0.0, 1, 1, 1, 0, 1, 1, 0]])
    q, p = dpace_dev_token_probs(logits, targets)
    original = dpace_loss_decay(None, mask, 4, 0.5, ce_loss(logits, targets))
    updated = dpace_dev_loss_decay(None, mask, 4, 0.5, q, p)
    torch.testing.assert_close(updated, original)


def test_token_probabilities_use_teacher_argmax_and_full_vocabulary():
    logits = torch.tensor([[[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]]])
    targets = torch.tensor([[[4.0, 1.0, 0.0], [1.0, 2.0, 5.0]]])
    q, p = dpace_dev_token_probs(logits, targets)
    expected_q = torch.tensor(
        [
            [
                math.exp(1) / sum(map(math.exp, [1, 3, 2])),
                math.exp(2) / sum(map(math.exp, [4, 1, 2])),
            ]
        ]
    )
    expected_p = torch.tensor(
        [
            [
                math.exp(4) / sum(map(math.exp, [4, 1, 0])),
                math.exp(5) / sum(map(math.exp, [1, 2, 5])),
            ]
        ]
    )
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(p, expected_p)


def _weighted_kl(logits, targets, mask, mode):
    q, p = dpace_dev_token_probs(logits, targets)
    weights = dpace_dev_loss_decay(None, mask, 4, 0.5, q, p, accept_prob=mode)
    return (kl_div_loss(logits, targets) * weights * mask).sum() / (mask.sum() + 1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_kl_gradients_treat_weights_as_constants(mode):
    torch.manual_seed(72)
    logits = torch.randn(1, 8, 7, requires_grad=True)
    targets = torch.randn_like(logits)
    mask = torch.tensor([[0.0, 1, 1, 1, 0, 1, 1, 1]])
    loss = _weighted_kl(logits, targets, mask, mode)
    gradient = torch.autograd.grad(loss, logits)[0]
    q, p = dpace_dev_token_probs(logits.detach(), targets)
    weights = _reference(q, p, mask, 4, 0.5, mode)
    reference_logits = logits.detach().clone().requires_grad_()
    expected = (kl_div_loss(reference_logits, targets) * weights * mask).sum()
    expected = expected / (mask.sum() + 1e-5)
    expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("mode", MODES)
def test_cpu_inductor_fullgraph_forward_backward(mode):
    torch.manual_seed(73)
    logits = torch.randn(1, 8, 7, requires_grad=True)
    targets = torch.randn_like(logits)
    mask = torch.tensor([[0.0, 1, 1, 1, 0, 1, 1, 1]])
    reference_loss = _weighted_kl(logits, targets, mask, mode)
    reference_grad = torch.autograd.grad(reference_loss, logits)[0]
    compiled = torch.compile(_weighted_kl, fullgraph=True, backend="inductor")
    actual = compiled(logits, targets, mask, mode)
    actual_grad = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual, reference_loss)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_dflash2_kl_and_recomputed_selector_probabilities(mode):
    torch.manual_seed(74)
    unary = torch.randn(1, 8, 7, requires_grad=True)
    teacher = torch.randn_like(unary)
    mask = torch.tensor([[0.0, 1, 1, 1, 0, 1, 1, 0]])
    candidates = unary.detach().topk(3, dim=-1).indices
    candidates, positions, contains = selector_training_candidates(
        candidates, teacher.argmax(-1)
    )
    # Deliberately differs from unary candidate logits, exercising reranked q.
    selector = torch.randn(1, 8, 3, requires_grad=True) * 2
    loss, metrics = compute_metrics(
        unary,
        teacher,
        candidates,
        selector,
        positions,
        contains,
        mask,
        block_size=4,
        top_k=3,
        loss_config=resolve_loss_config("kl_div", "eager"),
        tv_loss_fn=tv_loss,
        selector_loss_alpha=0.7,
        per_position_loss_weight="dpace_dev",
        dpace_alpha=0.5,
        dpace_dev_accept_prob=mode,
    )
    q, p = dpace_dev_token_probs(unary.detach(), teacher)
    unary_weights = _reference(q, p, mask, 4, 0.5, mode)
    selector_q = selector.detach().softmax(-1).gather(-1, positions.unsqueeze(-1))
    selector_weights = _reference(selector_q.squeeze(-1), p, mask, 4, 0.5, mode)
    ce = torch.nn.functional.cross_entropy(
        selector.flatten(0, -2), positions.flatten(), reduction="none"
    ).reshape_as(mask)
    # Match each loss's existing reduction epsilon.
    unary_expected = (kl_div_loss(unary, teacher) * mask * unary_weights).sum()
    unary_expected /= mask.sum() + 1e-5
    selector_expected = (ce * mask * selector_weights).sum() / (mask.sum() + 1e-5)
    expected = unary_expected + 0.7 * selector_expected
    torch.testing.assert_close(metrics["unary_loss_sum"], unary_expected)
    torch.testing.assert_close(metrics["selector_loss_sum"], selector_expected)
    torch.testing.assert_close(loss, expected)
    actual_gradients = torch.autograd.grad(loss, (unary, selector), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (unary, selector))
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)
        assert torch.isfinite(actual).all()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("loss_name", ["ce", "kl_div"])
def test_cli_roundtrip_and_dflash2_trainer_kwargs(mode, loss_name):
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "dummy",
            "--speculator-type",
            "dflash2",
            "--loss-fn",
            loss_name,
            "--loss-implementation",
            "eager",
            "--per-position-loss-weight",
            "dpace_dev",
            "--dpace-dev-accept-prob",
            mode,
            "--dpace-alpha",
            "0.3",
        ]
    )
    assert TrainConfig.from_flat(cfg.flatten()).model_dump() == cfg.model_dump()
    assert (
        TrainConfig.model_validate(
            yaml.safe_load(cfg.dump_yaml())["train"]
        ).model_dump()
        == cfg.model_dump()
    )
    train, val = DFlash2DraftModel.get_trainer_kwargs(**cfg.flatten())
    for kwargs in (train, val):
        assert kwargs["dpace_dev_accept_prob"] == mode
        assert kwargs["per_position_loss_weight"] == "dpace_dev"
        assert kwargs["dpace_alpha"] == pytest.approx(0.3)
        assert loss_name in kwargs["loss_config"]


@pytest.mark.parametrize("alpha", [0, -0.5, 1.1, float("nan"), float("inf")])
def test_schema_rejects_invalid_dpace_dev_alpha(alpha):
    with pytest.raises(ValueError):
        TrainConfig.model_validate(
            {
                "speculator_type": "dflash2",
                "dflash": {
                    "per_position_loss_weight": "dpace_dev",
                    "dpace_alpha": alpha,
                },
            }
        )


def test_schema_rejects_invalid_dpace_dev_mode():
    with pytest.raises(ValueError):
        TrainConfig.model_validate(
            {
                "speculator_type": "dflash2",
                "dflash": {
                    "per_position_loss_weight": "dpace_dev",
                    "dpace_dev_accept_prob": "bad",
                },
            }
        )


@pytest.mark.parametrize("loss_name", ["tv", '{"ce": 0.1, "tv": 0.9}'])
def test_schema_rejects_unsupported_dpace_dev_losses(loss_name):
    with pytest.raises(ValueError):
        TrainConfig.model_validate(
            {
                "speculator_type": "dflash2",
                "loss": {"loss_fn": loss_name},
                "dflash": {"per_position_loss_weight": "dpace_dev"},
            }
        )


def _combined_loss(  # noqa: PLR0917
    unary, teacher, selector, candidates, positions, contains, mask, mode
):
    loss, _ = compute_metrics(
        unary,
        teacher,
        candidates,
        selector,
        positions,
        contains,
        mask,
        block_size=4,
        top_k=3,
        loss_config=resolve_loss_config("kl_div", "eager"),
        tv_loss_fn=tv_loss,
        per_position_loss_weight="dpace_dev",
        dpace_dev_accept_prob=mode,
    )
    return loss


@pytest.mark.parametrize("mode", ["q", "p_over_q"])
def test_dflash2_cpu_inductor_fullgraph_forward_backward(mode):
    torch.manual_seed(75)
    unary = torch.randn(1, 8, 7, requires_grad=True)
    teacher = torch.randn_like(unary)
    selector = torch.randn(1, 8, 3, requires_grad=True)
    candidates, positions, contains = selector_training_candidates(
        unary.detach().topk(3, dim=-1).indices, teacher.argmax(-1)
    )
    mask = torch.tensor([[0.0, 1, 1, 1, 0, 1, 1, 0]])
    args = (unary, teacher, selector, candidates, positions, contains, mask, mode)
    expected = _combined_loss(*args)
    expected_grads = torch.autograd.grad(expected, (unary, selector))
    compiled = torch.compile(_combined_loss, fullgraph=True, backend="inductor")
    actual = compiled(*args)
    actual_grads = torch.autograd.grad(actual, (unary, selector))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
