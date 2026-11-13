"""Numerical, gradient, cache, and configuration checks for DPA-dev."""

import pytest
import torch
from torch.nn import functional

from speculators.losses import eager, resolve_loss_config
from speculators.losses.dpa_dev import prepare_dpa_dev
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics as dflash_metrics
from speculators.models.dflash2.core import DFlash2DraftModel
from speculators.models.dflash2.hidden_loss import predicted_hidden_mse
from speculators.models.dflash2.metrics import (
    compute_metrics as selector_metrics,
)
from speculators.models.dflash2.metrics import (
    selector_training_candidates,
)
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.train.config import TrainConfig


def inputs(device="cpu"):
    torch.manual_seed(42)
    q = torch.randn(1, 8, 7, device=device, requires_grad=True)
    p = torch.randn_like(q)
    c = torch.randn(1, 8, device=device, requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0]], device=device)
    ids = torch.tensor([[-100, 2, 3, 4, 5, 6, -100, -100]], device=device)
    return q, p, c, mask, ids


def reference_weights(prob, mask, alpha=0.3):
    prob = prob.detach()
    result = torch.zeros_like(prob)
    for start in (0, 4):
        prefix = 1.0
        for j in range(start, start + 4):
            if mask[0, j]:
                prefix *= alpha + (1 - alpha) * float(prob[0, j])
                for i in range(start, j + 1):
                    if mask[0, i]:
                        result[0, i] += prefix
    return result


@pytest.mark.parametrize("signal", ["ce", "ce_gt", "kl_div2", "accept"])
def test_shared_weights_gradients_and_reuse(signal):
    q, p, c, mask, ids = inputs()
    config = resolve_loss_config(
        '{"ce":0.1,"ce_gt":0.2,"kl_div2":0.3,"tv":1.8}', "eager"
    )
    calls = dict.fromkeys(config, 0)

    def counted(name, fn):
        def run(*args):
            calls[name] += 1
            return fn(*args)

        return run

    wrapped = {name: (counted(name, fn), coef) for name, (fn, coef) in config.items()}

    def unexpected_tv(*args):
        pytest.fail("TV in loss_fn must be reused")

    shared = {}
    loss, _ = compute_metrics(
        q,
        p,
        c,
        mask,
        4,
        wrapped,
        tv_loss_fn=unexpected_tv,
        per_position_loss_weight="dpa-dev",
        dpa_dev_signal=signal,
        dpa_dev_alpha=0.3,
        data_token_ids=ids,
        shared_weights=shared,
    )
    assert calls == dict.fromkeys(config, 1)
    terms = {
        name: fn(q, torch.where(mask.bool(), ids, 0) if name == "ce_gt" else p)
        for name, (fn, _) in config.items()
    }
    prob = (
        1 - terms["tv"].detach()
        if signal == "accept"
        else terms[signal].detach().neg().exp()
    )
    weights = reference_weights(prob, mask)
    torch.testing.assert_close(shared["weights"], weights)
    assert not shared["weights"].requires_grad
    denominator = mask.sum() + 1e-5
    reference = sum(
        coef * (terms[name] * weights * mask).sum() / denominator
        for name, (_, coef) in config.items()
    )
    bce = functional.binary_cross_entropy_with_logits(
        c, (1 - terms["tv"]).detach(), reduction="none"
    )
    reference += (bce * mask * weights).sum() / (mask.sum() + 1e-8)
    torch.testing.assert_close(loss, reference)
    actual_grad = torch.autograd.grad(loss, (q, c), retain_graph=True)
    expected_grad = torch.autograd.grad(reference, (q, c))
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "model", [DSparkDraftModel, DFlashDraftModel, DFlash2DraftModel]
)
@pytest.mark.parametrize("signal", ["ce", "ce_gt", "kl_div2", "accept"])
def test_cli_and_trainer_roundtrip(model, signal):
    kind = {
        DSparkDraftModel: "dspark",
        DFlashDraftModel: "dflash",
        DFlash2DraftModel: "dflash2",
    }[model]
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "dummy",
            "--speculator-type",
            kind,
            "--loss-fn",
            '{"ce":0.1,"ce_gt":0.2,"kl_div2":0.3,"tv":1.8}',
            "--loss-implementation",
            "eager",
            "--per-position-loss-weight",
            "dpa-dev",
            "--dpa-dev-signal",
            signal,
            "--dpa-dev-alpha",
            "0.3",
        ]
    )
    assert TrainConfig.from_flat(cfg.flatten()).model_dump() == cfg.model_dump()
    for kwargs in model.get_trainer_kwargs(**cfg.flatten()):
        assert kwargs["dpa_dev_signal"] == signal
        assert kwargs["dpa_dev_alpha"] == pytest.approx(0.3)


@pytest.mark.parametrize("signal", ["ce", "ce_gt", "kl_div2", "kl_div"])
def test_reject_missing_or_unsupported_signal(signal):
    with pytest.raises(ValueError):
        TrainConfig.model_validate(
            {
                "speculator_type": "dspark",
                "loss": {"loss_fn": "tv"},
                "dflash": {
                    "per_position_loss_weight": "dpa-dev",
                    "dpa_dev_signal": signal,
                },
            }
        )


def test_kl_div2_formula_and_gradient():
    q, p, _, _, _ = inputs()
    value = eager.kl_div2_loss(q, p)
    expected = -(p.softmax(-1) * q.log_softmax(-1)).sum(-1)
    torch.testing.assert_close(value, expected)
    entropy = -(p.softmax(-1) * p.log_softmax(-1)).sum(-1)
    torch.testing.assert_close(value, eager.kl_div_loss(q, p) + entropy)
    torch.testing.assert_close(
        torch.autograd.grad(value.sum(), q)[0], (q.softmax(-1) - p.softmax(-1))
    )


def test_accept_without_tv_and_empty_mask():
    q, p, _, mask, _ = inputs()
    config = resolve_loss_config("kl_div2", "eager")
    _, w, accept = prepare_dpa_dev(q, p, mask * 0, 4, config, tv_loss_fn=eager.tv_loss)
    assert torch.count_nonzero(w) == 0
    torch.testing.assert_close(accept, 1 - eager.tv_loss(q, p))
    assert not accept.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_kl_div2_and_compile():
    q, p, _, _, _ = inputs("cuda")
    fused = resolve_loss_config("kl_div2")["kl_div2"][0]
    reference = eager.kl_div2_loss(q, p)
    for fn in (fused, torch.compile(fused, fullgraph=True)):
        value = fn(q, p)
        torch.testing.assert_close(value, reference)
        torch.testing.assert_close(
            torch.autograd.grad(value.sum(), q)[0], q.softmax(-1) - p.softmax(-1)
        )


@pytest.mark.parametrize("signal", ["ce", "ce_gt", "kl_div2", "accept"])
def test_dflash2_selector_uses_unary_weight(signal):
    q, p, _, mask, ids = inputs()
    candidates, positions, contains = selector_training_candidates(
        q.detach().topk(3, dim=-1).indices, p.argmax(-1)
    )
    selector = torch.randn(1, 8, 3, requires_grad=True)
    config = resolve_loss_config(
        '{"ce":0.1,"ce_gt":0.2,"kl_div2":0.3,"tv":1.8}', "eager"
    )
    shared = {}
    loss, _ = selector_metrics(
        q,
        p,
        candidates,
        selector,
        positions,
        contains,
        mask,
        4,
        3,
        loss_config=config,
        tv_loss_fn=eager.tv_loss,
        per_position_loss_weight="dpa-dev",
        dpa_dev_signal=signal,
        dpa_dev_alpha=0.3,
        data_token_ids=ids,
        dpa_dev_weights=shared,
    )
    unary, _ = compute_metrics(
        q,
        p,
        None,
        mask,
        4,
        config,
        tv_loss_fn=eager.tv_loss,
        per_position_loss_weight="dpa-dev",
        dpa_dev_signal=signal,
        dpa_dev_alpha=0.3,
        data_token_ids=ids,
        sample_from_anchor=False,
    )
    ce = functional.cross_entropy(
        selector.reshape(-1, 3), positions.flatten(), reduction="none"
    ).reshape(1, 8)
    expected = unary + (ce * mask * shared["weights"]).sum() / (mask.sum() + 1e-5)
    torch.testing.assert_close(loss, expected)
    for actual, reference in zip(
        torch.autograd.grad(loss, (q, selector), retain_graph=True),
        torch.autograd.grad(expected, (q, selector)),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("signal", ["ce", "ce_gt", "kl_div2", "accept"])
def test_fused_shared_fullgraph(signal):
    q, p, c, mask, ids = inputs("cuda")
    spec = '{"ce":0.1,"ce_gt":0.2,"kl_div2":0.3,"tv":1.8}'

    configs = {impl: resolve_loss_config(spec, impl) for impl in ("eager", "fused")}
    tv_fns = {impl: resolve_loss_config("tv", impl)["tv"][0] for impl in configs}

    def run(q, p, c, mask, ids, *, implementation):
        return compute_metrics(
            q,
            p,
            c,
            mask,
            4,
            configs[implementation],
            tv_loss_fn=tv_fns[implementation],
            per_position_loss_weight="dpa-dev",
            dpa_dev_signal=signal,
            dpa_dev_alpha=0.3,
            data_token_ids=ids,
        )[0]

    reference = run(q, p, c, mask, ids, implementation="eager")
    reference_grad = torch.autograd.grad(reference, (q, c))
    for fn in (run, torch.compile(run, fullgraph=True)):
        actual = fn(q, p, c, mask, ids, implementation="fused")
        torch.testing.assert_close(actual, reference)
        for grad, expected in zip(
            torch.autograd.grad(actual, (q, c)), reference_grad, strict=True
        ):
            torch.testing.assert_close(grad, expected, atol=1e-6, rtol=1e-5)


def test_hidden_mse_shared_weights():
    hidden = torch.arange(8.0).reshape(1, 4, 2).requires_grad_()
    teacher = torch.zeros_like(hidden)
    mask = torch.ones(1, 4)
    weights = torch.tensor([[0.7, 0.5, 0.2, 0.1]], requires_grad=True)
    loss = predicted_hidden_mse(
        hidden,
        teacher,
        torch.arange(4),
        mask,
        mask,
        torch.zeros(1, 4, dtype=torch.long),
        block_size=4,
        sample_from_anchor=False,
        position_weights=weights,
    )
    expected = (hidden.square().mean(-1) * weights.detach()).mean()
    torch.testing.assert_close(loss, expected)
    grad, weight_grad = torch.autograd.grad(loss, (hidden, weights), allow_unused=True)
    torch.testing.assert_close(
        grad, hidden.detach() * weights.detach().unsqueeze(-1) / 4
    )
    assert weight_grad is None


def test_dflash_accept_without_tv():
    q, p, _, mask, _ = inputs()
    config = resolve_loss_config("kl_div2", "eager")
    loss, _ = dflash_metrics(
        q,
        p,
        mask,
        4,
        loss_config=config,
        per_position_loss_weight="dpa_dev",
        dpa_dev_signal="accept",
        tv_loss_fn=eager.tv_loss,
    )
    weights = reference_weights(1 - eager.tv_loss(q, p), mask, alpha=0.5)
    expected = (eager.kl_div2_loss(q, p) * mask * weights).sum() / (mask.sum() + 1e-5)
    torch.testing.assert_close(loss, expected)
