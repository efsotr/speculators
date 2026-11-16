"""DPA-dev2 recurrence, signal, precision and compiled gradient checks."""

import pytest
import torch

from speculators.losses import resolve_loss_config
from speculators.losses.dpa_dev import dpa_dev2_weights, prepare_dpa_dev
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.core import DFlash2DraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.train.config import TrainConfig


@pytest.mark.parametrize("prefix_only", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("size", [1, 4, 16])
def test_scan_reference(prefix_only, alpha, size):
    torch._dynamo.reset()
    prob = torch.rand(2, size * 2, device="cuda", dtype=torch.bfloat16)
    mask = torch.randint(0, 2, prob.shape, device="cuda")
    expected = torch.zeros_like(prob, dtype=torch.float32)
    for row in range(2):
        for start in range(0, size * 2, size):
            valid = [i for i in range(start, start + size) if mask[row, i]]
            for k, i in enumerate(valid):
                prefix = 1.0
                for j in valid[:k]:
                    prefix *= alpha + (1 - alpha) * float(prob[row, j])
                suffix = 1.0
                product = 1.0
                if not prefix_only:
                    for j in valid[k + 1 :]:
                        product *= alpha + (1 - alpha) * float(prob[row, j])
                        suffix += product
                expected[row, i] = prefix * suffix
    for fn in (dpa_dev2_weights, torch.compile(dpa_dev2_weights, fullgraph=True)):
        actual = fn(prob, mask, size, alpha, prefix_only=prefix_only)
        torch.testing.assert_close(actual, expected)
        assert actual.dtype == torch.float32
        assert not actual.requires_grad


@pytest.mark.parametrize("signal", ["q", "p", "q_draft", "p_draft", "acc"])
@pytest.mark.parametrize("prefix_only", [False, True])
def test_signal_compile(signal, prefix_only):
    torch._dynamo.reset()
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "dummy",
            "--speculator-type",
            "dspark",
            "--loss-fn",
            "ce_gt",
            "--per-position-loss-weight",
            "dpa-dev2",
            "--dpa-dev-signal",
            signal,
            *(["--dpa-dev-prefix-only"] if prefix_only else []),
        ]
    )
    assert TrainConfig.from_flat(cfg.flatten()).model_dump() == cfg.model_dump()
    q = torch.randn(1, 8, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    p = torch.randn_like(q)
    mask = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 0]], device="cuda")
    ids = torch.randint(0, 17, (1, 8), device="cuda")
    config = resolve_loss_config('{"ce_gt":0.1,"tv":1.8}', "fused")

    def run(q, p, mask, ids):
        return compute_metrics(
            q,
            p,
            None,
            mask,
            4,
            config,
            per_position_loss_weight="dpa-dev2",
            dpa_dev_signal=signal,
            dpa_dev_prefix_only=prefix_only,
            data_token_ids=ids,
        )[0]

    expected = run(q, p, mask, ids)
    grad = torch.autograd.grad(expected, q)[0]
    actual = torch.compile(run, fullgraph=True)(q, p, mask, ids)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, q)[0], grad, atol=1e-3, rtol=1e-2
    )
    _, weights, _ = prepare_dpa_dev(
        q,
        p,
        mask,
        4,
        config,
        dpa_dev2=True,
        dpa_dev_signal=signal,
        dpa_dev_prefix_only=prefix_only,
        data_token_ids=ids,
    )
    qp = q.detach().float().softmax(-1).gather(-1, ids[..., None]).squeeze(-1)
    pp = p.float().softmax(-1).gather(-1, ids[..., None]).squeeze(-1)
    prob = {
        "q": qp,
        "p": pp,
        "q_draft": (pp / qp).clamp(max=1),
        "p_draft": (qp / pp).clamp(max=1),
        "acc": torch.minimum(q.detach().float().softmax(-1), p.float().softmax(-1)).sum(
            -1
        ),
    }[signal]
    torch.testing.assert_close(
        weights, dpa_dev2_weights(prob, mask, 4, 0.5, prefix_only=prefix_only)
    )


@pytest.mark.parametrize("kind", ["dspark", "dflash", "dflash2"])
@pytest.mark.parametrize("signal", ["q", "p", "q_draft", "p_draft", "acc"])
def test_trainer_roundtrip(kind, signal):

    model = {
        "dspark": DSparkDraftModel,
        "dflash": DFlashDraftModel,
        "dflash2": DFlash2DraftModel,
    }[kind]
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "dummy",
            "--speculator-type",
            kind,
            "--loss-fn",
            "ce",
            "--per-position-loss-weight",
            "dpa-dev2",
            "--dpa-dev-signal",
            signal,
            "--dpa-dev-prefix-only",
        ]
    )
    for kwargs in model.get_trainer_kwargs(**cfg.flatten()):
        assert kwargs["per_position_loss_weight"] == "dpa-dev2"
        assert kwargs["dpa_dev_signal"] == ("accept" if signal == "acc" else signal)
        assert kwargs["dpa_dev_prefix_only"]


def test_zero_smoothing_cpu():
    from unittest.mock import patch

    from speculators.losses.dpa_dev import _compose_affine

    # CPU reference for the GPU-only scan; exercise prepare and zero factors.
    def scan_reference(combine, values, dim, reverse, combine_mode):
        assert combine is _compose_affine and dim == 1 and reverse
        a, b = values
        out_a, out_b = a.clone(), b.clone()
        for i in range(a.shape[1] - 2, -1, -1):
            out_a[:, i], out_b[:, i] = combine(
                (out_a[:, i + 1], out_b[:, i + 1]), (a[:, i], b[:, i])
            )
        return out_a, out_b

    cfg = TrainConfig.resolve([
        "--verifier-name-or-path", "dummy", "--speculator-type", "dspark",
        "--per-position-loss-weight", "dpa-dev2", "--dpa-dev-alpha", "0",
    ])
    assert cfg.dflash.dpa_dev_alpha == 0
    with pytest.raises(SystemExit):
        TrainConfig.resolve([
            "--verifier-name-or-path", "dummy", "--speculator-type", "dspark",
            "--per-position-loss-weight", "dpa-dev", "--dpa-dev-alpha", "0",
        ])
    prob = torch.tensor([[0.5, 0.0, 0.8, 0.2]])
    with patch("speculators.losses.dpa_dev.associative_scan", scan_reference):
        actual = dpa_dev2_weights(prob, torch.ones_like(prob), 4, 0.0)
        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.98, 0.0, 0.0]]))
        logits = torch.randn(1, 4, 7, requires_grad=True)
        terms, weights, _ = prepare_dpa_dev(
            logits, torch.randn_like(logits), torch.ones(1, 4), 4,
            resolve_loss_config('{"ce_gt":0.1,"tv":1.8}', 'eager'),
            dpa_dev2=True, dpa_dev_signal="p_draft", dpa_dev_alpha=0.0,
            data_token_ids=torch.zeros(1, 4, dtype=torch.long),
        )
        assert torch.isfinite(weights).all() and not weights.requires_grad
        loss = sum((term * weights).sum() for term in terms.values())
        loss.backward()
        assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("signal", ["p_draft", "q_draft"])
@pytest.mark.parametrize("delta", [-2.0, 0.0, 2.0])
def test_ratio_log_space_underflow(signal, delta):
    q = torch.tensor([[[-1000.0 + delta, 0.0]]]).expand(1, 2, 2)
    p = torch.tensor([[[-1000.0, 0.0]]]).expand_as(q)
    _, weights, _ = prepare_dpa_dev(
        q, p, torch.ones(1, 2), 2, resolve_loss_config("ce", "eager"),
        dpa_dev2=True, dpa_dev_signal=signal, dpa_dev_alpha=0.0,
        dpa_dev_prefix_only=True, data_token_ids=torch.zeros(1, 2, dtype=torch.long),
        tv_loss_fn=lambda q, p: torch.zeros(q.shape[:-1]),
    )
    log_ratio = delta if signal == "p_draft" else -delta
    expected = torch.tensor([[1.0, min(log_ratio, 0.0)]] )
    expected[:, 1] = expected[:, 1].exp()
    torch.testing.assert_close(weights, expected)
