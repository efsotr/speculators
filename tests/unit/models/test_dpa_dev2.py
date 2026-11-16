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
@pytest.mark.parametrize("alpha", [0.3, 1.0])
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
