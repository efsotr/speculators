import math

import pytest
import torch
from torch.nn import functional

from speculators.losses import compound_loss, resolve_loss_config
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash2.metrics import compute_metrics as dflash2_metrics
from speculators.models.dspark.metrics import compute_metrics as dspark_metrics
from speculators.train.config import TrainConfig


@pytest.mark.parametrize(
    "spec",
    [
        '{"REAL": 0.8}',
        '{"REAL": 0.8, "ce_gt": 0.3}',
        '{"REAL": 0.8, "ce_gt": 0.2000000000001}',
        '{"REAL": 0.8, "ce_gt": 0.2, "tv": 0}',
        '{"REAL": -0.1, "ce_gt": 1.1}',
        '{"REAL": NaN, "ce_gt": 0}',
        '{"REAL": 0, "ce_gt": true}',
    ],
)
def test_invalid_real_weights(spec):
    with pytest.raises(ValueError, match="REAL|ce_gt"):
        resolve_loss_config(spec)


@pytest.mark.parametrize("kind", ["dflash", "dflash2", "dspark"])
def test_real_schema(kind):
    cfg = TrainConfig.model_validate(
        {
            "speculator_type": kind,
            "loss": {"loss_fn": '{"REAL": 0.7, "ce_gt": 0.3}'},
        }
    )
    assert cfg.loss.loss_normalization == "token_count"
    assert resolve_loss_config(cfg.loss.loss_fn)["ce_gt"][1] == 0.3
    assert resolve_loss_config("REAL")["ce_gt"][1] == 0.1


def test_real_schema_rejects_weight_sum():
    with pytest.raises(ValueError, match="REAL requires loss_normalization=token_count"):
        TrainConfig.model_validate(
            {
                "speculator_type": "dspark",
                "loss": {"loss_fn": "REAL", "loss_normalization": "weight_sum"},
            }
        )


def test_real_schema_rejects_unaligned_model():
    with pytest.raises(ValueError, match="Trajectory losses require"):
        TrainConfig.model_validate(
            {
                "speculator_type": "eagle3",
                "loss": {"loss_fn": '{"REAL": 0.7, "ce_gt": 0.3}'},
            }
        )


def reference_loss(logits, targets, ids, mask, start, gamma):
    draft, teacher = logits[0].softmax(-1), targets[0].detach().softmax(-1)
    rewards = []
    ce_losses = []
    for block in range(0, logits.shape[1], 4):
        survival = 1.0
        for slot in range(block + start, block + 4):
            if not mask[0, slot]:
                break
            token = ids[0, slot]
            rewards.append(survival * torch.minimum(draft[slot], teacher[slot]).sum())
            ratio = draft[slot, token] / teacher[slot, token]
            survival = survival * ratio.clamp(max=1.0)
            ce_losses.append(
                functional.cross_entropy(
                    logits[0, slot : slot + 1], ids[0, slot : slot + 1]
                )
                * math.exp(-(slot - block - start) / gamma)
            )
    zero = logits.sum() * 0
    if not rewards:
        return zero, zero
    return (1 - torch.stack(rewards)).mean(), torch.stack(ce_losses).mean()


@pytest.mark.parametrize(
    ("implementation", "start", "ce_weight", "compiled", "empty"),
    [
        ("eager", 0, 0.0, False, False),
        ("eager", 1, 1.0, False, False),
        ("eager", 0, 0.3, False, True),
        ("eager", 1, 0.3, True, False),
        ("fused", 0, 0.3, True, False),
        ("fused", 1, 0.3, True, False),
    ],
)
def test_real_value_gradient_and_compile(
    implementation, start, ce_weight, compiled, empty
):
    if implementation == "fused" and not torch.cuda.is_available():
        pytest.skip("fused TV requires CUDA")
    device = "cuda" if implementation == "fused" else "cpu"
    generator = torch.Generator(device=device).manual_seed(2026)
    logits = torch.randn(
        1, 12, 19, generator=generator, device=device, requires_grad=True
    )
    targets = torch.randn(
        1, 12, 19, generator=generator, device=device, requires_grad=True
    )
    ids = torch.tensor(
        [[1, 2, 3, -100, 4, 5, -100, -100, -100, -100, -100, -100]], device=device
    )
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]], device=device)
    if empty:
        mask.zero_()
    config = resolve_loss_config(
        f'{{"REAL": {1 - ce_weight}, "ce_gt": {ce_weight}}}', implementation
    )
    tv_fn = resolve_loss_config("tv", implementation)["tv"][0]
    pos = torch.arange(12, device=device).reshape(1, -1) % 4
    gamma = 2.5

    def run(q, p, token_ids, loss_mask):
        return compound_loss(
            q,
            p,
            loss_mask,
            pos,
            config,
            data_token_ids=token_ids,
            block_size=4,
            start_pos=start,
            tv_loss_fn=tv_fn,
            gamma=gamma,
        )

    actual, terms = (torch.compile(run, fullgraph=True) if compiled else run)(
        logits,
        targets,
        ids,
        mask,
    )
    branch, ce = reference_loss(logits, targets, ids, mask, start, gamma)
    expected = (1 - ce_weight) * branch + ce_weight * ce
    torch.testing.assert_close(terms["real_branch_loss"], branch)
    torch.testing.assert_close(terms["ce_gt_loss"], ce)
    torch.testing.assert_close(actual, expected)
    actual_grad, target_grad = torch.autograd.grad(
        actual, (logits, targets), allow_unused=True
    )
    assert target_grad is None
    (expected_grad,) = torch.autograd.grad(expected, (logits,))
    torch.testing.assert_close(actual_grad, expected_grad)

    metric_loss, _ = compute_metrics(
        logits,
        targets,
        mask,
        4,
        loss_config=config,
        tv_loss_fn=tv_fn,
        data_token_ids=ids,
        sample_from_anchor=start == 0,
        gamma=gamma,
    )
    torch.testing.assert_close(metric_loss, expected)


@pytest.mark.parametrize("kind", ["dspark", "dflash2"])
def test_real_model_metrics(kind):
    logits = torch.tensor([[[1.0, -0.2], [0.5, 0.1], [0.2, 2.0], [0.0, 0.2]]])
    targets = -logits
    token_ids = torch.tensor([[0, 1, 0, 1]])
    mask = torch.ones_like(token_ids)
    config = resolve_loss_config('{"REAL": 0.8, "ce_gt": 0.2}', "eager")
    tv_fn = resolve_loss_config("tv", "eager")["tv"][0]
    gamma = 2.5
    kwargs = {
        "loss_config": config,
        "tv_loss_fn": tv_fn,
        "data_token_ids": token_ids,
        "real_loss_mask": mask,
        "loss_normalization": "token_count",
        "gamma": gamma,
        "per_position_loss_weight": "dpace",
    }
    start = int(kind == "dflash2")
    targets_at_slots = targets.argmax(-1)
    weights = torch.exp(-torch.arange(4 - start) / gamma)
    if kind == "dspark":
        confidence = logits[..., 0]
        loss, metrics = dspark_metrics(logits, targets, confidence, mask, 4, **kwargs)
        auxiliary = functional.binary_cross_entropy_with_logits(
            confidence, 1.0 - tv_fn(logits, targets), reduction="none"
        )
    else:
        candidates = torch.arange(2).expand_as(logits)
        loss, metrics = dflash2_metrics(
            logits,
            targets,
            candidates,
            logits,
            targets_at_slots,
            mask.bool(),
            mask,
            4,
            2,
            **kwargs,
        )
        auxiliary = functional.cross_entropy(
            logits.transpose(1, 2), targets_at_slots, reduction="none"
        )
    branch, ce = reference_loss(logits, targets, token_ids, mask, start, gamma)
    auxiliary = (auxiliary[:, start:] * weights).sum() / mask.sum()
    torch.testing.assert_close(loss, 0.8 * branch + 0.2 * ce + auxiliary)
    real_eal = metrics["real_branch_eal_sum"] / metrics["real_branch_eal_total"]
    torch.testing.assert_close(real_eal, 1.0 + (1.0 - branch) * (4 - start))
