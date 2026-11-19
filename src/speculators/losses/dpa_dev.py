"""Shared detached DPA-dev weights and cached per-token loss terms."""

import torch
from torch._higher_order_ops.associative_scan import associative_scan

from speculators.losses.real import observed_log_probability
from speculators.losses.utils import (
    LossConfig,
    _dpace_probability_weights,
    dpace_dev_token_probs,
    tv_loss,
)


def prepare_dpa_dev(  # noqa: C901
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    loss_config: LossConfig,
    *,
    dpa_dev2: bool = False,
    dpa_dev_signal: str = "accept",
    dpa_dev_alpha: float = 0.5,
    dpa_dev_prefix_only: bool = False,
    data_token_ids: torch.Tensor | None = None,
    tv_loss_fn=tv_loss,
):
    """Compute each configured loss once, then detach before deriving weights."""
    if dpa_dev_signal == "acc":
        dpa_dev_signal = "accept"
    if dpa_dev2 and dpa_dev_signal not in {"q", "p", "q_draft", "p_draft", "accept"}:
        raise ValueError("dpa-dev2 requires q, p, q_draft, p_draft or acc")
    if not dpa_dev2 and dpa_dev_signal not in {
        "ce",
        "ce_gt",
        "kl_div2",
        "accept",
        "p_draft",
        "q_draft",
    }:
        raise ValueError(f"Unsupported dpa_dev_signal: {dpa_dev_signal}")
    ratio_signal = dpa_dev_signal in {"p_draft", "q_draft", "q", "p"}
    if dpa_dev_prefix_only and not ratio_signal and not dpa_dev2:
        raise ValueError("dpa_dev_prefix_only requires p_draft or q_draft")
    if ratio_signal and (
        data_token_ids is None or data_token_ids.shape != logits.shape[:-1]
    ):
        raise ValueError("p_draft/q_draft require aligned data_token_ids")
    if (
        not ratio_signal
        and dpa_dev_signal != "accept"
        and dpa_dev_signal not in loss_config
    ):
        raise ValueError(
            f"dpa_dev_signal={dpa_dev_signal} requires that loss in loss_fn"
        )
    if not (0.0 <= dpa_dev_alpha <= 1.0 if dpa_dev2 else 0.0 < dpa_dev_alpha <= 1.0):
        raise ValueError("dpa_dev_alpha must be in [0, 1] for dpa-dev2 or (0, 1] for dpa-dev")
    terms = {}
    for name, (fn, _) in loss_config.items():
        labels = targets
        if name == "ce_gt":
            if data_token_ids is None or data_token_ids.shape != logits.shape[:-1]:
                raise ValueError("ce_gt requires aligned data_token_ids")
            labels = torch.where(loss_mask.bool(), data_token_ids, 0)
        terms[name] = fn(logits, labels)
    with torch.no_grad():
        tv = (
            terms["tv"].detach()
            if "tv" in terms
            else tv_loss_fn(logits.detach(), targets.detach())
        )
        accept = (1.0 - tv).clamp(0.0, 1.0)
        if ratio_signal:
            ids = torch.where(loss_mask.bool(), data_token_ids, 0)
            if dpa_dev2 and dpa_dev_signal in {"p_draft", "q_draft"}:
                log_q = observed_log_probability(logits, ids)
                log_p = observed_log_probability(targets, ids)
                log_ratio = (
                    log_q - log_p if dpa_dev_signal == "p_draft" else log_p - log_q
                )
                probability = log_ratio.clamp(max=0.0).exp()
            else:
                q, p = dpace_dev_token_probs(logits, targets, ids)
            if dpa_dev_signal in {"q", "p"}:
                probability = q if dpa_dev_signal == "q" else p
            elif not dpa_dev2:
                numerator, denominator = (
                    (q, p) if dpa_dev_signal == "p_draft" else (p, q)
                )
                probability = (
                    numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
                ).clamp(max=1.0)
        else:
            probability = (
                accept
                if dpa_dev_signal == "accept"
                else terms[dpa_dev_signal].detach().float().neg().exp()
            )
        weights_fn = dpa_dev2_weights if dpa_dev2 else _dpace_probability_weights
        weights = weights_fn(
            probability,
            loss_mask,
            block_size,
            dpa_dev_alpha,
            prefix_only=dpa_dev_prefix_only,
        )
    return terms, weights, accept


def _compose_affine(left, right):
    """Compose f_right(f_left(x)) for the reverse scan traversal."""
    a, b = left
    c, d = right
    return a * c, d + c * b


@torch.no_grad()
def dpa_dev2_weights(probability, loss_mask, block_size, alpha, *, prefix_only=False):
    """Exclusive P_i and S_i, with masked positions skipped in each block."""
    if probability.shape[-1] % block_size:
        raise ValueError("Sequence length must be divisible by block_size")
    shape = probability.shape
    mask = loss_mask.reshape(-1, block_size).bool()
    prob = probability.detach().float().reshape(-1, block_size)
    smooth = alpha + (1.0 - alpha) * prob
    smooth = torch.where(mask, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(
        torch.cat((torch.ones_like(smooth[:, :1]), smooth[:, :-1]), dim=-1),
        dim=-1,
    )
    if prefix_only or block_size == 1:
        return (prefix * mask).reshape(shape)
    # Inclusive affine scan T_i=r_i*(m_i+T_{i+1}). Shift it to obtain
    # S_i=1+T_{i+1}=1+r_{i+1}+r_{i+1}r_{i+2}+... on valid positions.
    # At holes, m_i=0 and r_i=1, so neither a reward nor a factor is added.
    _, suffix = associative_scan(
        _compose_affine,
        (smooth, smooth * mask.float()),
        dim=1,
        reverse=True,
        combine_mode="pointwise",
    )
    suffix = torch.cat((suffix[:, 1:], torch.zeros_like(suffix[:, :1])), dim=-1)
    return (prefix * (1.0 + suffix) * mask).reshape(shape)
