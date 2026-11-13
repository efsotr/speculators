"""Shared detached DPA-dev weights and cached per-token loss terms."""

import torch

from speculators.losses.utils import (
    LossConfig,
    _dpace_probability_weights,
    dpace_dev_token_probs,
    tv_loss,
)


def prepare_dpa_dev(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    loss_config: LossConfig,
    *,
    dpa_dev_signal: str = "accept",
    dpa_dev_alpha: float = 0.5,
    dpa_dev_prefix_only: bool = False,
    data_token_ids: torch.Tensor | None = None,
    tv_loss_fn=tv_loss,
):
    """Compute each configured loss once, then detach before deriving weights."""
    if dpa_dev_signal not in {"ce", "ce_gt", "kl_div2", "accept", "p_draft", "q_draft"}:
        raise ValueError(f"Unsupported dpa_dev_signal: {dpa_dev_signal}")
    ratio_signal = dpa_dev_signal in {"p_draft", "q_draft"}
    if dpa_dev_prefix_only and not ratio_signal:
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
    if not 0.0 < dpa_dev_alpha <= 1.0:
        raise ValueError("dpa_dev_alpha must be in (0, 1]")
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
            q, p = dpace_dev_token_probs(logits, targets, ids)
            numerator, denominator = (q, p) if dpa_dev_signal == "p_draft" else (p, q)
            probability = (
                numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
            ).clamp(max=1.0)
        else:
            probability = (
                accept
                if dpa_dev_signal == "accept"
                else terms[dpa_dev_signal].detach().float().neg().exp()
            )
        weights = _dpace_probability_weights(
            probability,
            loss_mask,
            block_size,
            dpa_dev_alpha,
            prefix_only=dpa_dev_prefix_only,
        )
    return terms, weights, accept
