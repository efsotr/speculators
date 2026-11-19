"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_dev_loss_decay,
    dpace_dev_token_probs,
    dpace_loss_decay,
    loss_function,
    tv_loss,
)
from speculators.losses.dpa_dev import prepare_dpa_dev
from speculators.losses.real import TRAJECTORY_LOSSES, observed_log_probability
from speculators.models.metrics import (
    compute_accuracy_multi_step,
    compute_greedy_prefix_counts,
    compute_real_branch_counts,
)

__all__ = [
    "compute_metrics",
]


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[..., torch.Tensor] | None,
    loss_normalization: str = "weight_sum",
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    if loss_normalization == "token_count":
        mask = loss_mask.to(elementwise.dtype)
        weight = (
            1.0
            if decay_fn is None
            else decay_fn(pos_idx.to(elementwise.dtype), elementwise_loss=elementwise)
        )
        return ((elementwise * mask * weight).sum(1) / (mask.sum(1) + 1e-8)).mean()
    return loss_function(
        elementwise,
        elementwise,
        loss_mask,
        pos_idx,
        decay_fn=decay_fn,
        elementwise_loss=elementwise,
        loss_normalization=loss_normalization,
    )


@torch.compiler.disable
def sample_vat_uniform(shape, device, seed=None):
    generator = torch.Generator(device=device)
    generator.manual_seed(torch.initial_seed() if seed is None else seed)
    return torch.rand(shape, device=device, dtype=torch.float32, generator=generator)


@torch.no_grad()
def _vat_weights(logits, targets, data_token_ids, loss_mask, *, block_size, gamma, uniform):
    """Sample acceptance on aligned data tokens and decay after the first rejection."""
    if data_token_ids is None or data_token_ids.shape != logits.shape[:-1]:
        raise ValueError("VAT requires aligned data_token_ids")
    valid = loss_mask.bool()
    ids = torch.where(valid, data_token_ids, 0)
    log_q = observed_log_probability(logits, ids)
    log_p = observed_log_probability(targets, ids)
    acceptance = (log_q - log_p).clamp(max=0.0).exp()
    if uniform is None:
        uniform = sample_vat_uniform(acceptance.shape, acceptance.device)
    rejected = (valid & (uniform >= acceptance)).reshape(-1, block_size)
    positions = torch.arange(block_size, device=logits.device)
    first_rejection = torch.where(rejected, positions, block_size).amin(
        dim=-1, keepdim=True
    )
    distance = (positions - first_rejection).clamp_min(0)
    decay = dflash_loss_decay(distance, gamma=gamma, sample_from_anchor=True)
    return decay.reshape_as(loss_mask) * valid


def compute_metrics(  # noqa: C901
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    loss_normalization: str = "weight_sum",
    per_position_loss_weight: str = "fixed-exp-decay",
    dpa_dev_signal: str = "accept",
    dpa_dev_alpha: float = 0.5,
    dpa_dev_prefix_only: bool = False,
    dpace_alpha: float = 0.5,
    dpace_dev_accept_prob: str = "q",
    sample_from_anchor: bool = True,
    dpace_token_source: str = "teacher_argmax",  # noqa: S107
    data_token_ids: torch.Tensor | None = None,
    shared_weights: dict | None = None,
    compute_prob_metrics: bool = False,
    vat_uniform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    start_pos = 0 if sample_from_anchor else 1
    trajectory = bool(TRAJECTORY_LOSSES.intersection(loss_config))
    elementwise_losses = None
    accept_rate = None
    tv_cache = {}
    if trajectory:
        decay_fn = None
    elif per_position_loss_weight in {"dpa_dev", "dpa-dev", "dpa_dev2", "dpa-dev2"}:
        elementwise_losses, weights, accept_rate = prepare_dpa_dev(
            logits,
            targets,
            loss_mask,
            block_size,
            loss_config,
            dpa_dev2=per_position_loss_weight in {"dpa_dev2", "dpa-dev2"},
            dpa_dev_signal=dpa_dev_signal,
            dpa_dev_alpha=dpa_dev_alpha,
            dpa_dev_prefix_only=dpa_dev_prefix_only,
            data_token_ids=data_token_ids,
            tv_loss_fn=tv_loss_fn,
        )

        def decay_fn(*_args, **_kwargs):
            return weights

        if shared_weights is not None:
            shared_weights["weights"] = weights
    elif per_position_loss_weight == "vat":
        weights = _vat_weights(
            logits, targets, data_token_ids, loss_mask,
            block_size=block_size, gamma=gamma, uniform=vat_uniform,
        )

        def decay_fn(*_args, **_kwargs):
            return weights

    elif per_position_loss_weight == "dpace_dev":
        if dpace_token_source == "data" and data_token_ids is None:  # noqa: S105
            raise ValueError("dpace_token_source=data requires aligned data_token_ids")
        q, p = dpace_dev_token_probs(
            logits,
            targets,
            data_token_ids if dpace_token_source == "data" else None,  # noqa: S105
        )
        if dpace_dev_accept_prob == "tv":
            with torch.no_grad():
                accept_rate = 1.0 - tv_loss_fn(logits, targets)
        decay_fn = partial(
            dpace_dev_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
            q=q,
            p=p,
            accept_prob=dpace_dev_accept_prob,
            overlap=accept_rate,
        )
    elif per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        loss_normalization=loss_normalization,
        elementwise_losses=elementwise_losses,
        block_size=block_size,
        start_pos=0 if sample_from_anchor else 1,
        tv_loss_fn=tv_loss_fn,
        gamma=gamma,
        data_token_ids=data_token_ids,
        tv_cache=tv_cache,
    )

    num_blocks = seq_len // block_size
    draft_mask = loss_mask.reshape(num_blocks, block_size)[:, start_pos:]
    has_prob_metrics = (
        compute_prob_metrics and data_token_ids is not None and start_pos < block_size
    )
    if (confidence_logits is not None or has_prob_metrics) and accept_rate is None:
        with torch.no_grad():
            tv = tv_cache.get("tv")
            if tv is None:
                tv = tv_loss_fn(logits, targets)
            accept_rate = 1.0 - tv

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        # Legacy D-PACE keeps fixed confidence decay; DPA-dev and VAT share weights.
        confidence_decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )
        conf_loss = _masked_decayed_mean(
            bce,
            loss_mask,
            pos_idx,
            (
                decay_fn
                if "REAL" not in loss_config
                and per_position_loss_weight
                in {"dpa_dev", "dpa-dev", "dpa_dev2", "dpa-dev2", "vat"}
                else confidence_decay_fn
            ),
            loss_normalization=loss_normalization,
        )
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total.clone()
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            draft_mask = draft_mask.to(accept_rate.dtype)
            accept_prefix = (
                accept_rate.reshape(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, start_pos:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Per-position greedy accuracy
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["greedy_independ_eal_sum"] = correct_per_pos[start_pos:].sum()
    metrics["greedy_independ_eal_total"] = total_per_pos[start_pos:].sum()
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_greedy_independ_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_greedy_independ_acc_total"] = total_per_pos[pos]

    metrics.update(
        compute_greedy_prefix_counts(
            (pred_ids == target_ids).reshape(num_blocks, block_size)[:, start_pos:],
            draft_mask.to(torch.bool),
            start_pos=start_pos,
        )
    )

    if has_prob_metrics:
        metrics.update(
            compute_real_branch_counts(
                logits,
                targets,
                data_token_ids,
                loss_mask,
                accept_rate,
                block_size=block_size,
                start_pos=start_pos,
            )
        )
    return loss, metrics
