"""Shared loss selection, weighting, and reduction utilities."""

import json
from collections.abc import Callable
from functools import partial

import torch

from speculators.losses import eager

_LOSS_REDUCTION_EPS = 1e-5

LossConfig = dict[
    str, tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]
]


def dflash_loss_decay(
    pos_idx: torch.Tensor, gamma: float, sample_from_anchor: bool = False, **_kwargs
):
    """Compute DFlash-style exponential decay weights per position.

    When ``sample_from_anchor`` is False (DFlash), position 0 is the anchor and
    gets weight 0; position 1 gets weight 1, and subsequent positions decay as
    ``exp(-(pos - 1) / gamma)``.

    When ``sample_from_anchor`` is True (DSpark), position 0 is the first real
    prediction and gets weight 1; subsequent positions decay as
    ``exp(-pos / gamma)``.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Decay rate (higher = slower decay).
        sample_from_anchor: Whether position 0 is a prediction (DSpark) or the
            anchor input (DFlash).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    if sample_from_anchor:
        # DSpark: slot 0 predicts the token after the anchor, so decay from 0.
        return torch.exp(-pos_idx / gamma)
    # DFlash: slot 0 is the anchor (no prediction); decay from slot 1.
    # pos_idx = 0 1 2 3 0 1 2 3, block_size = 4
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    # decay_mult = e^-(0 0 1 2 0 0 1 2) / gamma
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    # w = 0 1 e^-1/gamma e^-2/gamma 0 1 e^-1/gamma e^-2/gamma
    return decay_mult  # noqa: RET504


def exp_loss_decay(pos_idx: torch.Tensor, gamma: float, **_kwargs):
    """Compute simple exponential decay weights as gamma^pos_idx.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Base of the exponent (typically in (0, 1]).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    return gamma**pos_idx


def dpace_loss_decay(
    pos_idx: torch.Tensor,  # noqa: ARG001
    loss_mask: torch.Tensor,
    block_size: int,
    dpace_alpha: float,
    elementwise_loss: torch.Tensor,
    **_kwargs,
):
    """
    Per-position block-drafting loss weight based on D-PACE

    Args:
        elementwise_loss: requires to be cross-entropy loss, negative log-likelihood
            of per-position confidence
        dpace_alpha: confidence smoothing constant

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    with torch.no_grad():
        # convert CE to per-position confidence
        q = torch.exp(-elementwise_loss).float()

        return _dpace_probability_weights(q, loss_mask, block_size, dpace_alpha)


def _dpace_probability_weights(
    q: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    dpace_alpha: float,
    *,
    prefix_only: bool = False,
) -> torch.Tensor:
    # reshape loss to [num_anchors, block_size]
    # for intra-block cumulative multiplication
    if q.shape[1] % block_size != 0:
        raise ValueError(
            f"q.shape[1] ({q.shape[1]}) must be divisible by block_size ({block_size})"
        )
    num_anchors = q.shape[1] // block_size
    q = q.reshape(num_anchors, block_size)
    mask = loss_mask.reshape(num_anchors, block_size).to(q.dtype)

    # smoothed confidence for numerical stability
    smooth = (1.0 - dpace_alpha) * q + dpace_alpha
    smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))

    if prefix_only:
        exclusive = torch.cat((torch.ones_like(smooth[:, :1]), smooth[:, :-1]), dim=-1)
        return (torch.cumprod(exclusive, dim=-1) * mask).reshape(1, -1)

    # prefix cumulative production
    prefix = torch.cumprod(smooth, dim=-1)

    # suffix summation: flip -> cumsum -> flip
    weight = torch.flip(
        torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1]
    )
    weight = weight * mask

    # reshape weight
    return weight.reshape(1, -1)


@torch.no_grad()
def dpace_dev_token_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draft and teacher probabilities at supplied IDs (default: teacher argmax).

    Reduce in fp32 without retaining a full softmax tensor or an autograd graph.
    """
    ids = (targets.argmax(dim=-1) if token_ids is None else token_ids).unsqueeze(-1)
    q = (
        logits.gather(-1, ids).squeeze(-1).float() - logits.float().logsumexp(dim=-1)
    ).exp()
    p = (
        targets.gather(-1, ids).squeeze(-1).float() - targets.float().logsumexp(dim=-1)
    ).exp()
    return q, p


@torch.no_grad()
def dpace_dev_loss_decay(  # noqa: PLR0917
    pos_idx: torch.Tensor,  # noqa: ARG001
    loss_mask: torch.Tensor,
    block_size: int,
    dpace_alpha: float,
    q: torch.Tensor,
    p: torch.Tensor,
    accept_prob: str = "q",
    overlap: torch.Tensor | None = None,
    **_kwargs,
) -> torch.Tensor:
    """D-PACE weights with explicit, detached acceptance probabilities.

    q and p refer to the same selected token. Selector q is recomputed
    from its corrected K-way logits; p remains the full-vocabulary teacher mass.
    Ratios clamp the denominator to fp32 tiny (including 0/0 -> 0).
    """
    q, p = q.float(), p.float()
    tiny = torch.finfo(torch.float32).tiny
    if accept_prob == "tv":
        if overlap is None:
            raise ValueError("tv acceptance requires per-position sum(min(p_i, q_i))")
        accept = overlap.float().clamp(0.0, 1.0)
    elif accept_prob == "q":
        accept = q
    elif accept_prob == "p":
        accept = p
    elif accept_prob == "min_q_p":
        accept = torch.minimum(q, p)
    elif accept_prob == "p_over_q":
        accept = (p / q.clamp_min(tiny)).clamp(max=1.0)
    elif accept_prob == "q_over_p":
        accept = (q / p.clamp_min(tiny)).clamp(max=1.0)
    else:
        raise ValueError(f"Unknown dpace_dev accept probability: {accept_prob}")
    return _dpace_probability_weights(accept, loss_mask, block_size, dpace_alpha)


def _fused_kernel(name: str):
    """Import a fused loss by name."""
    try:
        from speculators.losses import fused as mod  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "Fused losses require Triton; use --loss-implementation eager for "
            "compatibility testing."
        ) from error
    return getattr(mod, name)


def kl_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_kl_div_loss")(logits, targets)


def kl_div2_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_kl_div2_loss")(logits, targets)


def reverse_kl_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_reverse_kl_div_loss")(logits, targets)


def js_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_js_div_loss")(logits, targets)


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_ce_loss")(logits, targets)


def ce_gt_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_ce_gt_loss")(logits, target_ids)


def tv_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_tv_loss")(logits, targets)


def neg_log_acceptance_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return _fused_kernel("fused_nla_loss")(logits, targets)


def lk_hybrid_loss(
    logits: torch.Tensor, targets: torch.Tensor, eta: float = 3.0
) -> torch.Tensor:
    return _fused_kernel("fused_lk_hybrid_loss")(logits, targets, eta=eta)


_FUSED_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": kl_div_loss,
    "kl_div2": kl_div2_loss,
    "rkl": reverse_kl_div_loss,
    "jsd": js_div_loss,
    "ce": ce_loss,
    "ce_gt": ce_gt_loss,
    "tv": tv_loss,
    "nla": neg_log_acceptance_loss,
    "lk_hybrid": lk_hybrid_loss,
}


_EAGER_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": eager.kl_div_loss,
    "kl_div2": eager.kl_div2_loss,
    "rkl": eager.reverse_kl_div_loss,
    "jsd": eager.js_div_loss,
    "ce": eager.ce_loss,
    "ce_gt": eager.ce_gt_loss,
    "tv": eager.tv_loss,
    "nla": eager.neg_log_acceptance_loss,
    "lk_hybrid": eager.lk_hybrid_loss,
}


_LOSS_FN_MAPS = {"fused": _FUSED_LOSS_FN_MAP, "eager": _EAGER_LOSS_FN_MAP}


def resolve_loss_config(spec: str, implementation: str = "fused") -> LossConfig:
    """Parse a loss spec into ``{name: (loss_fn, weight)}``.

    Accepts either a plain loss name (``"kl_div"``) or a JSON dict mapping
    loss names to weights (``'{"ce": 0.1, "tv": 0.9}'``).
    """
    try:
        loss_fn_map = _LOSS_FN_MAPS[implementation]
    except KeyError:
        raise ValueError(
            f"Unknown loss implementation '{implementation}'. Choose from: "
            f"{sorted(_LOSS_FN_MAPS)}"
        ) from None

    if spec == "ce_tmx":
        spec = "ce"
    if spec in loss_fn_map:
        return {spec: (loss_fn_map[spec], 1.0)}

    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError:
        raise ValueError(
            f"Unknown loss function '{spec}'. Pass a known name "
            f"({sorted(loss_fn_map)}) or a JSON dict, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'.'
        ) from None

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            "Loss config must be a non-empty JSON dict mapping loss names to weights, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'. Got: {spec}'
        )

    config: LossConfig = {}
    for raw_name, weight in parsed.items():
        name = "ce" if raw_name == "ce_tmx" else raw_name
        if name not in loss_fn_map:
            raise ValueError(
                f"Unknown loss function '{name}' in loss config. "
                f"Choose from: {sorted(loss_fn_map)}"
            )
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Loss weight for '{name}' must be a number, "
                f"got {type(weight).__name__}"
            )
        previous_weight = config[name][1] if name in config else 0.0
        config[name] = (loss_fn_map[name], previous_weight + float(weight))

    return config


def compound_loss(  # noqa: PLR0917
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[..., torch.Tensor] | None = None,
    *,
    data_token_ids: torch.Tensor | None = None,
    elementwise_losses: dict[str, torch.Tensor] | None = None,
    loss_normalization: str = "weight_sum",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a weighted sum of loss terms.

    Each entry in *loss_config* maps a name to ``(loss_fn, weight)``; the
    result is ``sum(weight * loss_function(logits, targets, ..., loss_fn))``
    over all entries.

    Returns the total loss and a dict of per-term (unweighted) scalar losses
    keyed as ``"{name}_loss"``.  When the config contains a single term the
    dict is empty (the overall loss already captures it).
    """
    total = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    term_losses: dict[str, torch.Tensor] = {}
    multi = len(loss_config) > 1
    base_decay_fn = decay_fn
    while isinstance(base_decay_fn, partial):
        base_decay_fn = base_decay_fn.func
    if multi and base_decay_fn is dpace_loss_decay:
        raise ValueError("Compound loss does not support loss-dependent dpace weights")
    reduction = None
    for name, (fn, weight) in loss_config.items():
        term_targets = targets
        if name == "ce_gt":
            if data_token_ids is None:
                raise ValueError("ce_gt requires aligned data_token_ids")
            if data_token_ids.shape != logits.shape[:-1]:
                raise ValueError("ce_gt data_token_ids must match logits positions")
            # Masked positions may contain padding/sentinel IDs. Never gather them.
            term_targets = torch.where(loss_mask.bool(), data_token_ids, 0)
        elementwise = (
            fn(logits, term_targets)
            if elementwise_losses is None
            else elementwise_losses[name]
        )
        if reduction is None:
            reduction = _prepare_loss_reduction(
                elementwise, loss_mask, pos_idx, decay_fn, loss_normalization
            )
        total = total + weight * elementwise
        if multi:
            term_losses[f"{name}_loss"] = _reduce_loss(
                elementwise.detach(), reduction, loss_normalization
            ).detach()
    if reduction is not None:
        total = _reduce_loss(total, reduction, loss_normalization)
    return total, term_losses


def loss_function(  # noqa: PLR0917
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[..., torch.Tensor] | None = None,
    *,
    elementwise_loss: torch.Tensor | None = None,
    loss_normalization: str = "weight_sum",
):
    """Compute masked, optionally position-decayed training loss.

    Args:
        logits: Draft model logits.
        targets: Teacher logits, or aligned dataset token IDs for ce_gt.
        loss_mask: Boolean mask selecting positions to include in the loss.
        pos_idx: Position indices within each speculative block.
        loss_fn: Per-position loss function (default: kl_div_loss).
        decay_fn: Optional position-dependent decay weighting function.

    Returns:
        Scalar mean loss across the batch.
    """
    if elementwise_loss is None:
        elementwise_loss = loss_fn(logits, targets)  # shape: [1, seq_len]

    reduction = _prepare_loss_reduction(
        elementwise_loss, loss_mask, pos_idx, decay_fn, loss_normalization
    )
    return _reduce_loss(elementwise_loss, reduction, loss_normalization)


def _prepare_loss_reduction(
    elementwise_loss, loss_mask, pos_idx, decay_fn, normalization
):
    """Prepare shared mask, multiplier and detached normalization once."""
    mask = loss_mask.to(elementwise_loss.dtype)
    multiplier = None
    effective_weight = mask.detach().float()
    if decay_fn is not None:
        multiplier = decay_fn(
            pos_idx.to(elementwise_loss.dtype), elementwise_loss=elementwise_loss * mask
        )
        effective_weight = effective_weight * multiplier.detach().float()
    if normalization == "weight_sum":
        denominator = effective_weight.sum().detach()
        world_size = 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            torch.distributed.all_reduce(denominator)
            denominator = denominator / world_size
        denominator = denominator + _LOSS_REDUCTION_EPS / world_size
    elif normalization == "token_count":
        denominator = mask.sum(dim=1) + _LOSS_REDUCTION_EPS
    else:
        raise ValueError(f"Unknown loss_normalization: {normalization}")
    return mask, multiplier, denominator


def _reduce_loss(elementwise_loss, reduction, normalization):
    mask, multiplier, denominator = reduction
    weighted_loss = elementwise_loss * mask.to(elementwise_loss.dtype)
    if multiplier is not None:
        weighted_loss = weighted_loss * multiplier
    if normalization == "weight_sum":
        return weighted_loss.sum() / denominator
    return (weighted_loss.sum(dim=1) / denominator).mean()
