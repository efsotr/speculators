"""Target-path objectives from Tian et al., arXiv:2611.14399."""

import torch

TRAJECTORY_LOSSES = {"REAL", "raw_path"}


@torch.compile(dynamic=True)
def observed_log_probability(logits, ids):
    return logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1).float() - (
        logits.float().logsumexp(-1)
    )


def trajectory_token_logs(
    logits, targets, token_ids, loss_mask, *, block_size, start_pos
):
    """Return prefix validity and observed-token log probabilities by block."""
    valid = loss_mask.reshape(-1, block_size)[:, start_pos:].bool().cumprod(-1).bool()
    ids = token_ids.reshape(-1, block_size)[:, start_pos:]
    ids = torch.where(valid, ids, 0)
    vocab = logits.shape[-1]
    q = logits.reshape(-1, block_size, vocab)[:, start_pos:]
    p = targets.reshape(-1, block_size, vocab)[:, start_pos:]
    log_q = observed_log_probability(q, ids)
    log_ratio = log_q - observed_log_probability(p, ids)
    return valid, log_q, log_ratio


def real_branch_scores(inclusive, overlap, valid):
    """Multiply exclusive prefix survival by the current slot overlap."""
    exclusive = torch.cat((torch.zeros_like(inclusive[:, :1]), inclusive[:, :-1]), -1)
    return torch.where(valid, exclusive.exp() * overlap, 0.0)


def trajectory_loss(
    logits,
    targets,
    token_ids,
    loss_mask,
    *,
    block_size,
    start_pos,
    tv_loss_fn,
    raw_path,
    ce_weight,
    ce_name="ce_gt",
    ce_position_weights=None,
    tv_cache=None,
):
    """Compute trajectory losses over the valid prefix of each block."""
    if token_ids is None or token_ids.shape != logits.shape[:-1]:
        raise ValueError("Trajectory losses require aligned data_token_ids")
    if logits.shape != targets.shape:
        raise ValueError("Trajectory losses require a shared target/draft vocabulary")
    window = block_size - start_pos
    if not 0 <= start_pos < block_size:
        raise ValueError("Trajectory losses require at least one prediction slot")

    targets = targets.detach()
    valid, log_q, log_ratio = trajectory_token_logs(
        logits,
        targets,
        token_ids,
        loss_mask,
        block_size=block_size,
        start_pos=start_pos,
    )
    log_b = torch.where(valid, log_ratio.clamp(max=0.0), 0.0)
    inclusive = log_b.cumsum(-1)
    if raw_path:
        denominator = valid.any(-1).sum().clamp_min(1) * window
        with torch.no_grad():
            survival = torch.where(valid, inclusive.exp(), 0.0)
            # Sum of all survival terms containing b_i equals W_i * b_i.
            weights = survival.flip(-1).cumsum(-1).flip(-1)
            weights = torch.where(valid & (log_ratio < 0), weights, 0.0)
        weighted_ce = (weights * torch.where(valid, -log_q, 0.0)).sum() / denominator
        objective = valid.any().float() - survival.sum() / denominator
        loss = objective + (weighted_ce - weighted_ce.detach())
        return loss, {"raw_path_loss": objective.detach()}

    tv = tv_loss_fn(logits, targets)
    if tv_cache is not None:
        tv_cache["tv"] = tv.detach()
    overlap = 1.0 - tv.reshape(-1, block_size)[:, start_pos:]
    branch = real_branch_scores(inclusive, overlap, valid)
    denominator = valid.sum().clamp_min(1)
    objective = torch.where(valid, 1.0 - branch, 0.0).sum() / denominator
    if ce_name == "ce":
        vocab = logits.shape[-1]
        q = logits.reshape(-1, block_size, vocab)[:, start_pos:]
        p = targets.reshape(-1, block_size, vocab)[:, start_pos:]
        log_q = observed_log_probability(q, p.argmax(dim=-1))
    ce = torch.where(valid, -log_q, 0.0)
    if ce_position_weights is not None:
        ce = ce * ce_position_weights.reshape(-1, block_size)[:, start_pos:]
    ce = ce.sum() / denominator
    return (1.0 - ce_weight) * objective + ce_weight * ce, {
        "real_branch_loss": objective.detach(),
        f"{ce_name}_loss": ce.detach(),
    }
