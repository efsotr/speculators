"""Shared model accuracy metrics."""

import torch

from speculators.losses.real import real_branch_scores, trajectory_token_logs


@torch.no_grad()
def compute_real_branch_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    overlap: torch.Tensor,
    *,
    block_size: int,
    start_pos: int,
) -> dict[str, torch.Tensor]:
    """Count REAL branch position scores over the same blocks as REAL EAL."""
    if start_pos >= block_size:
        return {}

    valid, _, log_ratio = trajectory_token_logs(
        logits,
        targets,
        token_ids,
        loss_mask,
        block_size=block_size,
        start_pos=start_pos,
    )
    return compute_prob_prefix_counts(
        log_ratio,
        overlap.reshape(-1, block_size)[:, start_pos:],
        valid,
        start_pos=start_pos,
    )


@torch.no_grad()
def compute_prob_prefix_counts(
    log_ratio: torch.Tensor,
    overlap: torch.Tensor,
    valid: torch.Tensor,
    *,
    start_pos: int,
) -> dict[str, torch.Tensor]:
    """Count probability-weighted prefix acceptance from block-shaped inputs."""
    inclusive = torch.where(valid, log_ratio.clamp(max=0.0), 0.0).cumsum(-1)
    scores = real_branch_scores(inclusive, overlap, valid)
    position_sums = scores.sum(dim=0)
    block_total = valid[:, 0].sum().float()
    metrics = {
        "prob_eal_sum": position_sums.sum() + block_total,
        "prob_eal_total": block_total,
    }
    for offset, position_sum in enumerate(position_sums.unbind()):
        pos = start_pos + offset
        metrics[f"position_{pos}_prob_acc_sum"] = position_sum
        metrics[f"position_{pos}_prob_acc_total"] = block_total
    return metrics


@torch.no_grad()
def compute_greedy_prefix_counts(
    correct: torch.Tensor,  # shape: [num_blocks, num_draft_slots]
    valid: torch.Tensor,  # shape: [num_blocks, num_draft_slots]
    *,
    start_pos: int,
) -> dict[str, torch.Tensor]:
    """Count accepted prefixes and EAL, including one bonus token per valid block."""
    accepted = torch.logical_and(correct, valid).to(torch.float32)
    position_sums = accepted.cumprod(dim=-1).sum(dim=0)
    block_total = valid.any(dim=-1).sum().float()
    metrics = {
        "greedy_eal_sum": position_sums.sum() + block_total,
        "greedy_eal_total": block_total,
    }
    for offset, position_sum in enumerate(position_sums.unbind()):
        pos = start_pos + offset
        metrics[f"position_{pos}_greedy_acc_sum"] = position_sum
        metrics[f"position_{pos}_greedy_acc_total"] = block_total
    return metrics


def compute_accuracy_single_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor | None,  # shape: [1, seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, seq_len]
):
    """Compute full and conditional accuracy counts for a single speculative step.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: If provided, restricts accuracy to masked positions.
        prev_correct: Boolean mask of positions correct so far. Updated in place
            via logical AND with the current step's correctness.

    Returns:
        Tuple of (full_correct, full_total, cond_correct, cond_total) as raw
        counts suitable for distributed reduction before computing ratios.
    """
    correct = pred_ids == target_ids
    cond_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)
    if prev_correct is not None:
        cond_total = prev_correct.sum().float()
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))

    correct_sum = correct.float().sum()
    full_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)

    return correct_sum, full_total, correct_sum.clone(), cond_total


@torch.no_grad()
def compute_accuracy_multi_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    num_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-position correct/total counts across multiple speculative steps.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: Boolean mask selecting positions to evaluate.
        pos_idx: Position index within each speculative block (e.g. 0,1,2,3,0,1,2,3).
        num_pos: Number of distinct positions (i.e. block size).

    Returns:
        Tuple of (correct_per_pos, total_per_pos) both with shape [num_pos].
        Overall counts can be derived by summing these.
    """
    correct = pred_ids == target_ids
    # Reserve bucket 0 for masked entries; real position 0 starts at bucket 1.
    # Fixed-length scatter avoids masked_select's data-dependent output shape.
    valid = loss_mask.to(torch.bool)
    pos_idx = torch.where(valid, pos_idx + 1, 0).reshape(-1)
    correct = correct.reshape(-1)

    correct_per_pos = torch.zeros(num_pos + 1, dtype=torch.float, device=correct.device)
    total_per_pos = torch.zeros(num_pos + 1, dtype=torch.float, device=correct.device)
    correct_per_pos.scatter_add_(0, pos_idx, correct.float())
    total_per_pos.scatter_add_(0, pos_idx, torch.ones_like(correct, dtype=torch.float))

    return correct_per_pos[1:], total_per_pos[1:]  # shape: [num_pos], [num_pos]
