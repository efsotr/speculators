"""Hidden-state supervision at the token positions predicted by draft slots."""

import torch


def predicted_hidden_mse(
    hidden: torch.Tensor,
    predicted_hidden_states: torch.Tensor,
    block_indices: torch.Tensor,
    aligned_loss_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    *,
    block_size: int,
    sample_from_anchor: bool,
) -> torch.Tensor:
    """Mean over supervised tokens and channels, with detached teacher targets.

    Slot j predicts token a+j in bonus-anchor mode, or a+j+1 when sampling
    from the anchor. Never wrap a next-token target around the packed sequence
    or into the next document. Invalid/padded blocks inherit zero weight.
    Both hidden tensors must be after their respective final model norms.
    """
    length = predicted_hidden_states.shape[1]
    target_indices = block_indices + int(sample_from_anchor)
    safe_indices = target_indices.clamp(max=length - 1)
    anchors = block_indices.view(-1, block_size)[:, :1].expand(-1, block_size)
    anchor_docs = document_ids[:, anchors.reshape(-1)]
    valid = (target_indices < length).unsqueeze(0)
    valid = valid & (document_ids[:, safe_indices] == anchor_docs)
    weights = aligned_loss_mask.float() * loss_mask[:, safe_indices].float()
    weights = weights * valid
    targets = predicted_hidden_states.detach()[:, safe_indices]
    # Mask before squaring so excluded nonfinite targets cannot poison gradients.
    difference = torch.where(
        (weights > 0).unsqueeze(-1), hidden.float() - targets.float(), 0.0
    )
    squared_error = difference.square().mean(dim=-1)
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)
