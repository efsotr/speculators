"""Static anchor selection on cached target rollouts.

Flatness and EDSD scores are migrated to anchor selection; these strategies do
not reproduce the papers' sample filtering or token loss masking objectives.
"""

import math

import torch

from .anchor_scoring import score_chunk


STATIC_STRATEGIES = frozenset(
    {"uniform", "target_flatness", "target_entropy_high", "target_entropy_low", "entropy_curriculum"}
)


def valid_anchor_mask(loss_mask, document_ids, block_size, sample_from_anchor=False):
    """Return legal starts with a fully supervised, document-local block.

    Both layouts first predict a+1: bonus-anchor blocks train B-1 targets,
    whereas sample-from-anchor blocks train B. Keep the historical exclusion
    of the final B packed positions even when fewer targets would fit.
    """
    if loss_mask.ndim != 2 or loss_mask.shape[0] != 1:
        raise ValueError("loss_mask must have shape [1, T]")
    if document_ids.shape != loss_mask.shape:
        raise ValueError("document_ids must have the same [1, T] shape as loss_mask")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    mask = loss_mask[0].bool()
    docs = document_ids[0]
    length = mask.numel()
    valid = mask & docs.ge(0)
    valid[max(0, length - block_size):] = False
    targets = block_size if sample_from_anchor else block_size - 1
    if targets == 0:
        return torch.zeros_like(valid)
    for offset in range(1, targets + 1):
        if offset >= length:
            return torch.zeros_like(valid)
        valid[:-offset] &= mask[offset:] & (docs[:-offset] == docs[offset:])
        valid[-offset:] = False
    return valid


@torch.no_grad()
def score_target_positions(last_hidden, positions, norm, head, strategy, chunk_size, backend="auto"):
    """Score h[a]'s raw next-token distribution using float32 reductions."""
    if strategy not in STATIC_STRATEGIES - {"uniform"}:
        raise ValueError(f"No target scorer for strategy {strategy!r}")
    if chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive")
    if last_hidden.ndim != 3 or last_hidden.shape[0] != 1:
        raise ValueError("verifier_last_hidden_states must have shape [1, T, H]")
    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if norm is None or head is None:
        raise ValueError("Target scoring requires verifier norm and LM head")
    chunks = []
    for start in range(0, positions.numel(), chunk_size):
        score = score_chunk(last_hidden, positions[start:start + chunk_size], norm,
                            head, strategy == "target_flatness", backend)
        if not torch.isfinite(score).all():
            raise ValueError("Target anchor scores must be finite")
        chunks.append(score)
    if not chunks:
        return torch.empty(0, dtype=torch.float32, device=last_hidden.device)
    return torch.cat(chunks)


def select_static_anchors(
    loss_mask,
    num_anchors,
    block_size,
    document_ids,
    *,
    sampling=None,
    verifier_last_hidden_states=None,
    verifier_norm=None,
    verifier_lm_head=None,
    epoch=0,
    num_epochs=1,
    sample_from_anchor=False,
    generator=None,
):
    """Filter scored candidates, then sample without replacement.

    A supplied persistent generator controls tie-breaking and selection. When
    absent, torch's global RNG is used; this function never reseeds per batch.
    """
    sampling = sampling or {}
    strategy = sampling.get("strategy", "uniform")
    if strategy not in STATIC_STRATEGIES:
        raise ValueError(f"Unknown static anchor strategy {strategy!r}")
    if num_anchors < 0:
        raise ValueError("num_anchors must be nonnegative")
    retain = sampling.get("retain_fraction", 1.0)
    if not 0 < retain <= 1:
        raise ValueError("retain_fraction must be in (0, 1]")
    preset = sampling.get("curriculum_preset", "paper")
    if preset not in {"paper", "github"}:
        raise ValueError("curriculum_preset must be paper or github")
    if strategy in {"uniform", "entropy_curriculum"} and retain != 1:
        raise ValueError(f"retain_fraction does not apply to {strategy}")
    if strategy != "entropy_curriculum" and preset != "paper":
        raise ValueError("curriculum_preset only applies to entropy_curriculum")
    chunk_size = sampling.get("score_chunk_size", 1024)
    if chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive")
    drop = 0.0
    if strategy == "entropy_curriculum":
        if num_epochs < 1 or not 0 <= epoch < num_epochs:
            raise ValueError("epoch must be in [0, num_epochs)")
        if preset == "paper":
            if (num_epochs - 1) * 0.03 >= 1:
                raise ValueError("paper curriculum initial drop ratio must be below 1")
            drop = max(0, num_epochs - epoch - 1) * 0.03
        else:
            drop = min(max(0, num_epochs - epoch - 1) * 0.02, 0.4)
    device = loss_mask.device
    anchors = torch.zeros(num_anchors, dtype=torch.long, device=device)
    valid = torch.zeros(num_anchors, dtype=torch.bool, device=device)
    candidates = valid_anchor_mask(
        loss_mask, document_ids, block_size, sample_from_anchor
    ).nonzero(as_tuple=True)[0]
    if candidates.numel() == 0 or num_anchors == 0:
        return anchors, valid
    pool_size = candidates.numel()
    if strategy != "uniform" and not (strategy == "entropy_curriculum" and drop == 0):
        if verifier_last_hidden_states is None:
            raise ValueError(f"{strategy} requires cached verifier_last_hidden_states")
        if verifier_last_hidden_states.shape[:2] != loss_mask.shape:
            raise ValueError("Target hidden sequence must align with loss_mask")
        scores = score_target_positions(
            verifier_last_hidden_states, candidates, verifier_norm,
            verifier_lm_head, strategy, chunk_size, sampling.get("score_backend", "triton"),
        )
        pool_size = (
            pool_size - math.floor(drop * pool_size)
            if strategy == "entropy_curriculum"
            else math.ceil(pool_size * retain)
        )
        # Stable sort after a random permutation gives exact, unbiased ties.
        perm = torch.randperm(candidates.numel(), device=device, generator=generator)
        descending = strategy not in {"target_entropy_low", "entropy_curriculum"}
        order = torch.argsort(scores[perm], descending=descending, stable=True)
        candidates = candidates[perm[order[:pool_size]]]
    count = min(num_anchors, pool_size)
    perm = torch.randperm(candidates.numel(), device=device, generator=generator)
    anchors[:count] = candidates[perm[:count]].sort().values
    valid[:count] = True
    return anchors, valid
