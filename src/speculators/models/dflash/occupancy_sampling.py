"""Conditional local-horizon round occupancy for fixed offline token sequences.

The recurrences and capped-PPS design follow EDR, arXiv:2610.10411v1,
Eqs. (3), (5), (27)-(28). Reference implementation:
y-x-zhao/AngelSpec-EDR, commit 0a31a8774a1484ed4af2aeb5ee70c77a4d026360,
angelspec/models/ops/edr.py. This module does not implement the EDR loss.
"""

from collections.abc import Callable

import torch


def _realized_log_probabilities(
    logits: torch.Tensor, token_ids: torch.Tensor, policy: dict
) -> torch.Tensor:
    """Gather log probabilities without underflowing positive softmax support."""
    mode = policy.get("mode")
    if mode == "greedy":
        return torch.where(logits.argmax(-1).eq(token_ids), 0.0, -torch.inf).double()
    if mode != "softmax":
        raise ValueError("sampling policy mode must be greedy or softmax")
    required = {"temperature", "top_k", "top_p"}
    if not required.issubset(policy):
        raise ValueError("softmax policy requires temperature, top_k and top_p")
    temperature, top_k, top_p = (
        policy["temperature"], policy["top_k"], policy["top_p"]
    )
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    values = logits.float() / temperature
    if top_k and top_k < values.shape[-1]:
        threshold = values.topk(top_k, dim=-1).values[..., -1:]
        values = values.masked_fill(values < threshold, -torch.inf)
    if top_p < 1:
        sorted_values, order = values.sort(descending=True, dim=-1)
        cumulative = sorted_values.softmax(-1).cumsum(-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_values = sorted_values.masked_fill(remove, -torch.inf)
        values = torch.empty_like(values).scatter(-1, order, sorted_values)
    log_probabilities = values.log_softmax(-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    if bool((torch.isnan(log_probabilities) | torch.isposinf(log_probabilities)).any()):
        raise ValueError("sampling logits do not define finite probabilities")
    return log_probabilities.double()


def _realized_probabilities(
    logits: torch.Tensor, token_ids: torch.Tensor, policy: dict
) -> torch.Tensor:
    """Probability convenience API; exponentiate only after promotion to FP64."""
    return _realized_log_probabilities(logits, token_ids, policy).exp()


def round_start_occupancy(acceptance: torch.Tensor) -> torch.Tensor:
    """Return round-start masses for a finite horizon with B proposals per round.

    acceptance[n,j] is min(1, q[n,n+j+1](x)/p[n+j+1](x)).
    The final positional boundary is unoptimized; exhaustion commits a bonus
    token and hence advances by B+1. Only rows before the boundary are returned.
    """
    if acceptance.ndim != 2 or acceptance.shape[1] < 1:
        raise ValueError("acceptance must have shape [horizon_length, proposals>0]")
    probabilities = acceptance.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(probabilities).all()) or bool(
        ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("acceptance probabilities must be finite and in [0, 1]")
    length, width = probabilities.shape
    masses = torch.zeros(length + 1, dtype=torch.float64)
    masses[0] = 1
    for prefix in range(length):
        mass = masses[prefix].clone()
        for offset in range(min(width, length - prefix)):
            position = prefix + offset + 1
            accept = probabilities[prefix, offset]
            masses[position] += mass * (1 - accept)
            mass = mass * accept
        exhaustion = prefix + width + 1
        if exhaustion <= length:
            masses[exhaustion] += mass
    return masses[:-1]


@torch.no_grad()
def build_occupancy_scores(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    block_size: int,
    *,
    sample_from_anchor: bool,
    proposal: dict,
    target: dict,
    score_chunk_size: int,
    scan_draft: Callable[[torch.Tensor], torch.Tensor],
    verifier_last_hidden_states: torch.Tensor,
    verifier_norm: Callable,
    verifier_lm_head: Callable,
) -> torch.Tensor:
    """Score existing input anchors; no rollout is generated.

    Each contiguous supervised span has a fresh initial round. Its last token
    is a positional terminal boundary, matching the reference implementation's
    local horizon convention. For [s,e), prefix n maps to input anchor s-1+n.
    If s=0 or its predecessor belongs to another document, token s becomes
    committed context and the horizon starts at s+1. This explicit local reset
    does not claim the occupancy of the original complete response.

    The scanner must return the actual declared DFlash proposal logits, with
    padded queries allowed at the tail. It must not return training-only target
    injected candidates or substitute DFlash2 unary logits for its proposal.
    score_chunk_size bounds projected/query tokens, not the number of anchors;
    one indivisible block is the minimum draft chunk.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("occupancy sampling expects a single packed row [1,T]")
    if loss_mask.shape != input_ids.shape or document_ids.shape != input_ids.shape:
        raise ValueError("token, supervision and document shapes must match")
    proposals = block_size if sample_from_anchor else block_size - 1
    if proposals < 1 or score_chunk_size < 1:
        raise ValueError("occupancy requires positive proposal width and chunk size")
    length = input_ids.shape[1]
    scores = torch.zeros(length, device=input_ids.device, dtype=torch.float64)
    supervised = loss_mask[0].bool().cpu().tolist()
    docs = document_ids[0].cpu().tolist()
    cursor = 0
    while cursor < length:
        if not supervised[cursor] or docs[cursor] < 0:
            cursor += 1
            continue
        start = cursor
        doc = docs[start]
        cursor += 1
        while cursor < length and supervised[cursor] and docs[cursor] == doc:
            cursor += 1
        end = cursor
        if start == 0 or docs[start - 1] != doc:
            start += 1
        ordinary_length = end - start - 1
        if ordinary_length <= 0:
            continue
        all_anchors = torch.arange(
            start - 1, start - 1 + ordinary_length, device=input_ids.device
        )
        # Every realized target token is shared by overlapping anchor blocks.
        # Project it once, in token-sized chunks rather than [anchors,B,V].
        target_log_probabilities = torch.empty(
            ordinary_length, device=input_ids.device, dtype=torch.float64
        )
        for target_start in range(0, ordinary_length, score_chunk_size):
            positions = torch.arange(
                start + target_start,
                start + min(target_start + score_chunk_size, ordinary_length),
                device=input_ids.device,
            )
            target_logits = verifier_lm_head(
                verifier_norm(verifier_last_hidden_states[0, positions - 1])
            )
            token_log_probabilities = _realized_log_probabilities(
                target_logits, input_ids[0, positions], target
            )
            if bool(torch.isneginf(token_log_probabilities).any()):
                raise ValueError("realized rollout token has zero probability under target policy")
            target_log_probabilities[target_start : target_start + positions.numel()] = token_log_probabilities
            del target_logits
        acceptance = torch.zeros((ordinary_length, proposals), dtype=torch.float64)
        anchor_chunk_size = max(1, score_chunk_size // block_size)
        for chunk_start in range(0, ordinary_length, anchor_chunk_size):
            anchors = all_anchors[chunk_start : chunk_start + anchor_chunk_size]
            logits = scan_draft(anchors)
            if logits.ndim != 3 or logits.shape[:2] != (anchors.numel(), block_size):
                raise ValueError("draft scanner must return [anchors,block_size,vocab] logits")
            offsets = torch.arange(proposals, device=anchors.device)
            token_positions = anchors[:, None] + 1 + offsets[None, :]
            valid = token_positions < end - 1
            safe_positions = token_positions.clamp(max=end - 2)
            ids = input_ids[0, safe_positions]
            draft_logits = logits if sample_from_anchor else logits[:, 1:]
            draft_log_prob = _realized_log_probabilities(draft_logits, ids, proposal)
            target_log_prob = target_log_probabilities[safe_positions - start]
            log_ratio = draft_log_prob - target_log_prob
            accept = torch.where(valid, log_ratio.clamp(max=0).exp(), 0)
            acceptance[chunk_start : chunk_start + anchors.numel()] = accept.cpu()
        scores[all_anchors] = round_start_occupancy(acceptance).to(scores.device)
    return scores


def select_occupancy_anchors(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    num_anchors: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capped-PPS systematic sampling, with selected marginal probabilities.

    Returns sorted unique anchors, valid padding mask and selected inclusion
    probabilities, all [num_anchors]. Does not apply loss correction; that is
    determined by the objective, not by the sampling algorithm.
    """
    if scores.ndim != 1 or valid_mask.numel() != scores.numel():
        raise ValueError("scores and valid_mask must describe the same token positions")
    if num_anchors < 0:
        raise ValueError("num_anchors must be nonnegative")
    weights = scores.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("occupancy scores must be finite and nonnegative")
    eligible = torch.nonzero(valid_mask.detach().cpu().reshape(-1).bool() & (weights > 0)).flatten()
    selected_count = min(num_anchors, eligible.numel())
    anchors = torch.zeros(num_anchors, device=scores.device, dtype=torch.long)
    valid = torch.zeros(num_anchors, device=scores.device, dtype=torch.bool)
    selected_probabilities = torch.zeros(num_anchors, device=scores.device, dtype=torch.float64)
    if selected_count == 0:
        return anchors, valid, selected_probabilities
    candidate_weights = weights[eligible]
    if selected_count == eligible.numel():
        inclusion = torch.ones_like(candidate_weights)
        selected = torch.arange(eligible.numel())
    else:
        ordered = candidate_weights.sort(descending=True).values
        inverse_scale = None
        for saturated in range(selected_count):
            scale = ordered[saturated:].sum() / (selected_count - saturated)
            if bool(ordered[saturated] <= scale):
                inverse_scale = scale
                break
        if inverse_scale is None or not bool(inverse_scale > 0):
            raise RuntimeError("failed to solve capped-PPS inclusion probabilities")
        inclusion = (candidate_weights / inverse_scale).clamp(max=1)
        cumulative = inclusion.cumsum(0)
        cumulative[-1] = max(float(cumulative[-1]), selected_count)
        random_device = generator.device if generator is not None else torch.device("cpu")
        random_start = torch.rand((), generator=generator, device=random_device, dtype=torch.float64).cpu()
        offsets = torch.arange(selected_count, dtype=torch.float64)
        thresholds = torch.minimum(random_start + offsets, torch.nextafter(offsets + 1, offsets))
        selected = torch.searchsorted(cumulative, thresholds, right=True)
        if selected.unique().numel() != selected_count or bool((selected >= eligible.numel()).any()):
            raise RuntimeError("systematic PPS did not produce distinct valid anchors")
    anchors[:selected_count] = eligible[selected].to(scores.device)
    valid[:selected_count] = True
    selected_probabilities[:selected_count] = inclusion[selected].to(scores.device)
    return anchors, valid, selected_probabilities
