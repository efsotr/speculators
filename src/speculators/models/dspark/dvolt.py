"""Sampled-prefix candidate graph and continuation values for D-VOLT."""

import torch
from torch.nn.attention.flex_attention import create_block_mask

_create_block_mask = torch.compile(create_block_mask)


def candidate_probs(logits, candidates):
    """Evaluate selected tokens under a full-vocabulary softmax."""
    logits = logits.float()
    return (logits.gather(-1, candidates) - logits.logsumexp(-1, keepdim=True)).exp()


def continuation_values(overlaps, terminal):
    """Back up [anchor, predecessor, successor] overlap matrices."""
    values = [torch.zeros_like(terminal, dtype=torch.float32)]
    for overlap in reversed(overlaps):
        values.append((overlap * (1 + values[-1].unsqueeze(-2))).sum(-1))
    return torch.stack(values[::-1], dim=1)


@torch.compiler.disable
@torch.no_grad()
def build_candidate_graph(
    base_logits,
    anchor_tokens,
    markov_head,
    *,
    top_k,
    beta,
    score_branches,
    project_target,
    anchor_hidden,
    valid,
    chunk_size=128,
):
    """Return sampled predecessors, teacher logits and detached candidate values."""
    num_blocks, block_size, vocab_size = base_logits.shape
    prev = anchor_tokens
    predecessors, candidates, sampled_indices = [], [], []
    for step in range(block_size):
        predecessors.append(prev)
        path_logits = base_logits[:, step] + markov_head.markov_w2(
            markov_head.prev_embeddings(prev)
        )
        scores, ids = path_logits.topk(top_k, dim=-1)
        sampled = torch.multinomial(scores.float().softmax(-1), 1)
        candidates.append(ids)
        sampled_indices.append(sampled.squeeze(-1))
        prev = ids.gather(-1, sampled).squeeze(-1)

    active = valid.nonzero(as_tuple=True)[0]
    candidate_ids = torch.stack(candidates, dim=1)
    sampled = torch.stack(sampled_indices, dim=1)
    teacher_hidden = anchor_hidden.new_zeros(
        num_blocks, block_size, anchor_hidden.shape[-1]
    )
    teacher_hidden[:, 0] = anchor_hidden
    values = torch.zeros_like(candidate_ids, dtype=torch.float32)

    if block_size > 1 and active.numel():
        branch_ids = candidate_ids[active, :-1]
        branch_choices = sampled[active, :-1]
        branch_hidden = score_branches(active, branch_ids, branch_choices)
        rows = torch.arange(active.numel(), device=active.device)[:, None]
        steps = torch.arange(block_size - 1, device=active.device)[None, :]
        teacher_hidden[active, 1:] = branch_hidden[rows, steps, branch_choices]
        if beta:
            overlaps = []
            for step in range(block_size - 1):
                next_ids = candidate_ids[active, step + 1]
                overlap_chunks = []
                anchors_per_chunk = max(chunk_size // top_k, 1)
                for start in range(0, active.numel(), anchors_per_chunk):
                    sl = slice(start, start + anchors_per_chunk)
                    successors = next_ids[sl, None, :].expand(-1, top_k, -1)
                    target = candidate_probs(
                        project_target(branch_hidden[sl, step]), successors
                    )
                    draft = candidate_probs(
                        base_logits[active[sl], step + 1, None, :]
                        + markov_head.markov_w2(
                            markov_head.prev_embeddings(branch_ids[sl, step])
                        ),
                        successors,
                    )
                    overlap_chunks.append(torch.minimum(target, draft))
                overlaps.append(torch.cat(overlap_chunks))
            values[active] = continuation_values(overlaps, candidate_ids[active, -1])

    targets = project_target(teacher_hidden.flatten(0, 1)).view(
        1, num_blocks * block_size, vocab_size
    )
    return torch.stack(predecessors, dim=1), targets, candidate_ids, values


@torch.no_grad()
def score_qwen3_branches(
    model,
    input_ids,
    document_ids,
    position_ids,
    *,
    anchors,
    candidates,
    choices,
):
    """Evaluate all candidate nodes with one shared original-history prefill."""
    num_anchors, steps, top_k = candidates.shape
    prefix_length = input_ids.shape[-1]
    device = input_ids.device
    length = prefix_length + candidates.numel()

    branch_anchor = anchors.repeat_interleave(steps * top_k)
    branch_group = torch.arange(num_anchors, device=device).repeat_interleave(
        steps * top_k
    )
    branch_step = (
        torch.arange(steps, device=device).repeat_interleave(top_k).repeat(num_anchors)
    )
    sampled = choices.reshape(-1, 1) == torch.arange(top_k, device=device)
    sampled = sampled.flatten()
    docs = document_ids.flatten()
    tokens = torch.cat((input_ids, candidates.reshape(1, -1)), dim=1)
    branch_positions = position_ids[0, branch_anchor] + branch_step + 1
    positions = torch.cat((position_ids, branch_positions.unsqueeze(0)), dim=1)

    def mask_mod(_batch, _head, query, key):
        q_node = (query - prefix_length).clamp(0, candidates.numel() - 1)
        k_node = (key - prefix_length).clamp(0, candidates.numel() - 1)
        q_anchor = torch.where(query < prefix_length, query, branch_anchor[q_node])
        q_anchor = q_anchor.clamp(max=prefix_length - 1)
        k_original = key.clamp(max=prefix_length - 1)
        original = (
            (key < prefix_length)
            & (key <= q_anchor)
            & (docs[q_anchor] == docs[k_original])
        )
        earlier_sample = (
            (query >= prefix_length)
            & (key >= prefix_length)
            & (branch_group[q_node] == branch_group[k_node])
            & (branch_step[k_node] < branch_step[q_node])
            & sampled[k_node]
        )
        return original | earlier_sample | (query == key)

    if device.type == "cuda":
        mask = _create_block_mask(mask_mod, 1, None, length, length, device=device)
        implementation = "flex_attention"
    else:
        indices = torch.arange(length, device=device)
        mask = mask_mod(None, None, indices[:, None], indices[None, :])[None, None]
        implementation = "sdpa"

    hidden = model.embed_tokens(tokens)
    rotary = model.rotary_emb(hidden, positions)
    # Qwen3's decoder layers dispatch through the shared Transformers config.
    previous_impl = model.config._attn_implementation  # noqa: SLF001
    model.config._attn_implementation = implementation  # noqa: SLF001
    try:
        for layer in model.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_embeddings=rotary,
                use_cache=False,
            )
    finally:
        model.config._attn_implementation = previous_impl  # noqa: SLF001
    # Keep the prenorm contract used by the DFlash-family verifier projection.
    return hidden[:, prefix_length:].reshape(num_anchors, steps, top_k, -1)
