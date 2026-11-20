"""Check the sampled-history mask and Bellman backup against direct references."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from speculators.models.dspark.dvolt import (
    build_candidate_graph,
    candidate_probs,
    score_qwen3_branches,
)
from speculators.models.dspark.model_definitions import MarkovHead


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_candidate_nodes_match_standalone_histories(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cfg = Qwen3Config(
        vocab_size=16,
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = Qwen3Model(cfg).eval().to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    docs = torch.tensor([[0, 0, 0, 1, 1, 1]], device=device)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]], device=device)
    anchors = torch.tensor([1, 4, 3], device=device)
    candidates = torch.tensor(
        [[[7, 8], [9, 10]], [[9, 8], [11, 12]], [[6, 7], [11, 10]]],
        device=device,
    )
    choices = torch.tensor([[1, 0], [0, 1], [1, 1]], device=device)
    actual = score_qwen3_branches(
        model,
        ids,
        docs,
        positions,
        anchors=anchors,
        candidates=candidates,
        choices=choices,
    )
    for n, (start, end) in enumerate([(0, 2), (3, 5), (3, 4)]):
        history = ids[:, start:end]
        for step in range(2):
            for k in range(2):
                branch = torch.cat((history, candidates[n, step, k].view(1, 1)), -1)
                with torch.no_grad():
                    expected = model(branch, use_cache=False).last_hidden_state[0, -1]
                    normalized = model.norm(actual[n, step, k])
                atol = 0.025 if device == "cuda" else 2e-6
                torch.testing.assert_close(normalized, expected, atol=atol, rtol=atol)
            sampled = candidates[n, step, choices[n, step]].view(1, 1)
            history = torch.cat((history, sampled), -1)


@pytest.mark.parametrize("beta", [0.0, 0.3])
def test_candidate_graph_matches_scalar_backup(beta):
    torch.manual_seed(7)
    vocab = 6
    markov = MarkovHead(
        verifier_vocab_size=vocab,
        draft_vocab_size=vocab,
        markov_rank=2,
        hidden_size=2,
        head_type="vanilla",
    )
    base = torch.randn(2, 3, vocab, requires_grad=True)
    initial = torch.randn(2, vocab)
    teacher = torch.randn(1, 2, 2, vocab)

    def score(active, ids, choices):
        assert active.tolist() == [0]
        assert ids.shape == (1, 2, 2)
        assert choices.shape == (1, 2)
        return teacher

    prev, targets, candidates, values = build_candidate_graph(
        base,
        torch.tensor([1, 2]),
        markov,
        top_k=2,
        beta=beta,
        score_branches=score,
        project_target=lambda hidden: hidden,
        anchor_hidden=initial,
        valid=torch.tensor([True, False]),
    )
    assert not values.requires_grad
    assert not targets.requires_grad
    assert not values[1].any()
    assert not values[:, -1].any()
    expected = torch.zeros_like(values[0])
    with torch.no_grad():
        for step in reversed(range(2)):
            for k, token in enumerate(candidates[0, step]):
                p = teacher[0, step, k].softmax(-1)
                q = (
                    base[0, step + 1] + markov.markov_w2(markov.prev_embeddings(token))
                ).softmax(-1)
                for index, u in enumerate(candidates[0, step + 1]):
                    expected[step, k] += min(p[u], q[u]) * (
                        1 + expected[step + 1, index]
                    )
    torch.testing.assert_close(
        values[0], expected if beta else torch.zeros_like(expected)
    )
    target_blocks = targets.reshape(2, 3, vocab)
    for step in range(2):
        k = (candidates[0, step] == prev[0, step + 1]).nonzero().item()
        torch.testing.assert_close(target_blocks[0, step + 1], teacher[0, step, k])
    logits = base + markov.markov_w2(markov.prev_embeddings(prev))
    q = candidate_probs(logits, candidates)
    torch.testing.assert_close(q, logits.softmax(-1).gather(-1, candidates))
    p = candidate_probs(target_blocks, candidates)
    l1 = (logits.float().softmax(-1) - target_blocks.float().softmax(-1)).abs().sum(-1)
    actual_loss = (l1 - 2 * beta * (torch.minimum(p, q) * values).sum(-1))[0].mean()
    dense_q = logits.softmax(-1)
    dense_p = target_blocks.softmax(-1)
    credit = torch.ones_like(dense_q[0]).scatter(-1, candidates[0], 1 + beta * expected)
    reference_loss = (
        2 - 2 * (torch.minimum(dense_p[0], dense_q[0]) * credit).sum(-1).mean()
    )
    torch.testing.assert_close(actual_loss, reference_loss)
    (ref_grad,) = torch.autograd.grad(reference_loss, base, retain_graph=True)
    actual_loss.backward()
    assert base.grad is not None
    torch.testing.assert_close(base.grad, ref_grad)
    assert markov.markov_w2.weight.grad is not None
