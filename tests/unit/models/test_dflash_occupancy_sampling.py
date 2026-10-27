"""Conditional occupancy and capped-PPS checks independent of the backbone."""

import pytest
import torch

from speculators.models.dflash.occupancy_sampling import (
    _realized_probabilities,
    build_occupancy_scores,
    round_start_occupancy,
    select_occupancy_anchors,
)


def _enumerated_round_starts(acceptance):
    length, width = acceptance.shape
    result = torch.zeros(length, dtype=torch.float64)

    def visit_round(prefix, probability):
        if prefix >= length:
            return
        result[prefix] += probability
        survival = probability
        for offset in range(min(width, length - prefix)):
            position = prefix + offset + 1
            accept = float(acceptance[prefix, offset])
            visit_round(position, survival * (1 - accept))
            survival *= accept
        visit_round(prefix + width + 1, survival)

    visit_round(0, 1.0)
    return result


@pytest.mark.parametrize("width", [1, 2, 4])
def test_dp_matches_enumerated_accept_reject_paths(width):
    acceptance = torch.rand(7, width, generator=torch.Generator().manual_seed(13))
    torch.testing.assert_close(
        round_start_occupancy(acceptance), _enumerated_round_starts(acceptance)
    )


def test_bonus_advances_one_additional_token():
    actual = round_start_occupancy(torch.ones(8, 2))
    torch.testing.assert_close(actual, torch.tensor([1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float64))
    torch.testing.assert_close(round_start_occupancy(torch.zeros(8, 2)), torch.ones(8, dtype=torch.float64))


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_scanner_alignment_horizons_and_tail(sample_from_anchor):
    # Prompt at 0, answer at 1..6. Final answer token 6 is positional boundary.
    ids = torch.zeros((1, 7), dtype=torch.long)
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1]])
    docs = torch.zeros_like(ids)
    block_size = 3
    calls = []

    def scanner(anchors):
        calls.extend(anchors.tolist())
        logits = torch.zeros(anchors.numel(), block_size, 2)
        # The input anchor slot in bonus mode is NOT a proposal.
        if not sample_from_anchor:
            logits[:, 0, 1] = 20
        return logits

    result = build_occupancy_scores(
        ids, mask, docs, block_size,
        sample_from_anchor=sample_from_anchor,
        proposal={"mode": "softmax", "temperature": 1.0, "top_k": 0, "top_p": 1.0},
        target={"mode": "softmax", "temperature": 1.0, "top_k": 0, "top_p": 1.0},
        score_chunk_size=2,
        scan_draft=scanner,
        verifier_last_hidden_states=torch.zeros(1, 7, 2),
        verifier_norm=lambda x: x,
        verifier_lm_head=lambda x: x,
    )
    assert calls == [0, 1, 2, 3, 4]
    expected = torch.zeros(7, dtype=torch.float64)
    expected[0] = 1
    expected[block_size + int(sample_from_anchor)] = 1
    torch.testing.assert_close(result, expected)


def test_supervised_gaps_and_documents_reset_local_horizons():
    ids = torch.zeros((1, 12), dtype=torch.long)
    docs = torch.tensor([[0] * 6 + [1] * 6])
    mask = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]])
    result = build_occupancy_scores(
        ids, mask, docs, 2, sample_from_anchor=False,
        proposal={"mode": "greedy"}, target={"mode": "greedy"},
        score_chunk_size=2,
        scan_draft=lambda anchors: torch.zeros(anchors.numel(), 2, 2),
        verifier_last_hidden_states=torch.zeros(1, 12, 2),
        verifier_norm=lambda x: x, verifier_lm_head=lambda x: x,
    )
    expected = torch.zeros(12, dtype=torch.float64)
    expected[[0, 6, 8]] = 1
    torch.testing.assert_close(result, expected)


def test_sampling_policy_support_and_validation():
    policy = {"mode": "softmax", "temperature": 1.0, "top_k": 1, "top_p": 0.8}
    actual = _realized_probabilities(torch.tensor([[2.0, 1.0], [1.0, 2.0]]), torch.tensor([0, 0]), policy)
    torch.testing.assert_close(actual, torch.tensor([1.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="requires"):
        _realized_probabilities(torch.ones(1, 2), torch.zeros(1, dtype=torch.long), {"mode": "softmax"})


def test_zero_target_probability_rejected():
    with pytest.raises(ValueError, match="zero probability"):
        build_occupancy_scores(
            torch.ones((1, 5), dtype=torch.long), torch.ones(1, 5), torch.zeros(1, 5, dtype=torch.long),
            2, sample_from_anchor=False,
            proposal={"mode": "greedy"}, target={"mode": "greedy"}, score_chunk_size=2,
            scan_draft=lambda anchors: torch.zeros(anchors.numel(), 2, 2),
            verifier_last_hidden_states=torch.zeros(1, 5, 2),
            verifier_norm=lambda x: x, verifier_lm_head=lambda x: x,
        )


def test_capped_pps_empirical_marginals_and_distinct_fixed_budget():
    weights = torch.tensor([1.0, 0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    # For K=3: first start is certain; remaining marginals = 2*w / sum(w[1:]).
    expected = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2], dtype=torch.float64)
    counts = torch.zeros(5, dtype=torch.float64)
    generator = torch.Generator().manual_seed(37)
    for _ in range(4000):
        anchors, valid, probabilities = select_occupancy_anchors(weights, torch.ones(5, dtype=torch.bool), 3, generator=generator)
        assert bool(valid.all())
        assert anchors.unique().numel() == 3
        assert bool((anchors[1:] > anchors[:-1]).all())
        torch.testing.assert_close(probabilities, expected[anchors])
        counts[anchors] += 1
    torch.testing.assert_close(counts / 4000, expected, atol=0.025, rtol=0)


def test_zero_occupancy_and_candidate_mask_padding_and_reproducibility():
    scores = torch.tensor([0.0, 1.0, 0.2, 0.3])
    mask = torch.tensor([True, False, True, True])
    anchors, valid, pi = select_occupancy_anchors(scores, mask, 4)
    assert anchors[valid].tolist() == [2, 3]
    assert pi.tolist() == [1, 1, 0, 0]
    assert not bool(valid[2:].any())
    args = (torch.ones(10), torch.ones(10, dtype=torch.bool), 3)
    first = select_occupancy_anchors(*args, generator=torch.Generator().manual_seed(3))
    second = select_occupancy_anchors(*args, generator=torch.Generator().manual_seed(3))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("low_logit", [-150.0, -1000.0])
def test_low_positive_target_support_uses_log_ratio_and_unique_token_chunks(low_logit):
    # Both probabilities underflow FP32 exp; their q/p ratio is exp(-1).
    ids = torch.ones((1, 10), dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[:, 0] = 0
    head_shapes = []
    policy = {"mode": "softmax", "temperature": 1.0, "top_k": 0, "top_p": 1.0}
    hidden = torch.zeros(1, 10, 2)
    hidden[..., 1] = low_logit

    def head(values):
        head_shapes.append(tuple(values.shape))
        return values

    def scanner(anchors):
        assert anchors.numel() == 1  # A block is the minimum chunk at budget 2.
        logits = torch.zeros(anchors.numel(), 4, 2)
        logits[..., 1] = low_logit - 1
        return logits

    scores = build_occupancy_scores(
        ids, mask, torch.zeros_like(ids), 4,
        sample_from_anchor=False, proposal=policy, target=policy,
        score_chunk_size=2, scan_draft=scanner,
        verifier_last_hidden_states=hidden,
        verifier_norm=lambda x: x, verifier_lm_head=head,
    )
    expected = round_start_occupancy(torch.full((8, 3), -1.0).exp())
    torch.testing.assert_close(scores[:8], expected)
    # Exactly eight unique ordinary target tokens, never anchors*block rows.
    assert sum(shape[0] for shape in head_shapes) == 8
    assert all(len(shape) == 2 and shape[0] <= 2 for shape in head_shapes)
    if low_logit == -150:
        probability = _realized_probabilities(
            torch.tensor([[0.0, low_logit]]), torch.tensor([1]), policy
        )
        assert float(probability[0]) > 0
