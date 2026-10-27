"""CPU checks of cached-rollout scores and anchor budget allocation."""

import pyarrow  # noqa: F401
import pytest
import torch
from torch import nn

from speculators.models.dflash.anchor_sampling import (
    score_target_positions,
    select_static_anchors,
    valid_anchor_mask,
)


def _select(strategy="uniform", *, length=20, budget=5, retain=1.0, **kwargs):
    sampling = {"strategy": strategy, "retain_fraction": retain}
    sampling.update(kwargs.pop("sampling", {}))
    hidden = kwargs.pop("hidden", torch.zeros(1, length, 3))
    return select_static_anchors(
        torch.ones(1, length), budget, 2, torch.zeros(1, length, dtype=torch.long),
        sampling=sampling, verifier_last_hidden_states=hidden,
        verifier_norm=nn.Identity(), verifier_lm_head=nn.Identity(), **kwargs,
    )


@pytest.mark.parametrize("sample,expected", [(False, [0, 1, 4]), (True, [0, 4])])
def test_document_and_supervision_boundaries(sample, expected):
    mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1]])
    docs = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    actual = valid_anchor_mask(mask, docs, 2, sample)
    assert actual.nonzero(as_tuple=True)[0].tolist() == expected


def test_last_packed_block_stays_excluded():
    result = valid_anchor_mask(torch.ones(1, 6), torch.zeros(1, 6), 3)
    assert result.tolist() == [True, True, True, False, False, False]


@pytest.mark.parametrize("strategy", ["target_entropy_high", "target_flatness"])
def test_scores_match_hand_calculation_and_chunking(strategy):
    probs = torch.tensor([[[0.5, 0.5], [0.9, 0.1], [0.25, 0.75]]])
    expected = (
        1 / (2 * probs.square().sum(-1)).sqrt()
        if strategy == "target_flatness"
        else -(probs * probs.log()).sum(-1)
    ).squeeze(0)
    actual = score_target_positions(probs.log(), torch.arange(3), nn.Identity(), nn.Identity(), strategy, 1)
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected)
    assert not actual.requires_grad


def test_entropy_and_flatness_can_have_opposite_rankings():
    probs = torch.tensor([[[0.7, 0.15, 0.15], [0.5, 0.499, 0.001]]])
    args = (probs.log(), torch.arange(2), nn.Identity(), nn.Identity())
    entropy = score_target_positions(*args, "target_entropy_high", 2)
    flatness = score_target_positions(*args, "target_flatness", 2)
    assert entropy[0] > entropy[1]
    assert flatness[0] < flatness[1]


def test_scoring_uses_anchor_hidden_not_following_token():
    hidden = torch.tensor([[[0., 0.], [10., 0.], [0., 0.], [10., 0.], [0., 0.], [0., 0.]]])
    anchors, valid = _select("target_entropy_high", length=6, budget=4, retain=0.5, hidden=hidden)
    assert anchors[valid].tolist() == [0, 2]


def test_low_entropy_reverses_order():
    hidden = torch.tensor([[[0., 0.], [10., 0.], [0., 0.], [10., 0.], [0., 0.], [0., 0.]]])
    anchors, valid = _select("target_entropy_low", length=6, budget=4, retain=0.5, hidden=hidden)
    assert anchors[valid].tolist() == [1, 3]


def test_ties_retain_exact_count_without_refilling_budget():
    anchors, valid = _select("target_flatness", length=12, budget=8, retain=0.21)
    assert valid.sum() == 3  # ceil(10*.21)
    selected = anchors[valid]
    assert selected.unique().numel() == 3
    assert torch.equal(selected, selected.sort().values)
    assert (anchors[~valid] == 0).all()


@pytest.mark.parametrize("preset,count", [("paper", 85), ("github", 90)])
def test_curriculum_initial_and_last_epoch(preset, count):
    _, valid = _select("entropy_curriculum", length=102, budget=102,
                       sampling={"curriculum_preset": preset}, epoch=0, num_epochs=6)
    assert valid.sum() == count
    _, valid = _select("entropy_curriculum", length=102, budget=102,
                       sampling={"curriculum_preset": preset}, epoch=5, num_epochs=6)
    assert valid.sum() == 100


def test_github_curriculum_caps_drop_at_forty_percent():
    _, valid = _select("entropy_curriculum", length=102, budget=102,
                       sampling={"curriculum_preset": "github"}, epoch=0, num_epochs=40)
    assert valid.sum() == 60
    with pytest.raises(ValueError, match="initial drop"):
        _select("entropy_curriculum", epoch=39, num_epochs=40)


def test_single_epoch_curriculum_matches_uniform_rng():
    a, v = _select(generator=torch.Generator().manual_seed(9))
    b, w = _select("entropy_curriculum", generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b) and torch.equal(v, w)


def test_generator_reproducible_and_advances():
    gen = torch.Generator().manual_seed(48)
    first, _ = _select("target_flatness", retain=0.5, generator=gen)
    state = gen.get_state()
    second, _ = _select("target_flatness", retain=0.5, generator=gen)
    restored = torch.Generator()
    restored.set_state(state)
    replay, _ = _select("target_flatness", retain=0.5, generator=restored)
    assert torch.equal(second, replay)
    assert not torch.equal(first, second)


def test_empty_and_short_sequences_need_no_scorer():
    for length in [0, 1, 2]:
        a, v = _select("target_flatness", length=length)
        assert not v.any() and (a == 0).all()
    a, v = select_static_anchors(torch.zeros(1, 20), 3, 2, torch.zeros(1, 20))
    assert a.shape == v.shape == (3,)
    assert not v.any()


@pytest.mark.parametrize("kwargs", [
    {"strategy": "bad"}, {"retain": 0}, {"retain": 1.1},
    {"retain": 0.5}, {"sampling": {"curriculum_preset": "github"}},
    {"sampling": {"score_chunk_size": 0}},
])
def test_invalid_options_fail(kwargs):
    with pytest.raises(ValueError):
        _select(**kwargs)


def test_missing_hidden_and_nonfinite_scores_fail():
    with pytest.raises(ValueError, match="cached"):
        select_static_anchors(torch.ones(1, 8), 2, 2, torch.zeros(1, 8),
                              sampling={"strategy": "target_flatness"})
    with pytest.raises(ValueError, match="finite"):
        _select("target_flatness", hidden=torch.full((1, 20, 3), float("nan")))


def test_auto_cpu_scorer_matches_eager_and_rejects_unknown_backend():
    hidden = torch.randn(1, 7, 11)
    args = (hidden, torch.tensor([6, 1, 4]), nn.Identity(), nn.Identity(), "target_flatness", 2)
    torch.testing.assert_close(score_target_positions(*args), score_target_positions(*args, backend="eager"))
    with pytest.raises(ValueError, match="backend"):
        score_target_positions(*args, backend="unknown")
    with pytest.raises(ValueError, match="CUDA"):
        score_target_positions(*args, backend="triton")
