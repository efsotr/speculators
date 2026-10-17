"""Regression coverage for verifier-owned DFlash2 output heads."""

import pyarrow  # noqa: F401  # Load Arrow before Torch native libraries.
import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


@pytest.mark.parametrize("tied_head", [False, True], ids=["independent-head", "tied-fallback"])
def test_dflash2_full_vocab_head_copies_and_freezes_verifier_weights(monkeypatch, tied_head):
    """The full-vocabulary draft head must use independent verifier weight storage."""
    transformer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        tie_word_embeddings=tied_head,
        _attn_implementation="eager",
    )
    config = DFlash2SpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        conv_group_size=4,
        selector_rank=8,
        selector_top_k=4,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash2",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="test-verifier",
                architectures=["Qwen3ForCausalLM"],
            ),
        ),
    )
    verifier_head = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16)
    verifier_weights = {
        "embed_tokens.weight": torch.full((64, 16), -1.0),
        "model.norm.weight": torch.ones(16),
    }
    if not tied_head:
        verifier_weights["lm_head.weight"] = verifier_head
    expected_head = verifier_weights[
        "embed_tokens.weight" if tied_head else "lm_head.weight"
    ]
    calls = []

    def load_verifier_layers(names, name_or_path):
        calls.append((names, name_or_path))
        return verifier_weights

    monkeypatch.setattr(
        "speculators.utils.loading.load_model_layers", load_verifier_layers
    )
    model = DFlash2DraftModel(config)
    model.load_verifier_weights()

    assert calls == [
        (
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
            "test-verifier",
        )
    ]
    assert not model.use_draft_vocab
    for head in (model.lm_head, model.verifier_lm_head):
        torch.testing.assert_close(head.weight, expected_head)
        assert not torch.isnan(head.weight).any()
        assert not head.weight.requires_grad
        assert head.weight.data_ptr() != expected_head.data_ptr()
        assert head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
    assert model.lm_head.weight.data_ptr() != model.verifier_lm_head.weight.data_ptr()
