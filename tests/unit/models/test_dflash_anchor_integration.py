"""CPU model integration for cached-rollout anchor selection."""

import pyarrow  # noqa: F401
import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.core import DFlash2DraftModel


def _model(model_type="dflash", sample=False, draft_vocab_size=32):
    torch.manual_seed(101)
    decoder = Qwen3Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, _attn_implementation="eager",
    )
    config_cls, model_cls = (
        (DFlashSpeculatorConfig, DFlashDraftModel) if model_type == "dflash"
        else (DFlash2SpeculatorConfig, DFlash2DraftModel)
    )
    config = config_cls(
        transformer_layer_config=decoder, draft_vocab_size=draft_vocab_size, block_size=4,
        aux_hidden_state_layer_ids=[0, 1], mask_token_id=0,
        sample_from_anchor=sample,
    )
    model = model_cls(config)
    # Verifier-owned matrices intentionally start as NaN until loaded; supply
    # synthetic finite verifier weights rather than loading a real checkpoint.
    torch.nn.init.normal_(model.embed_tokens.weight, std=0.1)
    torch.nn.init.normal_(model.lm_head.weight, std=0.1)
    torch.nn.init.normal_(model.verifier_lm_head.weight, std=0.1)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def _batch(length=20):
    generator = torch.Generator().manual_seed(19)
    return {
        "hidden_states": torch.randn(1, length, 32, generator=generator),
        "verifier_last_hidden_states": torch.randn(1, length, 16, generator=generator),
        "input_ids": torch.randint(0, 32, (1, length), generator=generator),
        "loss_mask": torch.ones(1, length),
        "document_ids": torch.zeros(1, length, dtype=torch.long),
    }


def _run(model, batch, policy, **overrides):
    train_kwargs, _ = model.get_trainer_kwargs(
        loss_fn="ce", loss_implementation="eager", max_anchors=3,
        anchor_sampling={"score_backend": "eager", **policy},
    )
    train_kwargs.update(overrides)
    return model(**model.prepare_training_inputs(**batch, **train_kwargs))


@pytest.mark.parametrize("model_type", ["dflash", "dflash2"])
@pytest.mark.parametrize("strategy", [
    "target_flatness", "target_entropy_high", "target_entropy_low", "entropy_curriculum",
])
def test_nonuniform_model_forward_backward_finite(model_type, strategy):
    model = _model(model_type).train()
    policy = {"strategy": strategy, "score_chunk_size": 2}
    if strategy != "entropy_curriculum":
        policy["retain_fraction"] = 0.5
    _, loss, metrics = _run(
        model, _batch(), policy, anchor_sampling_num_epochs=8,
        anchor_sampling_epoch=0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert metrics["anchor_selected_sum"].item() == 3
    assert metrics["anchor_selected_total"].item() == 1


@pytest.mark.parametrize("model_type", ["dflash", "dflash2"])
@pytest.mark.parametrize("supervised", [0, 5])
def test_empty_or_padded_anchors_have_finite_zero_safe_gradients(model_type, supervised):
    model = _model(model_type).train()
    batch = _batch()
    batch["loss_mask"].zero_()
    batch["loss_mask"][:, 2:2 + supervised] = 1
    _, loss, metrics = _run(model, batch, {
        "strategy": "target_flatness", "retain_fraction": 0.5,
    }, max_anchors=8)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert metrics["anchor_selected_sum"].item() <= 1
    if not supervised:
        assert loss.item() == 0


@pytest.mark.parametrize("model_type", ["dflash", "dflash2"])
def test_train_and_validation_policies_are_independent(model_type):
    model = _model(model_type)
    policy = {"strategy": "entropy_curriculum", "curriculum_preset": "github"}
    train, val = model.get_trainer_kwargs(
        loss_fn="ce", loss_implementation="eager", anchor_sampling=policy,
    )
    assert train["anchor_sampling"] == policy
    assert val["anchor_sampling"] == {"strategy": "uniform"}
    val["anchor_sampling"]["strategy"] = "changed"
    assert train["anchor_sampling"] == policy


@pytest.mark.parametrize("sample", [False, True])
def test_target_scorer_uses_hidden_at_anchor_not_adjacent_hidden(sample):
    model = _model(sample=sample)
    batch = _batch()
    batch["loss_mask"].zero_()
    batch["loss_mask"][:, 3:8] = 1
    # Only start3 is legal in both layouts after constraining its targets.
    if not sample:
        batch["loss_mask"][:, 7] = 0
    seen = []
    hook = model.verifier_norm.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    try:
        anchors, valid = model._sample_training_anchors(
            batch["hidden_states"], batch["input_ids"], batch["loss_mask"],
            batch["verifier_last_hidden_states"], batch["document_ids"], None,
            1, {"strategy": "target_entropy_high", "score_backend": "eager"}, 0, 1, 0, 0,
        )
    finally:
        hook.remove()
    assert valid.tolist() == [True]
    assert anchors.tolist() == [3]
    torch.testing.assert_close(seen[0], batch["verifier_last_hidden_states"][:, 3:4])


def test_resume_sampling_is_determined_by_step_epoch_rank_and_seed():
    model = _model()
    batch = _batch(64)
    policy = {"strategy": "target_flatness", "retain_fraction": 0.8, "seed": 7, "score_backend": "eager"}

    def select(model, epoch, step, rank):
        return model._sample_training_anchors(
            batch["hidden_states"], batch["input_ids"], batch["loss_mask"],
            batch["verifier_last_hidden_states"], batch["document_ids"], None,
            4, policy, epoch, 8, step, rank,
        )[0]

    reference = select(model, 2, 11, 1)
    torch.manual_seed(999)
    torch.rand(53)
    resumed = _model()
    resumed.load_state_dict(model.state_dict())
    torch.testing.assert_close(reference, select(resumed, 2, 11, 1))
    assert not torch.equal(reference, select(model, 2, 12, 1))
    assert not torch.equal(reference, select(model, 3, 11, 1))
    assert not torch.equal(reference, select(model, 2, 11, 0))


def test_actual_reduced_vocab_mapping_rejected_at_runtime():
    model = _model(draft_vocab_size=16)
    assert model.use_draft_vocab and model.draft_vocab_size < model.verifier_vocab_size
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        _run(model, _batch(), {"strategy": "target_flatness"})


def _occupancy_policy():
    return {"strategy": "round_occupancy", "score_chunk_size": 2,
            "proposal": {"mode": "softmax", "temperature": 1.0, "top_k": 0, "top_p": 1.0},
            "target": {"mode": "softmax", "temperature": 1.0, "top_k": 0, "top_p": 1.0}}


def test_dflash2_occupancy_rejected_before_scoring():
    with pytest.raises(ValueError, match="DFlash2 is unsupported"):
        _run(_model("dflash2"), _batch(), _occupancy_policy())


def test_occupancy_runs_real_drafter_scan_and_reports_only_selected_blocks(monkeypatch):
    model = _model().train()
    batch = _batch(12)
    real_backbone = model._backbone_forward
    scans = []

    def observe(*args, **kwargs):
        result = real_backbone(*args, **kwargs)
        if kwargs.get("_anchor_scan"):
            scans.append(result[1])
            assert result[2].numel() == 0
            assert not result[1].requires_grad
        return result

    monkeypatch.setattr(model, "_backbone_forward", observe)
    _, loss, metrics = _run(model, batch, _occupancy_policy(), max_anchors=2)
    assert scans
    assert sum(logits.shape[1] for logits in scans) > 2 * model.block_size
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert metrics["anchor_selected_sum"].item() == 2
    assert metrics["anchor_selected_total"].item() == 1
    assert metrics["anchor_score_seconds_total"].item() == 1
    assert metrics["anchor_inclusion_total"].item() == 2


@pytest.mark.parametrize("strategy", ["uniform", "target_flatness"])
@pytest.mark.parametrize("fail", [False, True])
def test_external_fsdp_sampling_gathers_root_and_always_reshards(monkeypatch, strategy, fail):
    """Model the public FSDP lifecycle without initializing CUDA/process groups."""
    from torch.distributed.fsdp import FSDPModule

    model = _model()
    model.__class__ = type("MockFSDPDFlash", (FSDPModule, type(model)), {})
    calls = []
    monkeypatch.setattr(model, "unshard", lambda: calls.append("unshard"))
    monkeypatch.setattr(model, "reshard", lambda: calls.append("reshard"))

    def sample(*args):
        calls.append("sample")
        if fail:
            raise RuntimeError("sampling failure")
        return torch.tensor([0]), torch.tensor([True])

    monkeypatch.setattr(model, "_sample_training_anchors", sample)
    monkeypatch.setattr(model, "_build_attention_mask", lambda *args, **kwargs: (None,) * 4)
    if fail:
        with pytest.raises(RuntimeError, match="sampling failure"):
            model.prepare_training_inputs(**_batch(), max_anchors=1,
                                          anchor_sampling={"strategy": strategy})
    else:
        model.prepare_training_inputs(**_batch(), max_anchors=1,
                                      anchor_sampling={"strategy": strategy})
    assert calls == (["sample"] if strategy == "uniform"
                     else ["unshard", "sample", "reshard"])
