"""Smoke checks for dataset-ID CE, alignment, reduction and fused compilation."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.nn.functional import cross_entropy
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import compound_loss, resolve_loss_config
from speculators.losses.eager import ce_gt_loss, ce_loss, tv_loss
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics as dflash_metrics
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics as dspark_metrics
from speculators.train.config import TrainConfig

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_ce_gt_matches_ce_dtype_and_gradient(dtype):
    torch.manual_seed(274)
    logits = torch.randn(1, 8, 17, dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(logits)
    actual = ce_gt_loss(logits, teacher.argmax(-1))
    expected = ce_loss(logits, teacher)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), logits)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("model_type", ["dflash", "dflash2", "dspark"])
def test_schema_accepts_ce_gt(model_type):
    cfg = TrainConfig.model_validate(
        {
            "speculator_type": model_type,
            "loss": {"loss_fn": '{"ce_gt": 0.1, "tv": 1.8}'},
            "dflash": {"per_position_loss_weight": "fixed-exp-decay"},
        }
    )
    assert "ce_gt" in resolve_loss_config(cfg.loss.loss_fn)


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_ce_gt_alignment(sample_from_anchor):
    model = SimpleNamespace(
        config=SimpleNamespace(sample_from_anchor=sample_from_anchor),
        use_draft_vocab=False,
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 0]])
    indices = torch.tensor([0, 1, 2, 3, 4, 5])
    actual = DFlashDraftModel._aligned_data_token_ids(
        model, tokens, indices, require_full_vocab=True
    )
    expected = tokens[:, indices + int(sample_from_anchor)]
    torch.testing.assert_close(actual, expected)
    model.use_draft_vocab = True
    with pytest.raises(ValueError, match="full verifier vocabulary"):
        DFlashDraftModel._aligned_data_token_ids(
            model, tokens, indices, require_full_vocab=True
        )


@pytest.mark.parametrize("implementation", ["eager", "fused"])
def test_ce_gt_requires_ids(implementation):
    with pytest.raises(ValueError, match="requires aligned"):
        compound_loss(
            torch.randn(1, 4, 7),
            torch.randn(1, 4, 7),
            torch.ones(1, 4),
            torch.arange(4).unsqueeze(0),
            resolve_loss_config("ce_gt", implementation),
        )


@pytest.mark.parametrize("kind", ["dflash", "dspark"])
@pytest.mark.parametrize("empty", [False, True])
def test_mixed_ce_gt_uses_data_and_preserves_teacher_tv(kind, empty):
    torch.manual_seed(269)
    logits = torch.randn(1, 8, 7, requires_grad=True)
    teacher = torch.randn_like(logits)
    ids = (teacher.argmax(-1) + 1) % 7  # deliberately disagree
    mask = torch.tensor([[1.0, 1, 0, 1, 1, 0, 1, 1]])
    if empty:
        mask.zero_()
    ids = torch.where(mask.bool(), ids, -100)
    cfg = resolve_loss_config('{"ce_gt": 0.1, "tv": 1.8}', "eager")
    kwargs = {
        "loss_config": cfg,
        "gamma": 4.0,
        "sample_from_anchor": True,
        "data_token_ids": ids,
    }
    if kind == "dspark":
        loss, metrics = dspark_metrics(
            logits, teacher, None, mask, 4, tv_loss_fn=tv_loss, **kwargs
        )
    else:
        loss, metrics = dflash_metrics(logits, teacher, mask, 4, **kwargs)
    weights = torch.exp(-(torch.arange(8) % 4) / 4.0)
    labels = torch.where(mask.bool(), ids, 0)
    ce = cross_entropy(
        logits.float().reshape(-1, 7), labels.reshape(-1), reduction="none"
    ).view(1, 8)
    tv = 0.5 * (logits.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)
    masked_weights = weights * mask
    expected = ((0.1 * ce + 1.8 * tv) * masked_weights).sum() / (
        masked_weights.sum() + 1e-5
    )
    torch.testing.assert_close(loss, expected)
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
    assert "ce_gt_loss_sum" in metrics
    if empty:
        assert loss.item() == 0
        assert torch.count_nonzero(actual_grad) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [7, 151936])
@pytest.mark.parametrize("compiled", [False, True])
def test_fused_ce_gt_value_gradient_and_compile(dtype, vocab, compiled):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(269)
    logits = (torch.randn(1, 8, vocab, device="cuda", dtype=dtype) * 3).requires_grad_()
    ids = torch.randint(vocab, (1, 8), device="cuda")
    ids[0, -1] = vocab - 1  # exercise non-power-of-two tail
    fused = resolve_loss_config("ce_gt")["ce_gt"][0]
    if compiled:
        fused = torch.compile(fused, fullgraph=True, backend="inductor")
    actual = fused(logits, ids)
    expected = cross_entropy(
        logits.float().reshape(-1, vocab), ids.reshape(-1), reduction="none"
    ).reshape(ids.shape)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    upstream = torch.randn_like(actual)
    upstream[:, ::3] = 0
    actual_grad = torch.autograd.grad((actual * upstream).sum(), logits)[0]
    expected_grad = torch.autograd.grad((expected * upstream).sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-3, rtol=1e-2)


def test_project_dspark_configs():
    root = Path(__file__).resolve().parents[5]
    if not (root / "scripts/dspark").exists():
        pytest.skip("Project configs unavailable outside parent repository")
    expected = {
        "ce_tv": {"ce": 0.1, "tv": 1.8},
        "ce_gt_tv": {"ce_gt": 0.1, "tv": 1.8},
        "kl_tv": {"kl_div": 0.1, "tv": 1.8},
    }
    configs = []
    for suffix, terms in expected.items():
        raw = yaml.safe_load(
            (root / f"scripts/dspark/dspark_{suffix}.yaml").read_text()
        )
        cfg = TrainConfig.model_validate(raw)
        resolved = resolve_loss_config(cfg.loss.loss_fn)
        assert {name: weight for name, (_, weight) in resolved.items()} == terms
        assert cfg.optimizer.weight_decay == 0.0
        assert cfg.optimizer.lr == 6e-4
        configs.append(cfg.model_dump())
    for cfg in configs:
        for group, key in [
            ("loss", "loss_fn"),
            ("trainer", "save_path"),
            ("logging", "run_name"),
            ("logging", "log_dir"),
        ]:
            cfg[group].pop(key)
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("signal", [None, "ce_gt", "p_draft", "q_draft"])
def test_dspark_model_ce_gt_forward_backward(monkeypatch, sample_from_anchor, signal):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = DSparkSpeculatorConfig(
        transformer_layer_config=Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            _attn_implementation="eager",
        ),
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
        markov_rank=8,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    model = DSparkDraftModel(config).train()
    for name in ("embed_tokens", "lm_head", "verifier_lm_head"):
        torch.nn.init.normal_(getattr(model, name).weight, std=0.1)
    torch.nn.init.ones_(model.verifier_norm.weight)
    model.configure_compile(
        mode="default", pad_target_hidden_states=True, target_seq_len=32
    )
    batch = {
        "hidden_states": torch.randn(1, 24, 32),
        "verifier_last_hidden_states": torch.randn(1, 24, 16),
        "input_ids": torch.randint(0, 64, (1, 24)),
        "loss_mask": torch.ones(1, 24),
        "document_ids": torch.zeros(1, 24, dtype=torch.long),
    }
    kwargs, _ = model.get_trainer_kwargs(
        loss_fn=(
            '{"ce": 0.1, "tv": 1.8}'
            if signal in {"p_draft", "q_draft"}
            else '{"ce_gt": 0.1, "tv": 1.8}'
        ),
        loss_implementation="eager",
        per_position_loss_weight="dpa-dev" if signal else "fixed-exp-decay",
        dpa_dev_signal=signal or "accept",
        dpa_dev_prefix_only=signal in {"p_draft", "q_draft"},
        max_anchors=3,
    )
    prepared = model.prepare_training_inputs(**batch, **kwargs)
    _, loss, metrics = model(**prepared)
    assert torch.isfinite(loss)
    assert (
        "ce_loss_sum" if signal in {"p_draft", "q_draft"} else "ce_gt_loss_sum"
    ) in metrics
    loss.backward()
    for head in (model.markov_head, model.confidence_head):
        grads = [p.grad for p in head.parameters() if p.requires_grad]
        assert grads
        assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_standalone_ce_gt_with_default_dflash_dpace():
    cfg = TrainConfig.model_validate(
        {"speculator_type": "dflash", "loss": {"loss_fn": "ce_gt"}}
    )
    assert cfg.dflash.per_position_loss_weight == "dpace"


def test_ce_gt_rejects_models_without_dataset_alignment():
    with pytest.raises(ValueError, match="ce_gt requires"):
        TrainConfig.model_validate(
            {"speculator_type": "eagle3", "loss": {"loss_fn": "ce_gt"}}
        )
