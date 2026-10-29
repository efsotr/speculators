"""CPU checks for native preprocessing with persistent max-autotune mask indices."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.stable_block_mask import INDEX_NAMES, StableBlockMasks


class MaskModel(torch.nn.Module):
    """Use real DFlash preparation and mask construction without model weights."""

    configure_compile = DFlashDraftModel.configure_compile
    prepare_training_inputs = DFlashDraftModel.prepare_training_inputs
    _pad_target_inputs = DFlashDraftModel._pad_target_inputs
    _build_attention_mask = DFlashDraftModel._build_attention_mask
    _create_attention_mask = DFlashDraftModel._create_attention_mask

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.block_size = 64
        self.config = SimpleNamespace(sample_from_anchor=False)
        self.uses_full_attn = self.uses_sliding_window_attn = True
        self.sliding_window = 128
        self.sliding_window_non_causal = True
        self._attn_impl = "simple_flex_attention"
        self._create_mask_fn = create_block_mask

    def forward(self, *, max_anchors=2):
        raise NotImplementedError


def _prepare(model, positions, validity):
    ids = torch.arange(256).unsqueeze(0)
    anchors = torch.tensor(positions)
    valid = torch.tensor(validity)
    prepared = model.prepare_training_inputs(
        hidden_states=torch.zeros(1, 256, 4),
        input_ids=ids,
        loss_mask=torch.ones_like(ids),
        verifier_last_hidden_states=torch.zeros(1, 256, 2),
        document_ids=torch.zeros_like(ids),
        max_anchors=2,
        _selected_anchors=(anchors, valid),
    )
    return prepared["_prepared_attention_mask"], anchors, valid


def _values(masks):
    return {
        (label, name): value.clone()
        for label, mask in zip(("full", "sliding"), masks[:2], strict=True)
        for name in INDEX_NAMES
        if (value := getattr(mask, name)) is not None
    }


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # Exercise native configuration but avoid compiling or allocating on a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_native_prepare_only_reuses_storage_in_max_autotune():
    model = MaskModel()
    model.configure_compile(mode="default")
    assert model._stable_block_masks is None
    first, _, _ = _prepare(model, [192, 224], [True, True])
    second, _, _ = _prepare(model, [64, 96], [True, False])
    assert first[0].kv_indices.data_ptr() != second[0].kv_indices.data_ptr()

    model.configure_compile(mode="max-autotune")
    assert isinstance(model._stable_block_masks, StableBlockMasks)
    first, _, _ = _prepare(model, [192, 224], [True, True])
    pointers = model._stable_block_masks.pointers()
    previous_values = _values(first)
    second, anchors, valid = _prepare(model, [64, 96], [True, False])
    assert model._stable_block_masks.pointers() == pointers
    assert second[2] is anchors
    assert second[3] is valid
    assert first[0] is not second[0]
    assert first[1] is not second[1]
    assert first[0].mask_mod is not second[0].mask_mod
    assert first[1].mask_mod is not second[1].mask_mod
    assert any(
        not torch.equal(value, previous_values[key])
        for key, value in _values(second).items()
    )

    reference, _, _ = _prepare(MaskModel(), [64, 96], [True, False])
    for key, value in _values(second).items():
        torch.testing.assert_close(value, _values(reference)[key])
    for actual, expected in zip(second[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual.to_dense(), expected.to_dense())
        q = torch.arange(128)[:, None]
        k = torch.arange(384)[None, :]
        torch.testing.assert_close(
            actual.mask_mod(torch.tensor(0), torch.tensor(0), q, k),
            expected.mask_mod(torch.tensor(0), torch.tensor(0), q, k),
        )

    model.configure_compile(mode="default")
    assert model._stable_block_masks is None
    fresh, _, _ = _prepare(model, [192, 224], [True, True])
    assert fresh[0].kv_indices.data_ptr() != pointers["full.kv_indices"]


def test_dense_backend_keeps_native_masks_in_max_autotune():
    from speculators.models.attention import create_float_mask

    model = MaskModel()
    model._attn_impl = "eager"
    model._create_mask_fn = create_float_mask
    model.configure_compile(mode="max-autotune")
    assert model._stable_block_masks is None
    masks, _, _ = _prepare(model, [64, 96], [True, False])
    assert isinstance(masks[0], torch.Tensor)
    assert isinstance(masks[1], torch.Tensor)


@pytest.mark.parametrize("change", ["shape", "stride", "dtype", "device", "optional"])
def test_layout_change_rejected_before_mutating_existing_buffers(change):
    def mask(value):
        return SimpleNamespace(
            **{
                name: torch.full((2, 2), value, dtype=torch.int32)
                for name in INDEX_NAMES
            },
            mask_mod=object(),
        )

    stable = StableBlockMasks()
    full, sliding = mask(1), mask(2)
    stable.update((full, sliding, None, None))
    previous = _values((full, sliding))
    new_full, new_sliding = mask(3), mask(4)
    replacements = {
        "shape": torch.zeros(3, 2, dtype=torch.int32),
        "stride": torch.zeros(2, 2, dtype=torch.int32).t(),
        "dtype": torch.zeros(2, 2, dtype=torch.int64),
        "device": torch.zeros(2, 2, dtype=torch.int32, device="meta"),
        "optional": None,
    }
    new_sliding.q_indices = replacements[change]
    with pytest.raises(ValueError, match="unchanged index"):
        stable.update((new_full, new_sliding, None, None))
    for key, value in _values((full, sliding)).items():
        torch.testing.assert_close(value, previous[key])


def test_optional_mask_and_indices_remain_optional():
    mask = create_block_mask(
        lambda b, h, q, k: q >= k,
        B=1, H=1, Q_LEN=256, KV_LEN=256, device="cpu",
    )
    mask.full_q_indices = None
    stable = StableBlockMasks()
    result = stable.update((None, mask, None, None))
    assert result[0] is None
    assert result[1].full_q_indices is None
    assert len(stable.pointers()) == 7
