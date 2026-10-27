"""Packed metadata/loader contracts; opt-in real FA2+Inductor GPU validation."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from speculators.train.data import ArrowDataset, CollateFn
from speculators.train.online_backend import OnlineTransfer, PackedQwen3Features


def make_transfer(tmp_path, limit=3):
    path = tmp_path / 'prepared'
    Dataset.from_dict({
        'input_ids': [[1, 2, 3, 4], [5], [6, 7], [8, 9, 10]],
        'loss_mask': [[0, 1, 1, 1], [0], [0, 1], [1, 1, 1]],
        'seq_len': [4, 1, 2, 3],
    }).with_format('torch').save_to_disk(str(path))
    transfer = OnlineTransfer(SimpleNamespace(target_layer_ids=[0, 1],
                                              max_sample_seq_len=limit), path)
    transfer.data = Dataset.load_from_disk(str(path)).with_format('torch')
    transfer.device = torch.device('cpu')
    transfer.model = SimpleNamespace(config=SimpleNamespace(vocab_size=100))
    calls = []

    def forward(ids, positions, cu, maximum):
        calls.append((ids.clone(), positions.clone(), cu.clone(), maximum))
        h = ids[0].unsqueeze(-1).expand(-1, 2).float()
        return torch.stack([h, h + 1, h + 1], dim=1)

    transfer.feature_forward = forward
    return path, transfer, calls


def test_one_teacher_call_per_dataloader_packing(tmp_path):
    path, transfer, calls = make_transfer(tmp_path)
    ds = ArrowDataset(max_len=8, datapath=path, transfer=transfer,
                      max_sample_seq_len=3, train_ratio=0.75,
                      hidden_states_dtype=torch.float32)
    loader = DataLoader(ds, batch_sampler=[[0, 1, 2]], num_workers=0,
                        collate_fn=CollateFn(8, 2, 2, dtype=torch.float32))
    batch = next(iter(loader))
    assert len(calls) == 1
    ids, pos, cu, maximum = calls[0]
    assert ids.tolist() == [[1, 2, 3, 5, 6, 7]]
    assert pos.tolist() == [[0, 1, 2, 0, 0, 1]]
    assert cu.dtype == torch.int32 and cu.tolist() == [0, 3, 4, 6]
    assert maximum == 3
    assert batch['document_ids'].tolist() == [[0, 0, 0, 1, 2, 2, -1, -1]]
    assert batch['loss_mask'][0, :6].tolist() == [0, 1, 1, 0, 0, 1]
    assert batch['hidden_states'].shape == (1, 8, 4)
    torch.testing.assert_close(batch['hidden_states'][0, :6, -2:],
                               batch['verifier_last_hidden_states'][0, :6])
    assert ds.__getitems__([]) == []
    assert len(calls) == 1


def test_val_storage_offsets_and_oversized_row(tmp_path):
    path, transfer, calls = make_transfer(tmp_path, limit=None)
    ds = ArrowDataset(max_len=2, datapath=path, transfer=transfer,
                      train_ratio=0.75, split='val', hidden_states_dtype=torch.float32)
    sample = ds.__getitems__([0])[0]
    assert sample['input_ids'].tolist() == [8, 9]
    assert sample['position_ids'].tolist() == [0, 1]
    assert sample['lengths'].tolist() == [2]
    assert calls[0][2].tolist() == [0, 2]
    with pytest.raises(ValueError, match='exceeds budget'):
        transfer.get_packed([0, 2], max_tokens=3)
    assert len(calls) == 1


def test_non_online_loader_keeps_single_sample_path(tmp_path):
    path, transfer, calls = make_transfer(tmp_path)

    class SingleOnly:
        def get_cached(self, index):
            return transfer.get_cached(index)

    ds = ArrowDataset(max_len=8, datapath=path, transfer=SingleOnly(),
                      max_sample_seq_len=3, hidden_states_dtype=torch.float32)
    assert len(ds.__getitems__([0, 1])) == 2
    assert len(calls) == 2


@pytest.mark.parametrize('compile_setting', [None, True, False],
                         ids=['default-compiled', 'compiled', 'eager'])
def test_online_setup_compile_switch_preserves_fa2(monkeypatch, compile_setting):
    """Disabling teacher compilation must keep packed FA2 and frozen weights."""
    import datasets
    import transformers
    import transformers.modeling_flash_attention_utils as fa_utils

    config = SimpleNamespace(model_type='qwen3', layer_types=['full_attention'],
                             num_hidden_layers=1)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained',
                        Mock(return_value=config))
    model = torch.nn.Linear(2, 2)
    move = Mock(return_value=model)
    monkeypatch.setattr(model, 'to', move)
    load_model = Mock(return_value=model)
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', load_model)
    dataset = Mock()
    dataset.with_format.return_value = dataset
    load_data = Mock(return_value=dataset)
    monkeypatch.setattr(datasets, 'load_from_disk', load_data)
    register_fa2 = Mock()
    monkeypatch.setattr(fa_utils, 'lazy_import_flash_attention', register_fa2)
    monkeypatch.setitem(sys.modules, 'flash_attn',
                        SimpleNamespace(flash_attn_varlen_func=Mock()))
    compile_fn = Mock(return_value=Mock(name='compiled_forward'))
    monkeypatch.setattr(torch, 'compile', compile_fn)
    monkeypatch.setenv('LOCAL_RANK', '0')
    args = SimpleNamespace(target_layer_ids=[0, 1],
                           verifier_name_or_path='local-qwen3')
    if compile_setting is not None:
        args.online_compile_hidden_states = compile_setting
    transfer = OnlineTransfer(args, 'prepared')

    transfer.setup()

    load_model.assert_called_once_with('local-qwen3', dtype=torch.bfloat16,
                                       attn_implementation='flash_attention_2',
                                       local_files_only=True)
    register_fa2.assert_called_once_with('flash_attention_2')
    move.assert_called_once_with(torch.device('cuda', 0))
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    if compile_setting is False:
        compile_fn.assert_not_called()
        assert isinstance(transfer.feature_forward, PackedQwen3Features)
        assert transfer.feature_forward.model is model
    else:
        compile_fn.assert_called_once()
        assert isinstance(compile_fn.call_args.args[0], PackedQwen3Features)
        assert compile_fn.call_args.kwargs == {
            'backend': 'inductor', 'fullgraph': True, 'dynamic': True,
            'options': {'emulate_precision_casts': True},
        }
        assert transfer.feature_forward is compile_fn.return_value
    # Repeated setup must reuse the loaded model and selected execution path.
    transfer.setup()
    assert load_model.call_count == 1
    assert load_data.call_count == 1


@pytest.mark.skipif(os.environ.get('SPECDECODE_TEST_PACKED_CUDA') != '1',
                    reason='opt-in CUDA workload; select an idle GPU first')
def test_real_fa2_compiled_features_and_document_isolation():
    from transformers import Qwen3Config, Qwen3Model
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

    lazy_import_flash_attention('flash_attention_2')

    assert torch.cuda.is_available()
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16,
                        max_position_embeddings=128, attention_dropout=0.0)
    config._attn_implementation = 'flash_attention_2'
    model = Qwen3Model(config).to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    layers = [0, 1, 2]
    eager = PackedQwen3Features(model, layers).eval()
    compiled = torch.compile(eager, backend='inductor', fullgraph=True, dynamic=True,
                             options={'emulate_precision_casts': True})
    # An independent SDPA model supplies individual-document reference features.
    ref_config = Qwen3Config.from_dict(config.to_dict())
    ref_config._attn_implementation = 'sdpa'
    reference = Qwen3Model(ref_config).to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    reference.load_state_dict(model.state_dict())
    selected = {}
    handles = []
    for index, layer in enumerate(reference.layers):
        def capture(module, inputs, index=index):
            selected[index] = inputs[0][0].clone()
        handles.append(layer.register_forward_pre_hook(capture))
    handles.append(reference.norm.register_forward_pre_hook(
        lambda module, inputs: selected.__setitem__(2, inputs[0][0].clone())))
    try:
        with torch.no_grad():
            # Different token totals, document counts and maximum document lengths.
            for lengths in ([5, 1, 3], [4, 2], [1, 1, 1], [5, 1, 3]):
                total = sum(lengths)
                ids = torch.randint(0, 128, (1, total), device='cuda')
                pos = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0).cuda()
                cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32).cuda()
                actual = compiled(ids, pos, cu, max(lengths))
                torch.testing.assert_close(actual, eager(ids, pos, cu, max(lengths)), atol=0.02, rtol=0.02)
                expected = []
                for part in ids.split(lengths, dim=1):
                    reference(part, use_cache=False)
                    expected.append(torch.stack([selected[i] for i in layers] + [selected[2]], dim=1))
                torch.testing.assert_close(actual, torch.cat(expected), atol=0.03, rtol=0.03)
                changed = ids.clone()
                changed[:, :lengths[0]] = (changed[:, :lengths[0]] + 7) % 128
                altered = compiled(changed, pos, cu, max(lengths))
                torch.testing.assert_close(actual[lengths[0]:], altered[lengths[0]:], atol=0, rtol=0)
    finally:
        for handle in handles:
            handle.remove()
