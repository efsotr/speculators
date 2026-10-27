"""Per-row prefix caps must agree across teacher, tensors and packing."""
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from pydantic import ValidationError
from safetensors.torch import save_file

from speculators.train.config.schema import DataArgs
from speculators.train.data import ArrowDataset, CollateFn
from speculators.train.online_backend import OnlineTransfer
from speculators.train.recovery import SampleUnavailable


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / 'prepared'
    Dataset.from_dict({
        'input_ids': [[1, 2, 3, 4, 5], [6, 7]],
        'loss_mask': [[0, 0, 1, 1, 1], [0, 1]],
        'seq_len': [5, 2],
    }).with_format('torch').save_to_disk(str(path))
    cache = path / 'hidden_states'
    cache.mkdir()
    for i, ids in enumerate(([1, 2, 3, 4, 5], [6, 7])):
        save_file({'token_ids': torch.tensor(ids),
                   'hidden_states': torch.arange(len(ids) * 6).reshape(len(ids), 3, 2).float()},
                  str(cache / f'hs_{i}.safetensors'))
    return path


def test_file_cap_keeps_all_tensors_positions_and_packing_aligned(prepared):
    ds = ArrowDataset(max_len=16, datapath=prepared, max_sample_seq_len=3,
                      hidden_states_dtype=torch.float32, on_missing='raise')
    assert ds.approx_lengths == [3, 2]
    sample = ds[0]
    assert sample['input_ids'].tolist() == [1, 2, 3]
    assert sample['loss_mask'].tolist() == [0, 0, 1]
    assert sample['position_ids'].tolist() == [0, 1, 2]
    assert sample['lengths'].tolist() == [3]
    expected = torch.arange(30).reshape(5, 3, 2).float()[:3]
    torch.testing.assert_close(sample['hidden_states'], expected[:, :-1].flatten(1))
    torch.testing.assert_close(sample['verifier_last_hidden_states'], expected[:, -1])
    packed = CollateFn(16, hidden_size=2, num_target_layers=2,
                       dtype=torch.float32)([sample, ds[1]])
    assert packed['input_ids'][0, :5].tolist() == [1, 2, 3, 6, 7]
    assert packed['document_ids'][0, :6].tolist() == [0, 0, 0, 1, 1, -1]
    assert packed['position_ids'][0, :5].tolist() == [0, 1, 2, 0, 1]


def test_default_does_not_truncate(prepared):
    ds = ArrowDataset(max_len=16, datapath=prepared, on_missing='raise')
    assert ds.approx_lengths == [5, 2]
    assert ds[0]['input_ids'].tolist() == [1, 2, 3, 4, 5]
    assert DataArgs().max_sample_seq_len is None


@pytest.mark.parametrize('limit', [0, -1])
def test_nonpositive_cap_rejected(prepared, limit):
    with pytest.raises(ValidationError):
        DataArgs(max_sample_seq_len=limit)
    with pytest.raises(ValueError):
        ArrowDataset(max_len=16, datapath=prepared, max_sample_seq_len=limit)


def test_full_file_cache_mismatch_after_cap_is_still_rejected(prepared):
    save_file({'token_ids': torch.tensor([1, 2, 3, 4, 99]),
               'hidden_states': torch.zeros(5, 3, 2)},
              str(prepared / 'hidden_states' / 'hs_0.safetensors'))
    ds = ArrowDataset(max_len=16, datapath=prepared, max_sample_seq_len=3)
    with pytest.warns(UserWarning, match="match input ids"):
        assert isinstance(ds[0], SampleUnavailable)


@pytest.mark.parametrize('limit, expected_length', [(None, 5), (3, 3)])
def test_online_teacher_only_forwards_capped_tokens(prepared, limit, expected_length):
    transfer = OnlineTransfer(SimpleNamespace(target_layer_ids=[0, 1],
                                              max_sample_seq_len=limit), prepared)
    transfer.data = Dataset.load_from_disk(str(prepared)).with_format('torch')
    transfer.device = torch.device('cpu')
    calls = []

    class FakeTeacher:
        config = SimpleNamespace(vocab_size=100)

        def __call__(self, input_ids, *args, **kwargs):
            calls.append(input_ids.clone())
            features = input_ids.unsqueeze(-1).expand(-1, -1, 2).float()
            return torch.stack([features[0], features[0] + 1, features[0] + 2], dim=1)

    transfer.model = FakeTeacher()
    transfer.feature_forward = transfer.model
    ds = ArrowDataset(max_len=16, datapath=prepared, transfer=transfer,
                      max_sample_seq_len=limit, hidden_states_dtype=torch.float32)
    sample = ds[0]
    assert len(calls) == 1
    assert calls[0].shape == (1, expected_length)
    assert sample['input_ids'].shape == (expected_length,)
    assert sample['hidden_states'].shape == (expected_length, 4)
    assert sample['verifier_last_hidden_states'].shape == (expected_length, 2)
    assert sample['loss_mask'].shape == (expected_length,)
    assert not hasattr(transfer, "captured")
