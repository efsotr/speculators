"""Native vLLM client/transfer adapter contracts without a GPU or server."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file

import speculators.train.online_speculators_backend as backend


def transfer(tmp_path):
    args = SimpleNamespace(target_layer_ids=[2, 10, 18, 26, 36],
                           hidden_states_path=str(tmp_path), max_sample_seq_len=3,
                           request_timeout=10, max_retries=0)
    instance = backend.SpeculatorsOnlineTransfer(args, 'unused')
    instance.data = Dataset.from_dict({'input_ids': [[1, 2, 3, 4], [5, 6]]}).with_format('torch')
    instance.client = Mock()
    instance.model_id = 'verifier'
    instance.final_layer = 36
    instance.server_layer_ids = [2, 10, 18, 26, 36]
    return instance


def test_exact_prefix_requests_packing_and_final_layer_duplication(tmp_path, monkeypatch):
    instance = transfer(tmp_path)
    requests = []

    def generate(client, model, item, **kwargs):
        ids = torch.tensor(item['input_ids'])
        requests.append(ids.tolist())
        filename = tmp_path / f'{len(requests)}.safetensors'
        hs = torch.arange(5).view(1, 5, 1).expand(len(ids), 5, 2).float().contiguous()
        save_file({'token_ids': ids, 'hidden_states': hs}, str(filename))
        return str(filename), {'hidden_states_path': str(filename)}

    monkeypatch.setattr(backend, 'generate_hidden_states', generate)
    samples = instance.get_packed([0, 1], max_tokens=5)
    assert requests == [[1, 2, 3], [5, 6]]
    assert samples[0]['hidden_states'].shape == (3, 6, 2)
    assert samples[0]['hidden_states'][0, :, 0].tolist() == [0, 1, 2, 3, 4, 4]
    assert not list(tmp_path.glob('*.safetensors'))
    assert instance.get_packed([]) == []


def test_payload_with_separate_final_slot_is_not_duplicated(tmp_path):
    instance = transfer(tmp_path)
    instance.layer_ids = [2, 10, 18, 26, 34]
    instance.server_layer_ids = [2, 10, 18, 26, 34, 36]
    ids = torch.tensor([1, 2])
    hs = torch.arange(12).view(2, 6, 1).float()
    result = instance._adapt_payload({'hidden_states': hs, 'token_ids': ids}, ids)
    torch.testing.assert_close(result['hidden_states'], hs)


def test_overflow_rejected_before_request(tmp_path, monkeypatch):
    instance = transfer(tmp_path)
    generate = Mock()
    monkeypatch.setattr(backend, 'generate_hidden_states', generate)
    with pytest.raises(ValueError, match='packing exceeds'):
        instance.get_packed([0, 1], max_tokens=4)
    generate.assert_not_called()


def test_invalid_payload_still_deleted(tmp_path, monkeypatch):
    instance = transfer(tmp_path)
    filename = tmp_path / 'bad.safetensors'
    save_file({'token_ids': torch.tensor([9]), 'hidden_states': torch.zeros(1, 5, 2)}, str(filename))
    monkeypatch.setattr(backend, 'generate_hidden_states', Mock(return_value=(str(filename), {'hidden_states_path': str(filename)})))
    with pytest.raises(ValueError, match="Token ids don't match"):
        instance.get_packed([0], max_tokens=3)
    assert not filename.exists()


def test_layer_shape_mismatch_rejected(tmp_path):
    instance = transfer(tmp_path)
    ids = torch.tensor([1])
    with pytest.raises(ValueError, match='differs from server layers'):
        instance._adapt_payload({'token_ids': ids, 'hidden_states': torch.zeros(1, 4, 2)}, ids)
