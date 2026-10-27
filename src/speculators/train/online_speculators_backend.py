"""Native Speculators/vLLM teacher-forcing with per-request file transfer.

Unlike the local ``online`` backend, the verifier runs in a separate vLLM
server. The existing sampler still decides the trainer packing; each document
is sent as exact, already truncated token IDs through the native v0.9 client.
"""
import inspect
from pathlib import Path

import openai
import torch
from hs_connectors import FileBackend, FileTransfer, HiddenStatesBackend, HiddenStatesTransfer

from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import generate_hidden_states


class SpeculatorsOnlineTransfer(HiddenStatesTransfer):
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path
        self.layer_ids = list(args.target_layer_ids)
        self.file_transfer = FileTransfer(Path(args.hidden_states_path or '/tmp/hidden_states'))
        # cuda currently has hs-connectors 0.8 (string handle); v0.9 sources
        # accept the complete kv_transfer_params mapping instead.
        self._legacy_file_transfer = 'handle' in inspect.signature(
            self.file_transfer.get_generated).parameters
        self.data = None
        self.client = None

    def setup(self):
        if self.client is not None:
            return
        from datasets import load_from_disk
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.args.verifier_name_or_path, local_files_only=True)
        self.final_layer = config.num_hidden_layers
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError('online_speculators requires distinct explicit target_layer_ids')
        if any(type(i) is not int or not 0 <= i <= self.final_layer for i in self.layer_ids):
            raise ValueError('online_speculators target layers must be within the verifier layer range')
        # launch_vllm appends the final prenorm if it is absent. It cannot
        # request the same final layer twice. Adapt that native wire format
        # into the local five-auxiliary-plus-target contract after loading.
        self.server_layer_ids = list(self.layer_ids)
        if self.final_layer not in self.server_layer_ids:
            self.server_layer_ids.append(self.final_layer)
        data = load_from_disk(self.data_path).with_format('torch')
        client = openai.OpenAI(base_url=self.args.vllm_endpoint, api_key='EMPTY', max_retries=0)
        models = client.models.list()
        if not models.data:
            raise ValueError('online_speculators vLLM server returned no models')
        model_id = models.data[0].id
        if model_id != self.args.verifier_name_or_path:
            raise ValueError(f'vLLM model {model_id!r} differs from verifier {self.args.verifier_name_or_path!r}')
        self.model_id = model_id
        self.data = data
        self.client = client

    def _adapt_payload(self, loaded, ids):
        check_hidden_states(loaded, ids.tolist())
        hidden = loaded['hidden_states']
        if hidden.ndim != 3 or hidden.shape[1] != len(self.server_layer_ids):
            raise ValueError(f'vLLM hidden-state shape {tuple(hidden.shape)} differs from server layers {self.server_layer_ids}')
        indices = [self.server_layer_ids.index(i) for i in self.layer_ids]
        indices.append(self.server_layer_ids.index(self.final_layer))
        return {'hidden_states': hidden[:, indices], 'token_ids': loaded['token_ids']}

    def get_packed(self, file_indices, max_tokens=None):
        if not len(file_indices):
            return []
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError('max_tokens must be positive')
        self.setup()
        rows = []
        for index in file_indices:
            ids = self.data[int(index)]['input_ids']
            if ids.ndim != 1 or not ids.numel() or ids.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError(f'online_speculators sample {index}: expected nonempty integer input_ids')
            limit = getattr(self.args, 'max_sample_seq_len', None)
            if limit is not None:
                ids = ids[:limit]
            if max_tokens is not None:
                ids = ids[:max_tokens]
            rows.append(ids.long())
        if max_tokens is not None and sum(map(len, rows)) > max_tokens:
            raise ValueError('online_speculators packing exceeds per-rank token budget')
        results = []
        for ids in rows:
            handle = None
            try:
                handle, params = generate_hidden_states(
                    self.client, self.model_id, {'input_ids': ids.tolist()},
                    timeout=self.args.request_timeout, max_retries=self.args.max_retries,
                )
                loaded = self.file_transfer.get_generated(
                    handle if self._legacy_file_transfer else params)
                if loaded is None:
                    raise ValueError(f'vLLM hidden-state payload is missing for {handle}')
                results.append(self._adapt_payload(loaded, ids))
            finally:
                if handle is not None:
                    self.file_transfer.delete(handle)
        return results

    def get_cached(self, file_idx):
        return self.get_packed([file_idx])[0]

    def get_generated(self, params):
        raise RuntimeError('online_speculators requests and validates exact token IDs in get_packed')


@HiddenStatesBackend.register('online_speculators')
class SpeculatorsOnlineBackend(HiddenStatesBackend):
    add_train_args = staticmethod(FileBackend.add_train_args)
    add_launch_args = staticmethod(FileBackend.add_launch_args)
    build_kv_transfer_config = staticmethod(FileBackend.build_kv_transfer_config)

    @staticmethod
    def from_train_args(args, data_path):
        return SpeculatorsOnlineTransfer(args, data_path)
