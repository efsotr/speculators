"""Row-indexed, read-only safetensors shard backend for Speculators."""
import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

# Keep the environment's Arrow-before-Torch import ordering.
import pyarrow  # noqa: F401
from hs_connectors import FileBackend, HiddenStatesBackend, HiddenStatesTransfer
from safetensors import safe_open

INDEX_NAME = 'hidden_states.index.json'


class ShardTransfer(HiddenStatesTransfer):
    """Open shards lazily in each worker; never pickle mmap handles."""

    def __init__(self, data_path, hidden_states_path=None):
        from datasets import load_from_disk

        self.data_path = Path(data_path)
        self.index = json.loads((self.data_path / INDEX_NAME).read_text())
        if self.index['format'] != 'specdecode-row-shards-v1':
            raise ValueError('Unsupported hidden-state index format')
        data = load_from_disk(str(self.data_path))
        if (len(data) != len(self.index['rows']) or
                data._fingerprint != self.index['dataset_fingerprint']):
            raise ValueError('Shard index does not match the prepared dataset')
        self.directory = (Path(hidden_states_path) if hidden_states_path else
                          self.data_path / self.index['shard_directory'])
        for name in self.index['shards']:
            if Path(name).name != name or not (self.directory / name).is_file():
                raise ValueError(f'Missing or invalid shard: {name}')
        self._handles = OrderedDict()
        self._pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handles'] = OrderedDict()
        return state

    def close(self):
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()

    def get_cached(self, file_idx):
        if not 0 <= file_idx < len(self.index['rows']):
            return None
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard_idx, key = self.index['rows'][file_idx]
        if shard_idx not in self._handles:
            handle = safe_open(self.directory / self.index['shards'][shard_idx],
                               framework='pt', device='cpu')
            handle.__enter__()
            self._handles[shard_idx] = handle
            if len(self._handles) > 2:
                _, old = self._handles.popitem(last=False)
                old.__exit__(None, None, None)
        self._handles.move_to_end(shard_idx)
        handle = self._handles[shard_idx]
        return {name: handle.get_tensor(f'{key}.{name}')
                for name in ('hidden_states', 'token_ids')}

    def get_generated(self, handle):
        raise RuntimeError('Shard backend is read-only; generate caches with the file backend')

    def cache(self, handle, file_idx):
        raise RuntimeError('Shard backend is read-only')


@HiddenStatesBackend.register('shard')
class ShardBackend(HiddenStatesBackend):
    @staticmethod
    def add_train_args(parser):
        parser.add_argument('--shard-directory', default=None,
                            help='Override the shard directory recorded in the data-path index')

    @staticmethod
    def add_launch_args(parser):
        pass

    @staticmethod
    def from_train_args(args, data_path):
        standard = getattr(args, 'hidden_states_path', None)
        legacy = getattr(args, 'shard_directory', None)
        if standard and legacy and Path(standard).resolve() != Path(legacy).resolve():
            raise ValueError('--hidden-states-path and --shard-directory disagree')
        transfer = ShardTransfer(data_path, standard or legacy)
        if not transfer.index['training_ready']:
            raise ValueError('Shard data has rows without supervised tokens; see '
                             f'{Path(data_path) / "shard_alignment.json"}. '
                             'Prompt-only hidden states cannot train assistant responses.')
        if getattr(args, 'on_missing', 'raise') != 'raise':
            raise ValueError('Read-only shard training requires --on-missing raise')
        return transfer

    @staticmethod
    def build_kv_transfer_config(args):
        raise ValueError('Shard backend is read-only; use file to generate hidden states')


@HiddenStatesBackend.register('auto')
class AutoBackend(ShardBackend):
    @staticmethod
    def add_train_args(parser):
        # Backend argument destinations must be globally unique.
        pass

    @staticmethod
    def from_train_args(args, data_path):
        if (Path(data_path) / INDEX_NAME).is_file():
            return ShardBackend.from_train_args(args, data_path)
        # The resolver backfills defaults only for the selected backend. With
        # auto selected, the file backend's optional flag may be absent.
        file_args = SimpleNamespace(hidden_states_path=getattr(args, 'hidden_states_path', None))
        return FileBackend.from_train_args(file_args, data_path)

    @staticmethod
    def build_kv_transfer_config(args):
        return FileBackend.build_kv_transfer_config(args)
