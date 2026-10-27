"""Exercise real upstream parsing/loaders, shuffled row mapping and worker spawn."""
import json
import pickle
import runpy
import shutil
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow  # noqa: F401
import torch
from datasets import Dataset, load_from_disk
from safetensors.torch import save_file

from speculators.train.shards.build_shard_data import build
from speculators.train.shards.shard_backend import AutoBackend, ShardBackend, ShardTransfer
from speculators.train.shards.validate_shard_data import validate
from speculators.train.dataloader import create_train_val_loaders


class ShardTests(unittest.TestCase):
    def test_mapping_spawn_loaders_and_guards(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='speculators_shards_') as tmp:
            root = Path(tmp)
            source, shards, output = root/'source', root/'shards', root/'data'
            shards.mkdir()
            original = [dict(input_ids=[10+i,20+i,30+i,40+i], loss_mask=[0,1,1,1], seq_len=4) for i in range(6)]
            Dataset.from_list(original).with_format('torch').save_to_disk(str(source))
            order = [2,0,5,1,4,3]
            manifest = dict(format='sample-keyed-safetensors-v1', samples=6, shards=[])
            for shard_idx, indices in enumerate((order[:4], order[4:])):
                payload = {}
                for local, i in enumerate(indices):
                    key = f'{local:05d}'
                    payload[f'{key}.token_ids'] = torch.tensor(original[i]['input_ids'])
                    payload[f'{key}.hidden_states'] = torch.full((4,4,2560), i, dtype=torch.bfloat16)
                name = f'shard-{shard_idx:05d}.safetensors'
                save_file(payload, shards/name)
                manifest['shards'].append(dict(file=name, samples=len(indices)))
            (shards/'manifest.json').write_text(json.dumps(manifest))
            result = build(shards, [source], output, root/'report.json')
            self.assertTrue(result['training_ready'])
            self.assertEqual(validate(output)['supervised_tokens'],18)
            transfer = ShardTransfer(output)
            # File order differs from source order; uneven shard sizes must work.
            for row, source_row in enumerate(order):
                item = transfer.get_cached(row)
                self.assertEqual(item['token_ids'].tolist(),original[source_row]['input_ids'])
                self.assertEqual(item['hidden_states'][0,0,0].item(),source_row)
            self.assertIsNone(transfer.get_cached(-1))
            self.assertIsNone(transfer.get_cached(6))
            restored = pickle.loads(pickle.dumps(transfer))
            self.assertEqual(restored.get_cached(4)['token_ids'].tolist(),original[4]['input_ids'])
            restored.close()
            transfer.close()
            args = SimpleNamespace(on_missing='raise', hidden_states_path=None)
            self.assertIsInstance(AutoBackend.from_train_args(args,str(output)),ShardTransfer)
            self.assertEqual(type(AutoBackend.from_train_args(args,str(source))).__name__,'FileTransfer')
            self.assertEqual(type(AutoBackend.from_train_args(SimpleNamespace(),str(source))).__name__,'FileTransfer')
            for workers in (0,2):
                train,val = create_train_val_loaders(
                    data_path=str(output), total_seq_len=16, hidden_states_dtype=torch.bfloat16,
                    transfer=ShardTransfer(output), vllm_endpoint='http://127.0.0.1:1/v1',
                    on_missing='raise', verifier_name_or_path='models/Qwen3-4B',
                    request_timeout=1,max_retries=0,generation_validation_retries=0,
                    max_consecutive_generation_failures=1, hidden_size=2560,num_target_layers=3,
                    num_workers=workers,prefetch_factor=1,preprocess=None,train_data_ratio=0.5)
                for loader in (train,val):
                    loader.pin_memory=False
                    iterator=iter(loader)
                    try:
                        batches=list(iterator)
                        self.assertTrue(batches)
                        self.assertTrue(all(b['error_records']==0 for b in batches))
                        self.assertTrue(all(b['hidden_states'].shape[-1]==7680 for b in batches))
                    finally:
                        if workers:
                            iterator._shutdown_workers()
            # Refuse an index reused with a different Arrow dataset.
            broken=load_from_disk(str(output)).select(range(5))
            broken_path=root/'broken'
            broken.save_to_disk(str(broken_path))
            (broken_path/'hidden_states.index.json').write_text((output/'hidden_states.index.json').read_text())
            with self.assertRaisesRegex(ValueError,'does not match'):
                ShardTransfer(broken_path)
            # A prompt-only cache remains a zero-mask diagnostic data-path.
            prompt_shards=root/'prompt_shards'
            prompt_shards.mkdir()
            save_file({'00000.token_ids':torch.tensor([10]),
                       '00000.hidden_states':torch.zeros((1,4,2560),dtype=torch.bfloat16)},
                      prompt_shards/'prompt.safetensors')
            (prompt_shards/'manifest.json').write_text(json.dumps(dict(
                format='sample-keyed-safetensors-v1',samples=1,
                shards=[dict(file='prompt.safetensors',samples=1)])))
            prompt_data=root/'prompt_data'
            self.assertFalse(build(prompt_shards,[source],prompt_data,root/'prompt_report.json')['training_ready'])
            self.assertEqual(validate(prompt_data)['supervised_tokens'],0)
            with self.assertRaisesRegex(ValueError,'without supervised'):
                ShardBackend.from_train_args(args,str(prompt_data))
            # Backend options are discovered by the real upstream config parser.
            from speculators.train.config import TrainConfig
            cfg=TrainConfig.resolve(['--verifier-name-or-path','models/Qwen3-4B',
                '--data-path',str(output),'--hidden-states-backend','shard',
                '--shard-directory',str(shards),'--train-data-ratio','0.95'])
            self.assertEqual(cfg.flatten()['shard_directory'],str(shards))
            self.assertEqual(cfg.data.train_data_ratio,0.95)
            # Exercise the native Speculators CLI -> Typer -> TrainConfig path.
            # Replace only the model-training stage with real data loading on CPU.
            relocated = root/'relocated_shards'
            shutil.copytree(shards, relocated)
            index_path = output/'hidden_states.index.json'
            index = json.loads(index_path.read_text())
            index['shard_directory'] = 'missing-index-location'
            index_path.write_text(json.dumps(index))
            file_data, file_cache = root/'file_data', root/'file_cache'
            load_from_disk(str(output)).save_to_disk(str(file_data))
            file_cache.mkdir()
            for i, source_row in enumerate(order):
                save_file({'token_ids':torch.tensor(original[source_row]['input_ids']),
                           'hidden_states':torch.full((4,4,2560),100+source_row,dtype=torch.bfloat16)},
                          file_cache/f'hs_{i}.safetensors')

            def cli_load(data_path, cache_path, backend, expected_value):
                called = []

                def load_only(config):
                    from hs_connectors import HiddenStatesBackend
                    args = SimpleNamespace(**config.flatten())
                    self.assertEqual(args.hidden_states_path,str(cache_path))
                    transfer = HiddenStatesBackend.registry[args.hidden_states_backend].from_train_args(args,args.data_path)
                    train, val = create_train_val_loaders(
                        data_path=args.data_path,total_seq_len=args.total_seq_len,
                        hidden_states_dtype=torch.bfloat16,transfer=transfer,
                        vllm_endpoint='http://127.0.0.1:1/v1',on_missing=args.on_missing,
                        verifier_name_or_path=args.verifier_name_or_path,request_timeout=1,max_retries=0,
                        generation_validation_retries=0,max_consecutive_generation_failures=1,
                        hidden_size=2560,num_target_layers=3,num_workers=0,prefetch_factor=1,
                        preprocess=None,train_data_ratio=args.train_data_ratio)
                    first = train.dataset[0]
                    self.assertEqual(first['input_ids'].tolist(),original[order[0]]['input_ids'])
                    self.assertEqual(first['hidden_states'][0,0].item(),expected_value)
                    for loader in (train,val):
                        loader.pin_memory=False
                        batches=list(loader)
                        self.assertTrue(batches)
                        self.assertTrue(all(b['error_records']==0 for b in batches))
                    if isinstance(transfer,ShardTransfer):
                        transfer.close()
                    called.append(True)

                argv=['speculators','train','--verifier-name-or-path','models/Qwen3-4B',
                      '--data-path',str(data_path),'--hidden-states-path',str(cache_path),
                      '--hidden-states-backend',backend,'--on-missing','raise',
                      '--train-data-ratio','0.5','--total-seq-len','16','--num-workers','0']
                with patch.object(sys,'argv',argv), patch('speculators.train.cli.main',load_only):
                    with self.assertRaises(SystemExit) as finished:
                        runpy.run_module('speculators',run_name='__main__')
                    self.assertEqual(finished.exception.code,0)
                self.assertEqual(called,[True])

            for backend in ('auto','shard'):
                cli_load(output,relocated,backend,order[0])
            for backend in ('auto','file'):
                cli_load(file_data,file_cache,backend,100+order[0])
            self.assertEqual(validate(output,hidden_states_path=relocated)['rows'],6)
            with self.assertRaisesRegex(ValueError,'Missing or invalid shard'):
                ShardBackend.from_train_args(SimpleNamespace(hidden_states_path=str(root/'wrong')),str(output))
            with self.assertRaisesRegex(ValueError,'disagree'):
                ShardBackend.from_train_args(SimpleNamespace(hidden_states_path=str(relocated),shard_directory=str(shards)),str(output))
            index['shard_directory'] = '../shards'
            index_path.write_text(json.dumps(index))


if __name__ == '__main__':
    unittest.main()
