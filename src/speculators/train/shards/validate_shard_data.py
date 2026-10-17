"""CPU-only full row/token/header validation; never materialize hidden tensors."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path

import pyarrow  # noqa: F401
from datasets import load_from_disk
from safetensors import safe_open

from speculators.train.shards.shard_backend import INDEX_NAME, ShardTransfer


def validate(data_path, require_training_ready=False, hidden_states_path=None):
    transfer = ShardTransfer(data_path, hidden_states_path)
    data = load_from_disk(str(data_path)).with_format(None)
    handles = OrderedDict()
    rows, tokens, supervised_tokens, zero_loss_rows = 0, 0, 0, 0
    try:
        for batch in data.select_columns(['input_ids', 'loss_mask', 'seq_len']).iter(batch_size=1000):
            for ids, mask, length in zip(batch['input_ids'], batch['loss_mask'], batch['seq_len'], strict=True):
                shard, key = transfer.index['rows'][rows]
                if shard not in handles:
                    handles[shard] = safe_open(transfer.directory / transfer.index['shards'][shard], framework='np')
                    handles[shard].__enter__()
                    if len(handles) > 2:
                        _, old = handles.popitem(last=False)
                        old.__exit__(None, None, None)
                handles.move_to_end(shard)
                handle = handles[shard]
                if (length != len(ids) or len(mask) != length or not length or
                        any(v not in (0, 1) for v in mask) or
                        handle.get_tensor(f'{key}.token_ids').tolist() != ids or
                        handle.get_slice(f'{key}.token_ids').get_dtype() != 'I64' or
                        handle.get_slice(f'{key}.hidden_states').get_dtype() != 'BF16' or
                        handle.get_slice(f'{key}.hidden_states').get_shape() != [length, 4, 2560]):
                    raise ValueError(f'Shard/data mismatch at row {rows}')
                rows += 1
                tokens += length
                supervised_tokens += sum(mask)
                zero_loss_rows += int(not any(mask))
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)
    ready = rows > 0 and zero_loss_rows == 0
    if ready != transfer.index['training_ready']:
        raise ValueError('Incorrect training_ready flag in shard index')
    result = dict(rows=rows, tokens=tokens, supervised_tokens=supervised_tokens,
                  zero_loss_rows=zero_loss_rows, training_ready=ready,
                  hidden_states_path=str(transfer.directory),
                  scope='All row tokens, masks, tensor headers; no hidden tensor values/provenance')
    print(json.dumps(result), flush=True)
    if require_training_ready and not ready:
        raise ValueError('Shard dataset has rows without supervision; see shard_alignment.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True, type=Path)
    parser.add_argument('--hidden-states-path', type=Path)
    parser.add_argument('--require-training-ready', action='store_true')
    args = parser.parse_args()
    validate(args.data_path, args.require_training_ready, args.hidden_states_path)


if __name__ == '__main__':
    main()
