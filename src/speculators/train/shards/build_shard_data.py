"""Create an aligned HF data-path and row index without copying hidden tensors.

Only exact token sequences and the source's initial unsupervised prompt prefix
are accepted. Loss masks are inherited, never invented for prompt-only caches.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow  # noqa: F401; before torch
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from safetensors import safe_open


def identity(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest()


def build(directory, source_paths, output, report_path):
    directory, output, report_path = map(Path, (directory, output, report_path))
    staging = output.with_name(output.name + '.incomplete')
    if output.exists() or staging.exists():
        raise ValueError('Output or staging directory already exists; choose a new output')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['format'] != 'sample-keyed-safetensors-v1':
        raise ValueError('Unsupported shard manifest format')
    sources = [load_from_disk(str(p)).with_format(None) for p in source_paths]
    lookup = defaultdict(list)
    for source_idx, data in enumerate(sources):
        offset = 0
        for batch in data.select_columns(['input_ids', 'loss_mask', 'seq_len']).iter(batch_size=1000):
            for i, (tokens, mask, length) in enumerate(zip(batch['input_ids'], batch['loss_mask'], batch['seq_len'], strict=True)):
                if len(tokens) != len(mask) or length != len(tokens) or not tokens:
                    raise ValueError(f'Invalid source row: {source_idx}/{offset+i}')
                if any(v not in (0, 1) for v in mask):
                    raise ValueError('Source loss masks must be binary')
                lookup[identity(tokens)].append((source_idx, offset+i))
                prompt_len = next((j for j, v in enumerate(mask) if v), len(mask))
                if 0 < prompt_len < len(tokens):
                    lookup[identity(tokens[:prompt_len])].append((source_idx, offset+i))
            offset += len(batch['input_ids'])
        print(json.dumps(dict(stage='indexed_source', path=str(source_paths[source_idx]), rows=len(data))), flush=True)

    stats = dict(samples=0, tokens=0, supervised_tokens=0, zero_loss_rows=0,
                 exact_matches=0, prompt_prefix_matches=0, unmatched=0, ambiguous_masks=0)
    rows, source_rows, examples, failures = [], [], [], []
    frequencies = Counter()
    # Keep only tokens/masks (not hidden states) in memory while building Arrow.
    def generate():
        for shard_idx, shard in enumerate(manifest['shards']):
            name = shard['file']
            if Path(name).name != name:
                raise ValueError('Shard filename must be a basename')
            with safe_open(directory / name, framework='np') as handle:
                keys = sorted(k[:-len('.token_ids')] for k in handle.keys() if k.endswith('.token_ids'))
                if len(keys) != shard['samples'] or len(handle.keys()) != 2 * len(keys):
                    raise ValueError(f'Invalid sample/key count: {name}')
                for key in keys:
                    tokens = handle.get_tensor(f'{key}.token_ids').tolist()
                    if (handle.get_slice(f'{key}.hidden_states').get_shape() != [len(tokens), 4, 2560]
                            or handle.get_slice(f'{key}.hidden_states').get_dtype() != 'BF16'
                            or handle.get_slice(f'{key}.token_ids').get_dtype() != 'I64'):
                        raise ValueError(f'Invalid tensor shape/dtype: {name}/{key}')
                    candidates = []
                    for source_idx, row_idx in lookup.get(identity(tokens), []):
                        original = sources[source_idx][row_idx]
                        if original['input_ids'][:len(tokens)] == tokens:
                            mask = original['loss_mask'][:len(tokens)]
                            candidates.append((source_idx, row_idx, original, mask))
                    if not candidates:
                        stats['unmatched'] += 1
                        failures.append(dict(shard=name, key=key, tokens=len(tokens)))
                        continue
                    if len({tuple(c[3]) for c in candidates}) != 1:
                        stats['ambiguous_masks'] += 1
                        failures.append(dict(shard=name, key=key, reason='conflicting source masks'))
                        continue
                    source_idx, row_idx, original, mask = candidates[0]
                    exact = len(tokens) == len(original['input_ids'])
                    stats['exact_matches' if exact else 'prompt_prefix_matches'] += 1
                    stats['samples'] += 1
                    stats['tokens'] += len(tokens)
                    supervised = sum(mask)
                    stats['supervised_tokens'] += supervised
                    stats['zero_loss_rows'] += int(supervised == 0)
                    frequencies.update(t for t, active in zip(tokens, mask, strict=True) if active)
                    rows.append([shard_idx, key])
                    source_rows.append([source_idx, row_idx])
                    if len(examples) < 10:
                        examples.append(dict(shard=name, key=key, source=source_idx,
                                             source_row=row_idx, exact=exact, supervised_tokens=supervised))
                    yield dict(input_ids=tokens, loss_mask=mask, seq_len=len(tokens))
            print(json.dumps(dict(stage='aligned_shard', shard=name, **stats)), flush=True)

    staging.parent.mkdir(parents=True, exist_ok=True)
    # Explicit schema also handles a completely unmatched corpus.
    features = Features(dict(input_ids=Sequence(Value('int64')),
                             loss_mask=Sequence(Value('int64')), seq_len=Value('int64')))
    # from_generator caches by source/closure identity; a fresh staging cache avoids
    # reusing stale results and keeps the investigation local to the new output.
    import tempfile
    with tempfile.TemporaryDirectory(prefix=output.name + '.build-', dir=output.parent) as cache:
        fingerprint = hashlib.sha256(json.dumps(dict(
            format='specdecode-row-shards-v1', manifest=manifest,
            sources=[ds._fingerprint for ds in sources]), sort_keys=True).encode()).hexdigest()
        data = Dataset.from_generator(generate, features=features, cache_dir=cache,
                                      fingerprint=fingerprint)
        report = dict(shard_directory=str(directory), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      sources=[dict(path=str(p), rows=len(ds), fingerprint=ds._fingerprint)
                               for p, ds in zip(source_paths, sources, strict=True)],
                      **stats, expected_samples=manifest['samples'], examples=examples,
                      failures=failures, source_rows=source_rows,
                      training_ready=stats['zero_loss_rows'] == 0 and stats['samples'] == manifest['samples'],
                      supervision='Inherited from matched source rows; prompt-prefix masks remain zero')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        if failures or stats['samples'] != manifest['samples']:
            raise ValueError(f'Not all shards aligned; see {report_path}')
        data.with_format('torch').save_to_disk(str(staging))
    saved = load_from_disk(str(staging))
    index = dict(format='specdecode-row-shards-v1', dataset_fingerprint=saved._fingerprint,
                 shard_directory=os.path.relpath(directory, output),
                 shards=[s['file'] for s in manifest['shards']], rows=rows,
                 training_ready=report['training_ready'])
    (staging / 'hidden_states.index.json').write_text(json.dumps(index) + '\n')
    (staging / 'shard_alignment.json').write_text(json.dumps(report, indent=2) + '\n')
    # Use the installed trainer's expected frequency mapping format.
    import torch
    torch.save(dict(frequencies), staging / 'token_freq.pt')
    staging.rename(output)
    print(json.dumps(dict(data_path=str(output), **stats, training_ready=report['training_ready'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--source-data', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    build(args.directory, args.source_data, args.output, args.report)


if __name__ == '__main__':
    main()
