"""Packed, frozen Qwen3 features using FlashAttention 2 and torch.compile.

The default DataLoader path computes one batch per rank and sampler packing
in a single verifier forward. FA2 is fixed; torch.compile defaults to enabled.
Explicit varlen boundaries isolate its documents. Both backends retain the file
backend's prenorm contract; online returns CPU features, online_gpu retains CUDA.
This implementation targets Transformers 5.14.1 and flash-attn 2.8.4.
"""
import os

import torch
from hs_connectors import HiddenStatesBackend, HiddenStatesTransfer


class PackedQwen3Features(torch.nn.Module):
    """Return only selected decoder inputs and final prenorm, without hooks.

    Mirrors the dense Qwen3 decoder loop in Transformers 5.14.1. The final
    norm/LM head are unnecessary because the consumer needs prenorm features.
    """

    def __init__(self, model, layer_ids):
        super().__init__()
        self.model = model
        self.layer_ids = tuple(layer_ids)

    def forward(self, input_ids, position_ids, cu_seqlens, max_seqlen):
        hidden = self.model.embed_tokens(input_ids)
        position_embeddings = self.model.rotary_emb(hidden, position_ids)
        features = {}
        for index, layer in enumerate(self.model.layers):
            if index in self.layer_ids:
                features[index] = hidden[0]
            hidden = layer(
                hidden,
                attention_mask=None,
                # RoPE is already computed with document-local position_ids.
                # Do not ask FA2 to infer packing from tensor values.
                position_ids=None,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
            )
        features[len(self.model.layers)] = hidden[0]
        return torch.stack(
            [features[index] for index in self.layer_ids] + [hidden[0]], dim=1,
        )


class OnlineTransfer(HiddenStatesTransfer):
    returns_cuda_tensors = False

    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path
        self.model = None
        self.feature_forward = None
        self.layer_ids = list(args.target_layer_ids)

    def setup(self):
        if self.model is not None:
            return
        from datasets import load_from_disk
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            self.args.verifier_name_or_path, local_files_only=True,
        )
        if config.model_type != 'qwen3':
            raise ValueError('packed online features require dense Qwen3')
        if any(kind != 'full_attention' for kind in config.layer_types):
            raise ValueError('packed online features currently require full-attention layers')
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError('online target_layer_ids must be nonempty and unique')
        if any(type(i) is not int or not 0 <= i <= config.num_hidden_layers
               for i in self.layer_ids):
            raise ValueError('online target_layer_ids are decoder input indices including final prenorm')
        dtype = getattr(self.args, 'hidden_states_dtype', 'bfloat16')
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
        if dtype != torch.bfloat16:
            raise ValueError('packed online FlashAttention 2 requires hidden_states_dtype: bfloat16')
        # Import failures must surface; do not silently fall back to SDPA/eager.
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

        # HF explicitly requires eager kernel registration before fullgraph capture.
        lazy_import_flash_attention('flash_attention_2')

        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        data = load_from_disk(self.data_path).with_format('torch')
        model = AutoModel.from_pretrained(
            self.args.verifier_name_or_path, dtype=dtype,
            attn_implementation='flash_attention_2', local_files_only=True,
        ).to(device).eval().requires_grad_(False)
        feature_forward = PackedQwen3Features(model, self.layer_ids).eval()
        if getattr(self.args, 'online_compile_hidden_states', True):
            feature_forward = torch.compile(
                feature_forward,
                backend='inductor', fullgraph=True, dynamic=True,
                # Preserve eager BF16 rounding through fused RMSNorm/RoPE operations.
                options={'emulate_precision_casts': True},
            )
        # Publish only after setup succeeds. First invocation performs compilation.
        self.device = device
        self.data = data
        self.feature_forward = feature_forward
        self.model = model

    @torch.no_grad()
    def get_packed(self, file_indices, max_tokens=None):
        """Compute one unpadded packing, then split its features by row.

        max_tokens is the sampler's per-rank budget. A single oversized row is
        prefix-truncated exactly as the sampler/CollateFn would truncate it.
        A multi-row overflow is an alignment error, never silently dropped.
        """
        if not len(file_indices):
            return []
        self.setup()
        rows = []
        for file_idx in file_indices:
            ids = self.data[int(file_idx)]['input_ids']
            if ids.ndim != 1 or ids.numel() == 0 or ids.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
            ):
                raise ValueError(f'online sample {file_idx}: expected nonempty integer input_ids')
            limit = getattr(self.args, 'max_sample_seq_len', None)
            if limit is not None:
                ids = ids[:limit]
            if max_tokens is not None:
                if max_tokens <= 0:
                    raise ValueError('max_tokens must be positive')
                ids = ids[:max_tokens]
            ids = ids.long()
            if int(ids.min()) < 0 or int(ids.max()) >= self.model.config.vocab_size:
                raise ValueError(f'online sample {file_idx}: input_ids exceed verifier vocabulary')
            rows.append(ids)
        lengths = [len(ids) for ids in rows]
        total = sum(lengths)
        if max_tokens is not None and total > max_tokens:
            raise ValueError(f'online packing has {total} tokens, exceeds budget {max_tokens}')
        ids = torch.cat(rows).unsqueeze(0).to(self.device)
        positions = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0).to(self.device)
        cu_seqlens = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(
            0, dtype=torch.int32,
        ).to(self.device)
        hidden = self.feature_forward(ids, positions, cu_seqlens, max(lengths))
        if not self.returns_cuda_tensors:
            hidden = hidden.cpu()
        return [
            {'hidden_states': features, 'token_ids': tokens}
            for features, tokens in zip(hidden.split(lengths), rows, strict=True)
        ]

    def get_cached(self, file_idx):
        # Keep the single-row lookup contract for direct Dataset indexing/tools.
        return self.get_packed([file_idx])[0]

    def get_generated(self, handle):
        raise RuntimeError('Local online forwards do not use generated handles')


class OnlineGPUTransfer(OnlineTransfer):
    """Reuse packed teacher compute while retaining features on the rank GPU."""

    returns_cuda_tensors = True


@HiddenStatesBackend.register('online')
class OnlineBackend(HiddenStatesBackend):
    transfer_type = OnlineTransfer

    @staticmethod
    def add_train_args(parser):
        pass

    @staticmethod
    def add_launch_args(parser):
        pass

    @classmethod
    def from_train_args(cls, args, data_path):
        if args.num_workers != 0:
            raise ValueError('online verifier requires data.num_workers: 0 (CUDA in training process)')
        if not args.target_layer_ids:
            raise ValueError('online verifier requires explicit draft.target_layer_ids')
        return cls.transfer_type(args, data_path)


@HiddenStatesBackend.register('online_gpu')
class OnlineGPUBackend(OnlineBackend):
    transfer_type = OnlineGPUTransfer
