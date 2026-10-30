import logging
import time
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.stable_block_mask import StableBlockMasks
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
)
from speculators.models.utils import (
    conditional_torch_compile,
    flatten_rope_parameters,
    resolve_target_layer_ids,
    resolve_verifier_norm_class,
)

logger = logging.getLogger(__name__)

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
_compiled_create_block_mask = torch.compile(create_block_mask)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    supports_gradient_checkpointing = True  # noqa: D003  # Qwen3DFlashDecoderLayer inherits GradientCheckpointingLayer
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        # verifier_lm_head is reloaded from the verifier (see load_verifier_weights)
        # and excluded on save, so it is expected to be absent from checkpoints.
        # lm_head is handled per-instance in __init__: omitted only for full-vocab
        # drafts, where it is the exact frozen verifier projection.
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> nn.Module:
        """Build one draft decoder layer.

        DFlash-family variants override this factory when they wrap the shared
        attention and MLP with additional modules.
        """
        return Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            _compiled_create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                self._make_decoder_layer(config, layer_idx)
                for layer_idx in range(num_draft_layers)
            ]
        )
        # Both are optional: declared only for alternating/sliding attention.
        self.sliding_window = getattr(tl_config, "sliding_window", None)
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(
                getattr(tl_config, "layer_types", None) or []
            )
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        rotary_config = flatten_rope_parameters(config.transformer_layer_config)
        self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = resolve_verifier_norm_class(config)(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size

        # Warn if using DFlash with sample_from_anchor=True (may not be supported)
        if type(self).__name__ == "DFlashDraftModel" and config.sample_from_anchor:
            logger.warning(
                "DFlash with sample_from_anchor=True may not be supported in "
                "all inference engines (e.g., vLLM). Verify compatibility with your "
                "deployment target."
            )

        self.post_init()
        # Verifier-owned weights are reconstructed on load. Keep a reduced-vocab
        # lm_head serialized because current runtimes cannot derive it from the
        # full verifier head.
        #
        # Shadow the ClassVar lists with per-instance copies so full- and
        # reduced-vocabulary siblings cannot mutate each other's save rules.
        keys_to_ignore_on_save = list(type(self)._keys_to_ignore_on_save)  # noqa: SLF001
        keys_to_ignore_on_load_missing = list(
            type(self)._keys_to_ignore_on_load_missing  # noqa: SLF001
        )
        keys_to_ignore_on_save.append("embed_tokens.weight")
        if not self.use_draft_vocab:
            keys_to_ignore_on_save.append("lm_head.weight")
            keys_to_ignore_on_load_missing.append("lm_head.weight")
        self.__dict__["_keys_to_ignore_on_save"] = keys_to_ignore_on_save
        self.__dict__["_keys_to_ignore_on_load_missing"] = (
            keys_to_ignore_on_load_missing
        )

    def configure_compile(
        self, *, mode="default", activation_memory_budget=0.99,
        emulate_precision_casts=False, pad_target_hidden_states=False,
        target_seq_len=8192,
        whole_forward=False,
    ):
        """Compile layers in max-autotune, or the whole forward for Python DDP."""
        if mode not in {"default", "max-autotune"}:
            raise ValueError(f"Unsupported draft compile mode: {mode}")
        if not 0 <= activation_memory_budget <= 1:
            raise ValueError("activation_memory_budget must be between 0 and 1")
        if target_seq_len <= 0:
            raise ValueError("target_seq_len must be positive")
        # In this torch version the partitioner setting lives in functorch,
        # not torch._dynamo.config.
        import torch._functorch.config as partition_config
        import torch._inductor.config as inductor_config
        partition_config.activation_memory_budget = activation_memory_budget
        inductor_config.emulate_precision_casts = emulate_precision_casts
        self._pad_target_seq_len = target_seq_len if pad_target_hidden_states else None
        self._stable_block_masks = (
            StableBlockMasks()
            if mode == "max-autotune" and self._attn_impl == "simple_flex_attention"
            else None
        )
        original = getattr(type(self).forward, "_torchdynamo_orig_callable", type(self).forward)
        bound_forward = original.__get__(self, type(self))
        if mode == "max-autotune" and not whole_forward:
            # Keep the backbone loop and Module.__call__ (including FSDP hooks)
            # eager. Compile each layer's forward body inside that boundary.
            self.forward = bound_forward
            for layer in self.layers:
                layer_forward = getattr(
                    type(layer).forward, "_torchdynamo_orig_callable", type(layer).forward
                ).__get__(layer, type(layer))
                layer.forward = (
                    torch.compile(layer_forward, mode=mode)
                    if torch.cuda.is_available() else layer_forward
                )
        else:
            self.forward = (
                torch.compile(bound_forward, mode=mode)
                if torch.cuda.is_available() else bound_forward
            )

    def _pad_target_inputs(
        self, hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
        document_ids, position_ids,
    ):
        """Right-pad the entire base sequence so hidden states and mask ids agree."""
        width = getattr(self, "_pad_target_seq_len", None)
        if width is None:
            return (hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                    document_ids, position_ids)
        length = hidden_states.shape[1]
        if length > width:
            raise ValueError("Target sequence exceeds configured padded length")
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        values = (hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                  document_ids, position_ids)
        for value in values:
            if value.shape[1] != length:
                raise ValueError("Target hidden states and all token/mask ids must align")
        def pad(value, fill=0):
            if length == width:
                return value
            extra = value.new_full((value.shape[0], width - length, *value.shape[2:]), fill)
            return torch.cat((value, extra), dim=1)
        return (pad(hidden_states), pad(input_ids), pad(loss_mask),
                pad(verifier_last_hidden_states), pad(document_ids, -1), pad(position_ids))

    @torch.compiler.disable
    def prepare_training_inputs(
        self, hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
        document_ids, position_ids=None, max_anchors=None, **kwargs,
    ):
        """Prepare padding, anchor selection and BlockMask before model forward.

        Call under the same autocast context as forward. The returned dictionary
        is passed to the wrapped DDP/FSDP model, keeping preprocessing outside
        both the compiled forward and distributed forward hooks. In max-autotune
        mode, sparse indices reuse storage: finish the preceding backward before
        preparing the next batch; overlapping prepared batches are unsupported.
        """
        if max_anchors is None:
            import inspect
            original = getattr(type(self).forward, "_torchdynamo_orig_callable", type(self).forward)
            max_anchors = inspect.signature(original).parameters["max_anchors"].default
        anchor_seq_len = hidden_states.shape[1]
        (hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
         document_ids, position_ids) = self._pad_target_inputs(
            hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
            document_ids, position_ids,
        )
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        sampling = kwargs.pop("anchor_sampling", None)
        epoch = kwargs.pop("anchor_sampling_epoch", 0)
        epochs = kwargs.pop("anchor_sampling_num_epochs", 1)
        step = kwargs.pop("anchor_sampling_step", 0)
        rank = kwargs.pop("anchor_sampling_rank", 0)
        selected = kwargs.pop("_selected_anchors", None)
        self._anchor_sampling_metrics = {}
        if selected is None and sampling is not None:
            # Frozen verifier weights still belong to the root FSDP group.
            # Nonuniform scoring runs before root forward hooks, so gather them
            # explicitly. Occupancy decoder calls retain their own FSDP hooks.
            from torch.distributed.fsdp import FSDPModule
            gather_for_sampling = (
                sampling.get("strategy", "uniform") != "uniform"
                and isinstance(self, FSDPModule)
            )
            if gather_for_sampling:
                self.unshard()
            try:
                selected = self._sample_training_anchors(
                    hidden_states[:, :anchor_seq_len], input_ids[:, :anchor_seq_len],
                    loss_mask[:, :anchor_seq_len], verifier_last_hidden_states[:, :anchor_seq_len],
                    document_ids[:, :anchor_seq_len], position_ids[:, :anchor_seq_len],
                    max_anchors, sampling, epoch, epochs, step, rank,
                )
            finally:
                if gather_for_sampling:
                    self.reshard()
        prepared_mask = self._build_attention_mask(
            loss_mask, max_anchors, document_ids, hidden_states.device,
            selected_anchors=selected, anchor_seq_len=anchor_seq_len,
        )
        stable_masks = getattr(self, "_stable_block_masks", None)
        if stable_masks is not None:
            prepared_mask = stable_masks.update(prepared_mask)
        return dict(
            hidden_states=hidden_states, input_ids=input_ids, loss_mask=loss_mask,
            verifier_last_hidden_states=verifier_last_hidden_states,
            document_ids=document_ids, position_ids=position_ids,
            max_anchors=max_anchors, _prepared_attention_mask=prepared_mask, **kwargs,
        )

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    def load_verifier_weights(self):
        """Reconstruct weights intentionally omitted from DFlash checkpoints."""
        self._load_verifier_weights(
            overwrite_embed_tokens=True,
            overwrite_lm_head=not self.use_draft_vocab,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        config = DFlashSpeculatorConfig(
            **cls._build_base_config_kwargs("dflash", verifier_config, **kwargs)
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _build_base_config_kwargs(
        algorithm: str,
        verifier_config: "PretrainedConfig",
        **kwargs,
    ) -> dict:
        """Shared DFlash-family config kwargs for ``from_training_args``.

        DSpark reuses this and appends its Markov/confidence/loss fields.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"),
            kwargs["verifier_name_or_path"],
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        block_size = kwargs.get("block_size", 8)

        default_sample_from_anchor = algorithm == "dspark"
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        sample_from_anchor = (
            default_sample_from_anchor
            if sample_from_anchor_arg is None
            else sample_from_anchor_arg
        )
        default_non_causal = algorithm == "dflash2"
        non_causal_arg = kwargs.get("sliding_window_non_causal")
        sliding_window_non_causal = (
            default_non_causal if non_causal_arg is None else non_causal_arg
        )

        # Calculate speculative tokens based on sample_from_anchor
        # False: anchor is bonus token (block_size - 1 tokens)
        # True: sample from anchor too (block_size tokens)
        speculative_tokens = block_size if sample_from_anchor else block_size - 1

        return {
            "transformer_layer_config": verifier_config,
            "draft_vocab_size": kwargs["draft_vocab_size"],
            "block_size": block_size,
            "aux_hidden_state_layer_ids": target_layer_ids,
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": sliding_window_non_causal,
            "sample_from_anchor": sample_from_anchor,
            "speculators_config": SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        }

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], kwargs.get("loss_implementation", "fused")
        )
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 512)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        train = dict(shared)
        train["anchor_sampling"] = kwargs.get("anchor_sampling") or {"strategy": "uniform"}
        val = dict(shared)
        val["anchor_sampling"] = {"strategy": "uniform"}
        return train, val

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
            sliding_window_non_causal=sliding_window_non_causal,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device,
                              selected_anchors=None, anchor_seq_len=None):
        total_seq_len = loss_mask.shape[1]

        if selected_anchors is None:
            from speculators.models.dflash.anchor_sampling import select_static_anchors
            anchor_positions, anchor_valid = select_static_anchors(
                loss_mask[:, :anchor_seq_len], max_anchors, self.block_size,
                document_ids[:, :anchor_seq_len],
                sample_from_anchor=self.config.sample_from_anchor,
            )
        else:
            anchor_positions, anchor_valid = selected_anchors

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    @torch.compiler.disable
    def _sample_training_anchors(
        self, hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
        document_ids, position_ids, num_anchors, sampling, epoch, num_epochs,
        step, rank,
    ):
        """Select before building query blocks; keep scoring outside compiled graphs.

        A per-step generator is a deterministic function of checkpointed step,
        epoch and rank. Validation therefore cannot consume training RNG state.
        """
        from speculators.models.dflash.anchor_sampling import (
            select_static_anchors, valid_anchor_mask,
        )

        sampling = sampling or {"strategy": "uniform"}
        strategy = sampling.get("strategy", "uniform")
        if strategy != "uniform" and self.use_draft_vocab:
            raise ValueError("Nonuniform anchor sampling requires the full verifier vocabulary")
        if strategy == "round_occupancy" and type(self).__name__ != "DFlashDraftModel":
            raise ValueError("round_occupancy requires a defined DFlash proposal; DFlash2 is unsupported")
        device = hidden_states.device
        generator = torch.Generator(device=device)
        seed = (int(sampling.get("seed", 0)) + 1_000_003 * int(step)
                + 97_409 * int(epoch) + 65_537 * int(rank)) % (2**63 - 1)
        generator.manual_seed(seed)
        if device.type == "cuda" and strategy != "uniform":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        inclusion = None
        if strategy == "round_occupancy":
            from speculators.models.dflash.occupancy_sampling import (
                build_occupancy_scores, select_occupancy_anchors,
            )

            # Statistics include terminal-adjacent starts. Pad only the context
            # arrays so their synthetic query indices are safe; padded docs are
            # invisible to attention and never become training candidates.
            def pad(tensor, value=0):
                shape = list(tensor.shape)
                shape[1] = self.block_size
                return torch.cat((tensor, tensor.new_full(shape, value)), dim=1)

            scan_hidden, scan_ids = pad(hidden_states), pad(input_ids)
            scan_mask = pad(loss_mask)
            scan_last = pad(verifier_last_hidden_states)
            scan_docs = pad(document_ids, -1)
            scan_positions = position_ids
            if scan_positions is None:
                scan_positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            tail = scan_positions[:, -1:] + torch.arange(1, self.block_size + 1, device=device)
            scan_positions = torch.cat((scan_positions, tail), dim=1)

            @torch.no_grad()
            def scan_draft(positions):
                _, logits, _, _, _ = self._backbone_forward(
                    scan_hidden, scan_ids, scan_mask, scan_last, scan_docs,
                    scan_positions, max_anchors=positions.numel(),
                    _prepared_attention_mask=self._build_attention_mask(
                        scan_mask, positions.numel(), scan_docs, device,
                        selected_anchors=(positions, torch.ones_like(positions, dtype=torch.bool)),
                        anchor_seq_len=scan_hidden.shape[1],
                    ),
                    _anchor_scan=True,
                )
                return logits.reshape(-1, self.block_size, logits.shape[-1])

            scores = build_occupancy_scores(
                input_ids, loss_mask, document_ids, self.block_size,
                sample_from_anchor=self.config.sample_from_anchor,
                proposal=sampling["proposal"], target=sampling["target"],
                score_chunk_size=sampling.get("score_chunk_size", 1024),
                scan_draft=scan_draft,
                verifier_last_hidden_states=verifier_last_hidden_states,
                verifier_norm=self.verifier_norm, verifier_lm_head=self.verifier_lm_head,
            )
            candidates = valid_anchor_mask(
                loss_mask, document_ids, self.block_size,
                sample_from_anchor=self.config.sample_from_anchor,
            )
            anchors, valid, inclusion = select_occupancy_anchors(
                scores, candidates, num_anchors, generator=generator,
            )
        else:
            anchors, valid = select_static_anchors(
                loss_mask, num_anchors, self.block_size, document_ids,
                sampling=sampling, verifier_last_hidden_states=verifier_last_hidden_states,
                verifier_norm=self.verifier_norm, verifier_lm_head=self.verifier_lm_head,
                epoch=epoch, num_epochs=num_epochs,
                sample_from_anchor=self.config.sample_from_anchor, generator=generator,
            )
        if device.type == "cuda" and strategy != "uniform":
            torch.cuda.synchronize(device)
        self._anchor_sampling_metrics = {
            "anchor_selected_sum": valid.sum().detach().float(),
            "anchor_selected_total": torch.ones((), device=device),
            "anchor_score_seconds_sum": torch.tensor(time.perf_counter() - started, device=device),
            "anchor_score_seconds_total": torch.ones((), device=device),
        }
        if inclusion is not None:
            self._anchor_sampling_metrics.update({
                "anchor_inclusion_sum": inclusion[valid].sum().detach().float(),
                "anchor_inclusion_total": valid.sum().detach().float(),
                "anchor_provenance_verified_sum": torch.tensor(
                    float(sampling.get("provenance_verified", False)), device=device),
                "anchor_provenance_verified_total": torch.ones((), device=device),
            })
        return anchors, valid

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        **kwargs,
    ):
        """Run the anchored-block draft transformer up to the draft logits.

        Returns ``(hidden, logits, targets, aligned_loss_mask,
        anchored_block_indices)``. DSpark reuses this and adds its Markov and
        confidence heads before computing its own loss.
        """
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 512)
        anchor_scan = kwargs.pop("_anchor_scan", False)
        prepared_mask = kwargs.pop("_prepared_attention_mask", None)
        if prepared_mask is None:
            raise ValueError("Call prepare_training_inputs before the DFlash-family forward")
        (full_attn_mask, sliding_window_attn_mask,
         anchor_positions, anchor_valid) = prepared_mask

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the hidden_states shape doesn't match position_ids but doesn't need
        # to, as hidden_states is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            if anchor_scan:
                targets = verifier_last_hidden_states.new_empty((1, 0, 0))
            elif anchored_block_indices.numel() < total_seq_len:
                target_indices = (
                    anchored_block_indices
                    if self.config.sample_from_anchor
                    else (anchored_block_indices - 1) % total_seq_len
                )
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states[:, target_indices])
                )
            else:
                verifier_logits = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                if not self.config.sample_from_anchor:
                    verifier_logits = torch.roll(verifier_logits, 1, dims=1)
                targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)
        # shape: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        # For sample_from_anchor=False, mask slot 0 (anchor) since it's not trained
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        _, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        metrics.update(getattr(self, "_anchor_sampling_metrics", {}))
        return None, loss, metrics
