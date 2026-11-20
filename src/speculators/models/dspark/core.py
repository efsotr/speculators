from collections.abc import Callable
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import (
    LossConfig,
    kl_div_loss,
    resolve_loss_config,
    tv_loss,
)
from speculators.losses.real import TRAJECTORY_LOSSES
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.dvolt import build_candidate_graph, candidate_probs
from speculators.models.dspark.metrics import compute_metrics, sample_vat_uniform
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        implementation = kwargs.get("loss_implementation", "fused")
        dvolt_top_k = kwargs.get("dvolt_top_k", 0)
        loss_config = resolve_loss_config(
            "tv" if dvolt_top_k else kwargs["loss_fn"], implementation
        )
        tv_loss_fn = resolve_loss_config("tv", implementation)["tv"][0]
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "loss_normalization": kwargs.get("loss_normalization", "weight_sum"),
            "real_beta": kwargs.get("real_beta", 0.0),
            "ste_r": kwargs.get("ste_r", False),
            "tv_loss_fn": tv_loss_fn,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "per_position_loss_weight": per_position_loss_weight,
            "dpa_dev_signal": kwargs.get("dpa_dev_signal", "accept"),
            "dpa_dev_alpha": kwargs.get("dpa_dev_alpha", 0.5),
            "dpa_dev_prefix_only": kwargs.get("dpa_dev_prefix_only", False),
            "dpace_alpha": dpace_alpha,
        }
        if per_position_loss_weight == "vat" and not TRAJECTORY_LOSSES.intersection(
            loss_config
        ):
            shared["vat_seed"] = kwargs.get("vat_seed", 42)
        if dvolt_top_k:
            shared.update(
                dvolt_top_k=dvolt_top_k, dvolt_beta=kwargs.get("dvolt_beta", 0.3)
            )
        val = dict(shared)
        val["anchor_sampling"] = DFlashDraftModel.validation_anchor_sampling(**kwargs)
        return shared, val

    @torch.compiler.disable
    def prepare_training_inputs(
        self, *args, vat_seed=42, vat_batch=(0, 0, 0), **kwargs
    ):
        prepared = super().prepare_training_inputs(*args, **kwargs)
        if prepared.get("per_position_loss_weight") == "vat" and not (
            TRAJECTORY_LOSSES.intersection(prepared.get("loss_config") or {})
        ):
            epoch, step, rank = vat_batch
            position_seed = vat_seed + 1_000_003 * step + 97_409 * epoch + 65_537 * rank
            phase_offset = 0 if self.training else 2_147_483_647
            seed = (position_seed + phase_offset) % (2**63 - 1)
            prepared["vat_uniform"] = sample_vat_uniform(
                (1, prepared["max_anchors"] * self.block_size),
                prepared["hidden_states"].device,
                seed,
            )
        return prepared

    @conditional_torch_compile
    def forward(  # noqa: C901, PLR0917
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 4.0,
        real_beta: float = 0.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        loss_normalization: str = "weight_sum",
        per_position_loss_weight: str = "fixed-exp-decay",
        dpa_dev_signal: str = "accept",
        dpa_dev_alpha: float = 0.5,
        dpa_dev_prefix_only: bool = False,
        dpace_alpha: float = 0.5,
        vat_uniform: torch.Tensor | None = None,
        ste_r: bool = False,
        dvolt_top_k: int = 0,
        dvolt_beta: float = 0.3,
        dvolt_score_branches: Callable | None = None,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                _anchor_scan=bool(dvolt_top_k),
                **kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)

        if dvolt_top_k:
            if (
                self.use_draft_vocab
                or not self.config.sample_from_anchor
                or self.markov_head is None
                or self.markov_head.head_type != "vanilla"
                or dvolt_top_k > logits.shape[-1]
                or dvolt_score_branches is None
            ):
                raise ValueError(
                    "D-VOLT requires a full vocabulary, a vanilla Markov head, "
                    "sample_from_anchor and online branch scoring"
                )
            anchors = anchored_block_indices.reshape(num_blocks, block)[:, 0]
            valid = aligned_loss_mask.reshape(num_blocks, block).bool().any(dim=-1)

            def score_branches(active, candidates, choices):
                return dvolt_score_branches(
                    input_ids, document_ids, position_ids,
                    anchors=anchors[active], candidates=candidates, choices=choices,
                )

            def project_target(features):
                return self.verifier_lm_head(self.verifier_norm(features))

            prev_token_ids, targets, candidates, values = build_candidate_graph(
                logits.reshape(num_blocks, block, -1),
                block_tokens[:, 0], self.markov_head,
                top_k=dvolt_top_k, beta=dvolt_beta, score_branches=score_branches,
                project_target=project_target,
                anchor_hidden=verifier_last_hidden_states[0, anchors], valid=valid,
            )
            loss_config = {"tv": (tv_loss_fn, 2.0)}
            if dvolt_beta:
                ids = candidates.reshape(1, mask_tokens_size, dvolt_top_k)

                def continuation_reward(draft, target):
                    overlap = torch.minimum(
                        candidate_probs(draft, ids), candidate_probs(target, ids)
                    )
                    return (overlap * values.reshape_as(ids)).sum(-1)

                loss_config["dvolt_continuation"] = (
                    continuation_reward, -2 * dvolt_beta
                )

        confidence_logits = None
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)],
                    dim=-1,
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        trajectory = bool(TRAJECTORY_LOSSES.intersection(loss_config or {}))
        uses_dataset_loss = (
            trajectory
            or "ce_gt" in (loss_config or {})
            or per_position_loss_weight == "vat"
        )
        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            real_beta=real_beta,
            ste_r=ste_r,
            confidence_head_alpha=confidence_head_alpha,
            loss_normalization=loss_normalization,
            per_position_loss_weight=per_position_loss_weight,
            dpa_dev_signal=dpa_dev_signal,
            dpa_dev_alpha=dpa_dev_alpha,
            dpa_dev_prefix_only=dpa_dev_prefix_only,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            data_token_ids=(
                self._aligned_data_token_ids(
                    input_ids,
                    anchored_block_indices,
                    require_full_vocab=uses_dataset_loss,
                )
                if not dvolt_top_k and (
                    uses_dataset_loss
                    or not self.use_draft_vocab
                    or (
                        per_position_loss_weight
                        in {"dpa_dev", "dpa-dev", "dpa_dev2", "dpa-dev2"}
                        and dpa_dev_signal in {"p_draft", "q_draft", "q", "p"}
                    )
                )
                else None
            ),
            compute_prob_metrics=not self.use_draft_vocab,
            vat_uniform=vat_uniform,
        )
        return None, loss, metrics
