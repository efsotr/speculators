# DFlash2

DFlash2 extends [DFlash](dflash.md) with local dynamic convolutions and a candidate selector. The convolution lets neighboring positions exchange local information while the draft block remains parallel. The selector then adds a low-rank, predecessor-conditioned transition score to the draft model's unary token scores.

## How It Works

### Local Convolution

Each draft layer wraps both attention and the MLP with a grouped causal convolution. A projection of the current hidden state produces two dynamic kernels: one for the sublayer input and one for its output. A learned base kernel is shared across positions, while channels in each group share the dynamic component.

Convolution never crosses an anchor-block boundary. This matters during training because Speculators flattens multiple anchored blocks into one token dimension.

### Candidate Selector

The unary draft logits first select `selector_top_k` candidates at each position. The selector scores each transition as

```text
unary(candidate)
  + dot(predecessor_codebook(previous) * hidden_projection(hidden),
        successor_codebook(candidate))
```

Inference starts with the anchor token and then conditions each position on the token selected at the preceding position. This preserves one parallel draft-model forward pass: only the small selector walks the candidate path.

## Training Objective

The public DFlash2 implementation specifies inference but does not publish its training objective. The current Speculators implementation is therefore an experimental split objective:

1. The configured DFlash-family compound loss trains the unary logits.
2. A separate hard cross-entropy term trains the selector over the unary top-K candidates while teacher-forcing the previous sequence token. If the target is outside that set, training replaces the weakest unary candidate with the target so the K-way loss remains defined. Validation loss uses the same replacement; serving and strict validation candidate/path metrics do not.

`--selector-loss-alpha` controls the second term. Both terms use the configured fixed exponential or D-PACE position weighting. The selector is never trained against a full-vocabulary corrected distribution, matching the candidate set it can rerank at serving time.

Validation reports clearly separated unary candidate recall and target mass, teacher-forced selector accuracy, and an actual greedy self-conditioned path. The path begins at the verified anchor and feeds each selected token to the next edge score. Its per-position accuracy is conditioned on the earlier path being correct. The accepted-length metrics include the verified anchor and report both the realized selector path and the oracle unary-top-K path.

DFlash2 currently requires the full verifier vocabulary. Pruned draft vocabularies are rejected because current serving implementations select candidates before any draft-to-target vocabulary mapping.

This experimental implementation directly trains full-vocabulary predecessor and successor codebooks. That is an expanded free-codebook parameterization: the public checkpoint format contains materialized codebooks, but the public sources do not disclose whether those tensors were free parameters during training. In particular, this prototype should not be interpreted as reproducing the selector parameter count reported in the DFlash2 blog.

## Key Parameters

| Parameter               | Default | Description                              |
| ----------------------- | ------: | ---------------------------------------- |
| `--conv-kernel-size`    |       2 | Number of causal convolution taps        |
| `--conv-group-size`     |      16 | Hidden channels sharing a dynamic kernel |
| `--selector-rank`       |     256 | Rank of the transition factorization     |
| `--selector-top-k`      |      16 | Unary candidates reranked per position   |
| `--selector-loss-alpha` |     1.0 | Weight of the selector K-way CE term     |

All [DFlash](dflash.md) backbone parameters also apply. DFlash2 defaults to five draft layers, block size 8, `sample_from_anchor: False`, fixed exponential position weighting, and KL divergence loss. Set all shared knobs explicitly when comparing it with another algorithm.

To train your own, see `examples/train/dflash2_qwen3_8b_ultrachat_online_5k.sh`.

## Experimental D-PACE development weighting

`dpace_dev` supports both `loss.loss_fn: kl_div` and `ce` for DFlash/DFlash2.
The original `dpace` mode still requires CE. Example additions to a DFlash2 YAML:

```yaml
loss:
  loss_fn: kl_div
dflash:
  per_position_loss_weight: dpace_dev
  dpace_alpha: 0.5
  dpace_dev_accept_prob: min_q_p
  dpace_token_source: teacher_argmax  # or data
```

`--dpace-dev-accept-prob` (YAML `dflash.dpace_dev_accept_prob`) selects:

| Value | Probability before D-PACE smoothing |
| --- | --- |
| `q` (default) | q |
| `p` | p |
| `min_q_p` | min(q, p) |
| `p_over_q` | min(1, p/q) |
| `q_over_p` | min(1, q/p) |

`--dpace-token-source` (YAML `dflash.dpace_token_source`) selects the token
used for both q and p: `teacher_argmax` (framework default) uses the teacher's
full-vocabulary argmax; `data` uses the dataset token aligned to that prediction.
With `sample_from_anchor: false` this is the token at the block position; with
`true` it is the next dataset token. This only changes D-PACE weights: unary
KL/CE targets and selector CE labels remain teacher-derived.

For the unary loss, q and p are the selected token's draft and teacher
full-vocabulary probabilities. For the selector, q is recomputed from corrected
logits over the **original unary top-K**, before any teacher-token injection.
If the selected token is absent, q is zero (including for `teacher_argmax`).
p always remains its full-vocabulary teacher probability. Selector training CE
still uses the teacher-injected candidate set; the two losses compute separate,
detached weights. A zero q does not necessarily give zero weight because the
chosen acceptance formula and D-PACE smoothing still apply.

The selected probability replaces `exp(-CE)` in the existing D-PACE algorithm.
Both losses retain the existing smoothing `(1-alpha)*accept + alpha`, mask,
prefix products, suffix sums, and masked-mean denominator. Probabilities and
weights are detached: gradients flow only through the weighted KL or CE.
`dpace_alpha` must be in `(0, 1]`. Ratio denominators clamp to fp32's smallest
normal positive value; 0/0 evaluates to 0. Probability reductions use fp32.
No inference changes are required. Project defaults are unchanged.

## Serving

Checkpoints emitted here follow the public Z Lab weight contract but use a speculators config. They can be served in vLLM using `vllm serve ./checkpoint`.

## Research and Implementation Reference

The architecture follows Z Lab's [DFlash2 implementation](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py) and the [DFlash2 technical blog](https://inco.ai/blog/dflash2/). The pinned source revision defines the inference architecture and checkpoint contract; the experimental training objective above is specific to this implementation.

## See Also

- [DFlash](dflash.md) -- The block-parallel draft backbone
- [DSpark](dspark.md) -- Adds a Markov head and confidence scheduling
- [Train a Speculator](../tutorials/train.md) -- End-to-end training workflow
