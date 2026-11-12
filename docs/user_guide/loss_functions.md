# Loss Functions

The training loss controls what the draft model is optimized for. Select it with `--loss-fn` in [`speculators train`](../cli/train.md), which accepts either a single loss name or a JSON dict for a weighted combination. It applies to Eagle-3, P-EAGLE, DFlash, and DSpark. MTP uses its own multi-step cross-entropy loss and ignores this flag.

The fused Triton implementation is the default. Use `--loss-implementation eager` for compatibility or numerical validation; eager losses may OOM with DFlash or DSpark.

```bash
speculators train ... --loss-fn kl_div
```

## Available Losses

`p` is the target distribution, `q` the draft distribution, and `alpha = sum_v min(p_v, q_v)` the distributional overlap, which equals the acceptance rate of speculative decoding.

| Name        | Objective                              | Notes                                                                          |
| ----------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| `kl_div`    | Forward KL, target to draft            | Default. Mass-covering: penalizes the draft for missing target mass.           |
| `rkl`       | Reverse KL, draft to target            | Mode-seeking: the draft concentrates on the target's dominant modes.           |
| `jsd`       | Jensen-Shannon divergence              | Symmetric, bounded by `log 2`. Balances forward and reverse KL.                |
| `ce`        | Cross-entropy against `argmax p`       | Hard labels from the target. Required by `--per-position-loss-weight dpace`.   |
| `ce_gt`     | Cross-entropy against dataset token IDs | DFlash, DFlash2 unary, DSpark; requires the full verifier vocabulary. |
| `tv`        | Total variation, `1 - alpha`           | Optimizes the acceptance rate directly. Gradients vanish when overlap is low.  |
| `nla`       | Negative log-acceptance, `-log(alpha)` | TV's target with a `1 / alpha` gradient boost, so it trains from a cold start. |
| `lk_hybrid` | Adaptive KL/TV blend                   | `lambda * KL + (1 - lambda) * TV` with `lambda = exp(-3 * alpha)`, detached.   |

All losses use the selected implementation; there is no automatic fallback.

## Choosing a Loss

- Start with `kl_div`. It is the default and the best-tested option.
- To optimize acceptance rate directly, use `lk_hybrid` or `nla`. Plain `tv` has the same objective but weak gradients early in training.
- Use `ce` when you want hard-label supervision or D-PACE position weighting.

## Weighted Combinations

Pass a JSON dict to train on a weighted sum of several losses. Weights are used as provided and are not normalized. Each term is also logged separately as `{name}_loss`, before its weight is applied.

```bash
speculators train ... --loss-fn '{"ce": 0.1, "tv": 0.9}'
```

## References

- Lin, "Divergence measures based on the Shannon entropy" (1991) -- Jensen-Shannon divergence
- Samarin et al., ["LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"](https://arxiv.org/abs/2602.23881) -- `nla` and `lk_hybrid`

## Dataset ground-truth CE

`--loss-fn ce_gt` uses integer token IDs from the training data instead of
`argmax` of the teacher logits. It supports both `fused` and `eager` implementations,
including weighted combinations such as `'{"ce_gt": 0.1, "tv": 1.8}'`.
For anchor `a`, slot `k` uses dataset token `a+k+1` when
`sample_from_anchor=true`, otherwise `a+k`. Alignment reuses the backbone's sampled
block indices; invalid/document-boundary/padded positions remain excluded by the
existing loss mask. The fused kernel computes `logsumexp(logits)-logits[target_id]`
in fp32 and reuses CE backward (`softmax-onehot`), without a dense one-hot tensor.

Only the `ce_gt` term consumes dataset IDs. Teacher-distribution losses, confidence
overlap targets, accuracy metrics and the DFlash2 selector objective retain their
existing targets. Position decay and mask-count normalization are unchanged.
Original D-PACE also accepts standalone `ce_gt` (confidence is the probability of
the dataset token); `dpace_dev` retains its independent token-source setting.
Models with a reduced draft vocabulary reject `ce_gt` rather than silently
interpreting verifier IDs as draft IDs.

DeepSpec DSpark's `0.1 CE + 0.9 L1` corresponds to `0.1 ce_gt + 1.8 tv` in
per-token objectives. This does not make the complete training loss identical:
DeepSpec normalizes by globally summed decayed mask weights, whereas the local
implementation normalizes by each rank's undecayed mask count.
