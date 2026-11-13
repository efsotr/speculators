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
| `kl_div2`  | Soft-target cross entropy, `-sum_i p_i log q_i` | Forward KL without the teacher entropy term; same draft gradient, different values and DPA-dev weights. |
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
`argmax` of the teacher logits. It supports both `fused` and `eager` implementations.
The eager implementation follows `ce` input/autocast dtype without an explicit
fp32 logits conversion. Both fused CE variants retain fp32 internal statistics.
It supports weighted combinations such as `'{"ce_gt": 0.1, "tv": 1.8}'`.
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

## DPA-dev shared position weights

DFlash, DFlash2 and DSpark support `dflash.per_position_loss_weight: dpa_dev`
(`dpa-dev` is also accepted). All loss terms use the same detached per-token
weights, including DSpark confidence BCE, DFlash2 selector CE and optional hidden
MSE. Existing loss coefficients and mask-count normalization remain in effect.
The weights reuse D-PACE smoothing, block prefix products and suffix sums.

```yaml
loss:
  loss_fn: '{"kl_div2": 1.0, "tv": 1.8}'
dflash:
  per_position_loss_weight: dpa_dev
  dpa_dev_signal: kl_div2
  dpa_dev_alpha: 0.5
```

The independent parameters are `dpa_dev_signal` (default `accept`) and
`dpa_dev_alpha` (default `0.5`, in `(0, 1]`); their CLI forms are
`--dpa-dev-signal` and `--dpa-dev-alpha`. DPA-dev ignores `dpace_*` settings.
The signals `ce`, `ce_gt`, `kl_div2` require the corresponding term in `loss_fn`
and use `exp(-loss.detach())` before smoothing. Loss coefficients do not enter
the signal. `kl_div` is not a supported signal: it includes `sum p log p`,
whereas `kl_div2 = -sum p log q` excludes that teacher entropy contribution.
Both loss implementations support `kl_div2`; fused forward streams vocabulary
tiles without materializing dense probabilities and reuses KL backward.

`accept` uses detached `1 - tv`, reusing the per-token TV loss when configured;
otherwise TV is computed without gradients. Each configured loss is evaluated
once and reused for reduction and weighting. DFlash2 derives weights from unary
losses and shares them with its auxiliary objectives. Weights are computed only
after detaching the signal; gradients flow through the original losses.
