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

`ce_tmx` is a configuration alias for `ce` (teacher argmax), accepted as a
single name, in weighted JSON combinations, and for `dpa_dev_signal`. Config
resolution normalizes it to `ce`; logs and runtime losses use `ce`. If both
names occur in a combination, their weights are added.

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

`dpa_dev_signal: p_draft` uses `min(1, q(x)/p(x))`; `q_draft` uses
`min(1, p(x)/q(x))`, where p is the teacher distribution, q the draft
distribution, and x is always the aligned dataset token (required).
The `q` signal uses the draft probability `q(x)` directly. Ratios use detached
fp32 probabilities and clamp the denominator to fp32 tiny.
`dpa_dev_prefix_only: true` (`--dpa-dev-prefix-only`) is supported only
with these two signals. Its weight at position i is the product over j < i
of `alpha + (1-alpha)*prob_j`, with the empty product equal to 1.
Masked positions receive zero weight and contribute a factor of 1; products
restart at each block. The default is false, retaining prefix/suffix weights.

## Loss normalization

`loss.loss_normalization` defaults to `weight_sum` for all position-weighting
modes (fixed exponential decay, dpace, dpace_dev and dpa_dev):

```yaml
loss:
  loss_normalization: weight_sum
```

For each forward step, the denominator is the sum of detached position weights
multiplied by the valid-token mask, reduced across the whole local batch and then
mean-all-reduced across ranks. Each rank divides its weighted loss numerator by
this common denominator plus `1e-5 / world_size`. DDP's gradient averaging therefore gives the
global weighted mean. A rank with no valid tokens still participates in the
collective. This also applies to DSpark confidence, DFlash2 selector and auxiliary
hidden MSE; each objective uses its own effective weights and valid mask.
Hidden MSE clamps its denominator to `1e-5 / world_size` instead of adding
the epsilon. Scaling the epsilon with world size preserves equivalence to the
same complete batch on one device.
Compound loss computes its shared position weights and normalization denominator
once, including a single denominator all-reduce, then reuses them for every
configured term. The coefficient-weighted elementwise losses are added before
the shared weighting and final normalization; logged terms use the same denominator. Original
`dpace` derives weights from the individual loss and is not supported with
compound loss; use `dpa_dev` or `dpa-dev2` for compound dynamic weights.

`token_count` retains the previous local token-count reduction, including the
legacy confidence/MSE denominators. The setting is independent of
`dflash.per_position_loss_weight`. A step here is one model forward; the trainer
currently performs one forward per optimizer step.

### DPA-dev2

Set `dflash.per_position_loss_weight: dpa-dev2` (alias `dpa_dev2`).
DFlash, DFlash2 and DSpark share detached fp32 weights across all configured
losses, confidence, selector and hidden MSE, using the existing normalization.

Reuse `dpa_dev_signal`: `q`, `p`, `q_draft=min(1,p/q)`,
`p_draft=min(1,q/p)`, or `acc=1-TV` (alias `accept`, framework default).
Here q is drafter and p is target; token signals use aligned dataset tokens,
with the same anchor alignment as ce_gt. They do not require ce_gt in loss_fn.

Smooth probabilities with `r_i=alpha+(1-alpha)*prob_i`, using
`dpa_dev_alpha` (framework default 0.5, range [0,1]); zero disables smoothing.
Within each block, `P_0=1`, `P_i=P_{i-1}*r_{i-1}`,
`S_last=1`, `S_i=1+r_{i+1}*S_{i+1}`, and `W_i=P_i*S_i`.
Thus neither factor includes the current probability.
`dpa_dev_prefix_only: true` uses W=P for every supported signal.
Masked positions have zero weight and are skipped by both recurrences.
S uses PyTorch associative_scan over affine compositions; PyTorch 2.13 exposes
this through `torch._higher_order_ops.associative_scan`, with CUDA/Inductor
support. No division by prefix products is used.

DPA-dev2 ratio signals use `exp(min(log_q - log_p, 0))` for p_draft
and `exp(min(log_p - log_q, 0))` for q_draft, computing log probabilities
in fp32 before exponentiation to avoid probability-ratio underflow.

### REAL with dataset-token CE

For DFlash, DFlash2 and DSpark with the verifier vocabulary, set:

```yaml
loss:
  loss_normalization: token_count
  loss_fn: '{"REAL": 0.9, "ce_gt": 0.1}'
```

The two coefficients must be in [0, 1] and sum to 1.
`ce_gt` is the CE mixing coefficient lambda, and `REAL` is its complement.
The shorthand `REAL` uses the weights in the example.
The trainer's confidence and selector losses retain their own coefficients
and use fixed exponential position decay with REAL.
DFlash2's DPA-dev and DPA-dev2 selector modes require a tokenwise unary loss.

At each prediction slot, REAL multiplies the draft/target distribution overlap
by the product of earlier dataset-token acceptance ratios
`min(1, q(token) / p(token))`, where q is the draft and p is the teacher.
Set `loss.real_beta` to replace the current slot's overlap `a_i` with
`(1 + beta) * a_i - beta * r_i`, where `r_i` is that slot's dataset-token
acceptance ratio. The exclusive prefix still uses the earlier acceptance
ratios. The default is `0.0`. This coefficient applies to the objective;
`prob_eal` continues to use the overlap.
Set `loss.ste_r: true` to pass gradients through the log acceptance clamp
at valid REAL slots, including where the draft token probability exceeds
the teacher token probability. The forward acceptance ratio is unchanged.
This applies to the exclusive prefix and the acceptance term in the beta
correction. The default is `false`.
The first masked slot or document boundary ends a trajectory. The REAL
component averages one minus the contribution at each valid prediction
slot. The CE component weights those same slots by `exp(-i / gamma)`,
where `i=0` is the first prediction and `gamma` is `dflash_decay_gamma`
(default 4.0). Both use the local
valid token count, without a denominator collective, and are zero for an
empty batch. Training configs with REAL default to `token_count` and reject
`weight_sum`.
