"""Triton RMSNorm + bounded tiled vocabulary statistics for anchor scores."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rmsnorm(X, Positions, Weight, Y, H: tl.constexpr, X_STRIDE: tl.constexpr,
                    X_H_STRIDE: tl.constexpr, P_STRIDE: tl.constexpr,
                    EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    pos = tl.load(Positions + row * P_STRIDE)
    x = tl.load(X + pos * X_STRIDE + col * X_H_STRIDE, col < H, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    # Qwen3RMSNorm rounds normalized values before multiplying the weight.
    normalized = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(Weight + col, col < H, 0)
    tl.store(Y + row * H + col, normalized * weight, col < H)


@triton.jit
def _vocab_stats(Logits, Stats, V: tl.constexpr, TILES: tl.constexpr,
                 FLATNESS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Logits + row * V + col, col < V, -float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    expx = tl.where(maximum == -float("inf"), 0., tl.exp(x - maximum))
    total = tl.sum(expx, 0)
    if FLATNESS:
        moment = tl.sum(expx * expx, 0)
    else:
        moment = tl.sum(expx * tl.where(x == -float("inf"), 0., x), 0)
    base = (row * TILES + tile) * 3
    tl.store(Stats + base, maximum)
    tl.store(Stats + base + 1, total)
    tl.store(Stats + base + 2, moment)


@triton.jit
def _finish_stats(Stats, Scores, V: tl.constexpr, TILES: tl.constexpr,
                  FLATNESS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    base = (row * TILES + tile) * 3
    maxima = tl.load(Stats + base, tile < TILES, -float("inf"))
    sums = tl.load(Stats + base + 1, tile < TILES, 0.)
    moments = tl.load(Stats + base + 2, tile < TILES, 0.)
    maximum = tl.max(maxima, 0)
    scale = tl.exp(maxima - maximum)
    total = tl.sum(sums * scale, 0)
    if FLATNESS:
        moment = tl.sum(moments * scale * scale, 0)
        score = total * tl.rsqrt(V * moment)
    else:
        moment = tl.sum(moments * scale, 0)
        score = maximum + tl.log(total) - moment / total
    tl.store(Scores + row, score)


def triton_score_chunk(last_hidden, positions, norm, head, flatness):
    if not last_hidden.is_cuda:
        raise ValueError("Triton anchor scoring requires CUDA tensors")
    if not (hasattr(norm, "variance_epsilon") and hasattr(norm, "weight")):
        raise ValueError("Triton anchor scoring requires Qwen-style RMSNorm")
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("Triton anchor scoring requires a Linear LM head")
    supported = {torch.float16, torch.bfloat16, torch.float32}
    if norm.weight.dtype not in supported or last_hidden.dtype not in supported:
        raise ValueError("Triton RMSNorm supports fp16, bf16 and fp32 hidden/weight dtypes")
    hidden_size = last_hidden.shape[-1]
    normalized = torch.empty((positions.numel(), hidden_size), device=last_hidden.device,
                             dtype=torch.promote_types(last_hidden.dtype, norm.weight.dtype))
    _gather_rmsnorm[(positions.numel(),)](
        last_hidden, positions, norm.weight, normalized, hidden_size,
        last_hidden.stride(1), last_hidden.stride(2), positions.stride(0),
        norm.variance_epsilon, triton.next_power_of_2(hidden_size),
    )
    # Preserve native autocast: fp32 master head/norm weights may project to bf16.
    # The norm output promotes its rounded input to the norm weight dtype first.
    logits = torch.nn.functional.linear(normalized, head.weight, head.bias)
    vocab = logits.shape[-1]
    tiles = triton.cdiv(vocab, 4096)
    stats = torch.empty((positions.numel(), tiles, 3), device=logits.device, dtype=torch.float32)
    scores = torch.empty((positions.numel(),), device=logits.device, dtype=torch.float32)
    _vocab_stats[(positions.numel(), tiles)](logits, stats, vocab, tiles, flatness, 4096)
    _finish_stats[(positions.numel(),)](
        stats, scores, vocab, tiles, flatness, triton.next_power_of_2(tiles),
    )
    return scores
