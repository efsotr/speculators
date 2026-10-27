"""Bounded-memory target scoring kernels, including the verifier RMS norm.

Both accelerated paths leave the LM-head GEMM to PyTorch/cuBLAS. The compile
path captures gather, norm, projection and reductions as one full graph; the
Triton path gathers/normalizes directly and reduces vocab tiles without storing
float32 softmax or log-probability tensors. No target gradients are recorded.
"""

import functools
import math

import torch


def eager_score_chunk(last_hidden, positions, norm, head, flatness):
    logits = head(norm(last_hidden[:, positions])).float().squeeze(0)
    logp = torch.log_softmax(logits, dim=-1)
    if flatness:
        return torch.exp(-0.5 * (math.log(logits.shape[-1]) + torch.logsumexp(2 * logp, dim=-1)))
    return -(logp.exp() * torch.where(torch.isfinite(logp), logp, 0.0)).sum(-1)


@functools.lru_cache(maxsize=1)
def _compiled_scorer():
    # Compile the norm too; compiling only a post-logits reduction leaves the
    # gather/norm and float32 vocabulary intermediates on the eager path.
    # Qwen RMSNorm deliberately casts to bf16/fp16 before its weight multiply.
    # Without this option Inductor can erase that rounding and perturb rankings.
    return torch.compile(eager_score_chunk, fullgraph=True, dynamic=True,
                         options={"emulate_precision_casts": True})


def score_chunk(last_hidden, positions, norm, head, flatness, backend):
    if backend == "auto":
        if not last_hidden.is_cuda:
            backend = "eager"
        elif (hasattr(norm, "variance_epsilon") and hasattr(norm, "weight")
              and isinstance(head, torch.nn.Linear)
              and last_hidden.dtype in {torch.float16, torch.bfloat16, torch.float32}
              and norm.weight.dtype in {torch.float16, torch.bfloat16, torch.float32}):
            backend = "triton"
        else:
            backend = "compile"
    if backend == "eager":
        return eager_score_chunk(last_hidden, positions, norm, head, flatness)
    if backend == "compile":
        return _compiled_scorer()(last_hidden, positions, norm, head, flatness)
    if backend == "triton":
        from .anchor_scoring_triton import triton_score_chunk

        return triton_score_chunk(last_hidden, positions, norm, head, flatness)
    raise ValueError(f"Unknown anchor score backend {backend!r}")
