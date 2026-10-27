"""CUDA numerical checks for compiled and handwritten Triton anchor scoring."""

import pyarrow  # noqa: F401
import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.models.dflash.anchor_sampling import score_target_positions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def isolate_compile_cache():
    # Each parameterized case creates a different model/layout. Isolate those
    # models instead of exhausting Dynamo's per-frame recompile budget.
    torch.compiler.reset()
    yield
    torch.compiler.reset()



@pytest.mark.parametrize("backend", ["compile", "triton", "auto"])
@pytest.mark.parametrize("strategy", ["target_flatness", "target_entropy_high"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_accelerated_scores_include_norm_and_ragged_strided_positions(backend, strategy, dtype):
    torch.manual_seed(63)
    # Both hidden and positions have nontrivial strides. Vocab size straddles
    # a Triton reduction tile; a final chunk of one verifies tail handling.
    hidden = torch.randn(1, 18, 256, device="cuda", dtype=dtype)[:, :, ::2]
    hidden.requires_grad_(True)
    positions = torch.arange(0, 18, device="cuda")[::2]
    norm = Qwen3RMSNorm(128).to(device="cuda", dtype=dtype)
    norm.weight.data.copy_(torch.linspace(.25, 2., 128, device="cuda", dtype=dtype))
    head = torch.nn.Linear(128, 4103, device="cuda", dtype=dtype)
    reference = score_target_positions(hidden, positions, norm, head, strategy, 4, "eager")
    result = score_target_positions(hidden, positions, norm, head, strategy, 4, backend)
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=5e-6)
    assert result.dtype == torch.float32
    assert not result.requires_grad
    assert hidden.grad is None and norm.weight.grad is None and head.weight.grad is None


@pytest.mark.parametrize("backend", ["compile", "triton"])
@pytest.mark.parametrize("strategy", ["target_flatness", "target_entropy_high"])
def test_masked_logits_and_nonfinite_validation(backend, strategy):
    hidden = torch.zeros(1, 3, 16, device="cuda")
    norm = Qwen3RMSNorm(16).cuda()
    head = torch.nn.Linear(16, 4103, device="cuda")
    head.bias.data.zero_()
    head.bias.data[:4096] = -float("inf")
    args = (hidden, torch.arange(3, device="cuda"), norm, head, strategy, 2)
    expected = score_target_positions(*args, backend="eager")
    torch.testing.assert_close(score_target_positions(*args, backend=backend), expected)
    head.bias.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        score_target_positions(*args, backend=backend)


@pytest.mark.parametrize("backend", ["compile", "triton", "auto"])
def test_fp32_master_norm_and_head_under_bf16_training_autocast(backend):
    torch.manual_seed(64)
    hidden = torch.randn(1, 9, 128, device="cuda", dtype=torch.bfloat16)
    norm = Qwen3RMSNorm(128).cuda()  # Real native draft retains FP32 masters.
    norm.weight.data.copy_(torch.linspace(.25, 2., 128, device="cuda"))
    head = torch.nn.Linear(128, 4103, device="cuda")
    positions = torch.tensor([0, 2, 5, 8, 1], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert norm(hidden).dtype == torch.float32
        assert head(norm(hidden)).dtype == torch.bfloat16
        for strategy in ("target_flatness", "target_entropy_high"):
            expected = score_target_positions(hidden, positions, norm, head, strategy, 4, "eager")
            actual = score_target_positions(hidden, positions, norm, head, strategy, 4, backend)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)
    assert norm.weight.dtype == head.weight.dtype == torch.float32
    assert not actual.requires_grad
