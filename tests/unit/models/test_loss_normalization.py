"""Step weight normalization, including unequal distributed rank weights."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from speculators.losses import resolve_loss_config
from speculators.losses.utils import loss_function
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.core import DFlash2DraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.train.config import TrainConfig


def _loss(values, mask, weights, mode="weight_sum"):
    return loss_function(
        values,
        values,
        mask,
        torch.zeros_like(mask),
        elementwise_loss=values,
        decay_fn=lambda *a, **k: weights,
        loss_normalization=mode,
    )


@pytest.mark.parametrize("mode", ["weight_sum", "token_count"])
def test_local_reduction_and_detached_denominator(mode):
    values = torch.tensor([[2.0, 4.0, 99.0], [3.0, 5.0, 7.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    weights = torch.tensor([[0.2, 0.5, 0.9], [0.8, 0.3, 0.1]])
    actual = _loss(values, mask, weights, mode)
    weighted = values * mask * weights
    expected = (
        weighted.sum() / ((mask * weights).sum() + 1e-5)
        if mode == "weight_sum"
        else (weighted.sum(1) / (mask.sum(1) + 1e-5)).mean()
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, values)[0], torch.autograd.grad(expected, values)[0]
    )
    assert _loss(values, mask * 0, weights).item() == 0


def _distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        parameter = torch.tensor(2.0, requires_grad=True)
        values = parameter * torch.tensor([[1.0, 3.0]])
        mask = torch.ones_like(values) if rank == 0 else torch.zeros_like(values)
        weights = torch.tensor([[0.2, 0.8]])
        # Rank 1 has no supervised tokens; both ranks must still join the reduce.
        compiled = torch.compile(_loss, backend="eager", fullgraph=True)
        actual = compiled(values, mask, weights)
        actual.backward()
        dist.all_reduce(parameter.grad)
        parameter.grad /= 2
        expected_grad = torch.tensor(2.6 / (1 + 1e-5))
        torch.testing.assert_close(parameter.grad, expected_grad)
        pooled_loss = actual.detach().clone()
        dist.all_reduce(pooled_loss)
        pooled_loss /= 2
        torch.testing.assert_close(pooled_loss, 2 * expected_grad)
    finally:
        dist.destroy_process_group()


def test_distributed_weight_sum_and_fullgraph(tmp_path):
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)


@pytest.mark.parametrize(
    "model", [DFlashDraftModel, DSparkDraftModel, DFlash2DraftModel]
)
@pytest.mark.parametrize("mode", ["weight_sum", "token_count"])
def test_config_and_trainer_roundtrip(model, mode):
    config = TrainConfig.model_validate(
        {"speculator_type": "dspark", "loss": {"loss_normalization": mode}}
    )
    assert config.flatten()["loss_normalization"] == mode
    train, val = model.get_trainer_kwargs(**config.flatten())
    assert train["loss_normalization"] == val["loss_normalization"] == mode


def _cuda_ddp_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(123)
    model = torch.nn.Linear(5, 7, bias=False).to(device)
    features = torch.randn(1, 16, 5, device=device)
    teacher = torch.randn(1, 16, 7, device=device)
    confidence = torch.randn(1, 16, device=device)
    mask = torch.tensor(
        [[1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]], device=device
    )
    # Full blocks remain intact when splitting the very same global batch.
    options = {
        "block_size": 4,
        "loss_config": resolve_loss_config('{"ce": 0.1, "tv": 1.8}', "eager"),
        "per_position_loss_weight": "dpa_dev",
        "dpa_dev_signal": "accept",
    }
    reference, _ = compute_metrics(
        model(features), teacher, confidence, mask, **options
    )
    reference.backward()
    reference_gradient = model.weight.grad.detach().clone()
    reference_weight = model.weight.detach() - 0.01 * reference_gradient
    model.zero_grad(set_to_none=True)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        ddp = DistributedDataParallel(model, device_ids=[rank])
        part = slice(rank * 8, (rank + 1) * 8)
        loss, _ = compute_metrics(
            ddp(features[:, part]),
            teacher[:, part],
            confidence[:, part],
            mask[:, part],
            **options,
        )
        loss.backward()
        pooled = loss.detach().clone()
        dist.all_reduce(pooled)
        pooled /= 2
        torch.testing.assert_close(pooled, reference, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            model.weight.grad, reference_gradient, atol=2e-6, rtol=2e-6
        )
        with torch.no_grad():
            model.weight -= 0.01 * model.weight.grad
        torch.testing.assert_close(model.weight, reference_weight, atol=2e-6, rtol=2e-6)
        if rank == 0:
            gradient_error = (model.weight.grad - reference_gradient).abs().max().item()
            weight_error = (model.weight - reference_weight).abs().max().item()
            print(  # noqa: T201 - numerical evidence for the distributed comparison
                f"single_loss={reference.item():.9f} ddp_loss={pooled.item():.9f} "
                f"gradient_max_abs={gradient_error:.3e} "
                f"updated_weight_max_abs={weight_error:.3e}"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_ddp_matches_same_single_gpu_batch(tmp_path):
    mp.spawn(
        _cuda_ddp_worker, args=(f"file://{tmp_path / 'cuda_rendezvous'}",), nprocs=2
    )
