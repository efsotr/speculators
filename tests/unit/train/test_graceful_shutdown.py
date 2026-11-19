"""The graceful-shutdown decorator must not leave its handlers installed.

`with_graceful_shutdown()` installs SIGINT/SIGTERM handlers for the duration of
one training call. It used to restore them only when it caught
`TrainingInterruptedError`, so a normal return or an unrelated exception left
this call's handlers in place. In a longer-lived process a later interrupt then
entered a handler belonging to a training run that had already finished.
"""

import signal
from pathlib import Path

import pytest

from speculators.train.graceful_shutdown import (
    GracefulShutdownHandler,
    TrainingInterruptedError,
    with_graceful_shutdown,
)


@pytest.fixture
def original_handlers():
    """Snapshot the process handlers and put them back afterwards."""
    signals = (signal.SIGINT, signal.SIGTERM)
    before = {sig: signal.getsignal(sig) for sig in signals}
    yield before
    for sig, handler in before.items():
        signal.signal(sig, handler)


def _installed(before):
    """True when the process handlers are the ones from before the call."""
    return all(signal.getsignal(sig) == handler for sig, handler in before.items())


class _Checkpointer:
    path = Path()


class _Trainer:
    def __init__(self):
        self.checkpointer = _Checkpointer()
        self.saved = []

    def maybe_save_checkpoint(self, label):
        self.saved.append(label)


def test_handlers_are_restored_after_a_normal_return(original_handlers):
    @with_graceful_shutdown()
    def train(self):
        # The handlers are this call's while it runs; that part already worked.
        assert not _installed(original_handlers)
        return "finished"

    assert train(_Trainer()) == "finished"
    assert _installed(original_handlers), "handlers outlived a successful call"


def test_handlers_are_restored_after_an_unrelated_exception(original_handlers):
    @with_graceful_shutdown()
    def train(self):
        raise ValueError("unrelated")

    with pytest.raises(ValueError, match="unrelated"):
        train(_Trainer())
    assert _installed(original_handlers), "handlers outlived a failed call"


def test_handlers_are_restored_after_an_interruption(original_handlers):
    """The path that already restored must keep doing so."""

    @with_graceful_shutdown()
    def train(self):
        raise TrainingInterruptedError("SIGINT")

    trainer = _Trainer()
    train(trainer)
    assert trainer.saved == ["interrupted"], "the checkpoint save no longer runs"
    assert _installed(original_handlers)


def test_handlers_are_restored_when_the_checkpoint_save_itself_fails(original_handlers):
    """A save failure is caught and logged, and must not strand the handlers."""

    class Failing(_Trainer):
        def maybe_save_checkpoint(self, label):
            raise RuntimeError("disk full")

    @with_graceful_shutdown()
    def train(self):
        raise TrainingInterruptedError("SIGTERM")

    train(Failing())
    assert _installed(original_handlers)


def test_nested_calls_unwind_to_the_original_handlers(original_handlers):
    """Repeated and nested decorated calls must not capture stale handlers."""

    @with_graceful_shutdown()
    def inner(self):
        return "inner"

    @with_graceful_shutdown()
    def outer(self):
        outer_handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        assert inner(self) == "inner"
        # The inner call restores what it found, which is the outer call's.
        for sig, handler in outer_handlers.items():
            assert signal.getsignal(sig) == handler
        return "outer"

    assert outer(_Trainer()) == "outer"
    assert _installed(original_handlers)

    # And running the same decorated function twice ends where it started.
    assert inner(_Trainer()) == "inner"
    assert _installed(original_handlers)


def test_restore_is_idempotent(original_handlers):
    """The decorator restores in both the except branch and the finally block,
    so a second restore must not undo a handler installed in between."""
    handler = GracefulShutdownHandler()
    handler.install()
    handler.restore()
    assert _installed(original_handlers)

    def someone_elses_handler(signum, frame):
        raise AssertionError("never called")

    signal.signal(signal.SIGINT, someone_elses_handler)
    handler.restore()
    assert signal.getsignal(signal.SIGINT) is someone_elses_handler


def test_install_is_idempotent(original_handlers):
    """A second install() must not capture this handler as the original."""
    handler = GracefulShutdownHandler()
    handler.install()
    handler.install()
    handler.restore()
    assert _installed(original_handlers)


@pytest.mark.smoke
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("mode", ["finish_batch", "immediate"])
def test_signal_save_observes_complete_update(original_handlers, sig, mode):
    """A real signal during backward must defer saving until the update boundary."""
    import os
    from types import SimpleNamespace

    import torch

    class Updating(_Trainer):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(shutdown_mode=mode)
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.optimizer = torch.optim.SGD([self.weight], lr=0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 1, gamma=0.5
            )
            self.global_step = 0
            self.snapshot = None

        def maybe_save_checkpoint(self, label):
            self.snapshot = (
                self.weight.item(),
                self.global_step,
                self.scheduler.last_epoch,
            )
            super().maybe_save_checkpoint(label)

        @with_graceful_shutdown()
        def train(self):
            self.weight.register_hook(lambda grad: (os.kill(os.getpid(), sig), grad)[1])
            self.weight.square().backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            self._shutdown_handler.check_boundary()
            pytest.fail("shutdown should prevent the next batch")

    trainer = Updating()
    trainer.train()
    expected = (0.8, 1, 1) if mode == "finish_batch" else (1.0, 0, 0)
    assert trainer.snapshot == pytest.approx(expected)
    assert trainer.saved == ["interrupted"]
    assert _installed(original_handlers)


@pytest.mark.smoke
def test_default_defers_and_ignores_duplicate_signals(original_handlers):
    import os

    handler = GracefulShutdownHandler()
    handler.install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGINT)
        with pytest.raises(TrainingInterruptedError, match="batch boundary"):
            handler.check_boundary()
    finally:
        handler.restore()


@pytest.mark.smoke
def test_remote_rank_request_is_coordinated(original_handlers, monkeypatch):
    import torch.distributed as dist

    handler = GracefulShutdownHandler()

    def remote_request(flag, op):
        assert op == dist.ReduceOp.MAX
        flag.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", remote_request)
    with pytest.raises(TrainingInterruptedError):
        handler.check_boundary(distributed=True, device="cpu")


@pytest.mark.smoke
def test_shutdown_config_default_and_validation():
    from pydantic import ValidationError

    from speculators.train.config.schema import TrainerArgs
    from speculators.train.trainer import TrainerConfig

    assert TrainerArgs().shutdown_mode == "finish_batch"
    assert (
        TrainerConfig(lr=0.1, num_epochs=1, save_path=".").shutdown_mode
        == "finish_batch"
    )
    assert TrainerArgs(shutdown_mode="immediate").shutdown_mode == "immediate"
    with pytest.raises(ValidationError):
        TrainerArgs(shutdown_mode="invalid")


@pytest.mark.smoke
def test_real_train_loop_finishes_one_batch_before_save(original_handlers):
    import os

    import torch

    from speculators.train.trainer import Trainer, TrainerConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            os.kill(os.getpid(), signal.SIGTERM)
            return None, self.weight.square(), {}

    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(
        lr=0.1,
        num_epochs=1,
        save_path=".",
        log_freq=100,
        hidden_states_dtype=torch.bfloat16,
    )
    trainer.model = Model()
    trainer.optimizers = [torch.optim.SGD(trainer.model.parameters(), lr=0.1)]
    trainer.schedulers = [torch.optim.lr_scheduler.StepLR(trainer.optimizers[0], 1)]
    trainer.local_rank = "cpu"
    trainer.rank = 1  # Disable progress output.
    trainer.device_type = "cpu"
    trainer.is_distributed = False
    trainer.global_step = 1  # Disable optional metrics for this batch.
    trainer.train_loader = torch.utils.data.DataLoader([{"error_records": 0}] * 3)
    trainer.checkpointer = _Checkpointer()
    saved = []
    trainer.maybe_save_checkpoint = lambda label: saved.append(
        (
            label,
            trainer.model.weight.item(),
            trainer.global_step,
            trainer.schedulers[0].last_epoch,
        )
    )

    @with_graceful_shutdown()
    def train(self):
        self.train_epoch(0)

    train(trainer)
    assert trainer.model.calls == 1
    assert saved[0][0] == "interrupted"
    assert saved[0][1:] == pytest.approx((0.2, 2, 1))


def _distributed_shutdown_worker(rank, init_file, output_dir):
    import os
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    handler = GracefulShutdownHandler()
    handler.install()
    try:
        if rank == 0:
            os.kill(os.getpid(), signal.SIGTERM)
        dist.barrier()
        with pytest.raises(TrainingInterruptedError):
            handler.check_boundary(distributed=True, device="cpu")
        Path(output_dir, str(rank)).write_text("stopped")
    finally:
        handler.restore()
        dist.destroy_process_group()


@pytest.mark.smoke
def test_two_ranks_stop_when_only_one_receives_signal(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _distributed_shutdown_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        start_method="fork",
        join=True,
    )
    assert (tmp_path / "0").read_text() == "stopped"
    assert (tmp_path / "1").read_text() == "stopped"
