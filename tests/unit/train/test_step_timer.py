import logging
from unittest.mock import patch

import pytest

from speculators.train.logger import FormatDictFilter
from speculators.train.trainer import _StepTimer


@pytest.fixture(autouse=True)
def mock_accelerator_availability(monkeypatch):
    # Timer unit tests must not initialize a real accelerator.
    monkeypatch.setattr("torch.accelerator.is_available", lambda: False)


def test_disabled_timer_returns_none():
    timer = _StepTimer(enabled=False)
    timer.mark_value("start", 0.0)
    timer.mark("queue")
    timer.mark("pre_h2d")
    timer.mark("fetch")
    timer.mark("fwd")
    timer.mark("pre_clip")
    timer.mark("bwd")
    timer.mark("opt")
    assert timer.now() is None
    assert timer.profile(num_tokens=1024) is None


@patch("speculators.train.trainer.torch.accelerator.synchronize")
def test_enabled_timer_returns_profile(mock_sync):
    timer = _StepTimer(enabled=True)

    with patch(
        "speculators.train.trainer.time.perf_counter",
        side_effect=[
            0.05,
            0.08,
            0.1,
            0.3,
            0.45,
            0.5,
            0.6,
            0.6,
        ],
    ):
        timer.mark_value("start", 0.0)
        timer.mark("queue")
        timer.mark("pre_h2d")
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("pre_clip")
        timer.mark("bwd")
        timer.mark("opt")
        t_next = timer.now()

    assert mock_sync.call_count == 8
    assert t_next == 0.6

    profile = timer.profile(num_tokens=4096)
    assert profile is not None
    assert profile["queue_ms"] == (0.05 - 0.0) * 1000
    assert profile["h2d_ms"] == (0.1 - 0.08) * 1000
    assert profile["fetch_ms"] == (0.1 - 0.0) * 1000
    assert profile["fwd_ms"] == (0.3 - 0.1) * 1000
    assert profile["clip_ms"] == (0.5 - 0.45) * 1000
    assert profile["bwd_ms"] == (0.5 - 0.3) * 1000
    assert profile["opt_ms"] == (0.6 - 0.5) * 1000
    assert profile["step_ms"] == (0.6 - 0.0) * 1000
    assert profile["tokens_per_s"] == 4096 / 0.6
    assert profile["fetch_frac"] == 100 / 600


def test_disabled_to_enabled_transition():
    """Simulate log_freq=2: a disabled step followed by an enabled step.

    The disabled step's ``timer.now()`` returns None so the training loop
    falls back to ``time.perf_counter()`` for ``t_before_fetch``.  The
    subsequent enabled step feeds that value into ``mark_value("start", ...)``.
    Verify the profile is valid with a realistic ``start`` mark.
    """
    timer = _StepTimer()

    # --- disabled step (global_step=1, log_freq=2 → 1%2 != 0) ---
    timer.reset(enabled=False)
    timer.mark_value("start", 1.0)
    timer.mark("queue")
    timer.mark("pre_h2d")
    timer.mark("fetch")
    timer.mark("fwd")
    timer.mark("pre_clip")
    timer.mark("bwd")
    timer.mark("opt")
    assert timer.now() is None
    assert timer.profile(num_tokens=512) is None

    # Simulate the fallback: t_before_fetch = timer.now() or time.perf_counter()
    t_before_fetch = 2.0  # stands in for the perf_counter() fallback

    # --- enabled step (global_step=2, log_freq=2 → 2%2 == 0) ---
    timer.reset(enabled=True)
    timer.mark_value("start", t_before_fetch)

    with (
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch(
            "speculators.train.trainer.time.perf_counter",
            side_effect=[2.05, 2.08, 2.1, 2.4, 2.45, 2.5, 2.6, 2.6],
        ),
    ):
        timer.mark("queue")
        timer.mark("pre_h2d")
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("pre_clip")
        timer.mark("bwd")
        timer.mark("opt")
        t_next = timer.now()

    assert t_next == 2.6
    profile = timer.profile(num_tokens=2048)
    assert profile is not None
    assert profile["queue_ms"] == (2.05 - 2.0) * 1000
    assert profile["h2d_ms"] == (2.1 - 2.08) * 1000
    assert profile["fetch_ms"] == (2.1 - 2.0) * 1000
    assert profile["fwd_ms"] == (2.4 - 2.1) * 1000
    assert profile["clip_ms"] == (2.5 - 2.45) * 1000
    assert profile["bwd_ms"] == (2.5 - 2.4) * 1000
    assert profile["opt_ms"] == (2.6 - 2.5) * 1000
    assert profile["step_ms"] == (2.6 - 2.0) * 1000
    assert profile["tokens_per_s"] == pytest.approx(2048 / 0.6)


def test_zero_step_ms_returns_zero_throughput():
    timer = _StepTimer(enabled=True)
    timer.mark_value("start", 1.0)
    with (
        patch("speculators.train.trainer.torch.accelerator.synchronize"),
        patch("speculators.train.trainer.time.perf_counter", return_value=1.0),
    ):
        timer.mark("queue")
        timer.mark("pre_h2d")
        timer.mark("fetch")
        timer.mark("fwd")
        timer.mark("pre_clip")
        timer.mark("bwd")
        timer.mark("opt")
    profile = timer.profile(num_tokens=4096)
    assert profile is not None
    assert profile["tokens_per_s"] == 0.0
    assert profile["fetch_frac"] == 0.0


def test_memory_metrics_are_logged_in_mib_and_reset_between_steps(monkeypatch):
    monkeypatch.setattr("torch.accelerator.is_available", lambda: True)
    phases = ("queue", "pre_h2d", "fetch", "fwd", "pre_clip", "bwd", "opt")
    timer = _StepTimer(enabled=True)
    timer.mark_value("start", 0.0)
    with (
        patch("torch.accelerator.synchronize"),
        patch("torch.accelerator.memory_allocated", return_value=1.5 * 1024**2),
        patch("torch.accelerator.memory_reserved", return_value=3 * 1024**2),
        patch("torch.accelerator.max_memory_allocated", return_value=2 * 1024**2),
        patch("torch.accelerator.max_memory_reserved", return_value=4 * 1024**2),
        patch("torch.accelerator.reset_peak_memory_stats") as reset_peaks,
    ):
        for phase in phases:
            timer.mark(phase)
        assert reset_peaks.call_count == len(phases)

    profile = timer.profile(num_tokens=1024)
    expected = {
        "allocated": 1.5,
        "cached": 3.0,
        "max-allocated": 2.0,
        "max-cached": 4.0,
    }
    assert profile["memory_mb"] == {"rank0": dict.fromkeys(phases, expected)}
    record = logging.makeLogRecord({"msg": {"profile": profile}})
    FormatDictFilter().filter(record)
    for phase in phases:
        for metric, value in expected.items():
            assert f"profile/memory_mb/rank0/{phase}/{metric}={value:.3f}" in record.msg

    timer.reset(enabled=False)
    assert timer._memory == {}
    with patch("torch.accelerator.memory_allocated") as allocated:
        timer.mark("fwd")
        allocated.assert_not_called()
    assert timer.profile(num_tokens=1024) is None
    # Resetting the timer leaves a previously returned profile intact.
    assert set(profile["memory_mb"]["rank0"]) == set(phases)


def test_no_memory_metrics_without_accelerator():
    timer = _StepTimer(enabled=True)
    timer.mark_value("start", 0.0)
    with (
        patch("torch.accelerator.synchronize"),
        patch("torch.accelerator.memory_reserved") as reserved,
    ):
        for phase in ("queue", "pre_h2d", "fetch", "fwd", "pre_clip", "bwd", "opt"):
            timer.mark(phase)
        reserved.assert_not_called()
    assert "memory_mb" not in timer.profile(num_tokens=1024)


@pytest.mark.parametrize("rank", [0, 1])
def test_memory_gather_preserves_each_rank_without_reducing(rank):
    timer = _StepTimer(enabled=True)
    timer._marks = dict.fromkeys(
        ("start", "queue", "pre_h2d", "fetch", "fwd", "pre_clip", "bwd", "opt"), 1.0
    )
    memories = [
        {
            "fwd": {
                "allocated": 1.0,
                "cached": 2.0,
                "max-allocated": 3.0,
                "max-cached": 4.0,
            }
        },
        {
            "fwd": {
                "allocated": 11.0,
                "cached": 12.0,
                "max-allocated": 13.0,
                "max-cached": 14.0,
            }
        },
    ]
    timer._memory = memories[rank]

    def gather(memory, object_gather_list, dst):
        assert memory == memories[rank]
        assert dst == 0
        if rank == 0:
            object_gather_list[:] = memories
        else:
            assert object_gather_list is None

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.gather_object", side_effect=gather) as collective,
    ):
        profile = timer.profile(num_tokens=1024)
    collective.assert_called_once()
    if rank == 0:
        assert profile["memory_mb"] == {"rank0": memories[0], "rank1": memories[1]}
    else:
        assert "memory_mb" not in profile
