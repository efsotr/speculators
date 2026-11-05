"""Unit tests for training dataloader setup."""

from unittest.mock import Mock

import pytest

from speculators.train import dataloader
from speculators.train.config.schema import DataArgs


@pytest.mark.parametrize("max_val_samples", [0, 10, None])
def test_validation_can_be_disabled_without_changing_training(
    monkeypatch, max_val_samples
):
    config = DataArgs(max_val_samples=max_val_samples)
    dataset = Mock()
    loader = Mock()
    monkeypatch.setattr(dataloader, "ArrowDataset", dataset)
    monkeypatch.setattr(dataloader, "_setup_dataloader", loader)

    train_loader, val_loader = dataloader.create_train_val_loaders(
        data_path="unused",
        total_seq_len=8192,
        hidden_states_dtype=dataloader.torch.bfloat16,
        vllm_endpoint="",
        on_missing="raise",
        verifier_name_or_path="unused",
        request_timeout=None,
        max_retries=0,
        generation_validation_retries=0,
        max_consecutive_generation_failures=1,
        hidden_size=16,
        num_target_layers=5,
        num_workers=0,
        prefetch_factor=4,
        preprocess=None,
        train_data_ratio=0.95,
        max_train_samples=400000,
        max_val_samples=config.max_val_samples,
    )

    assert train_loader is loader.return_value
    train_args = dataset.call_args_list[0].kwargs
    assert train_args["split"] == "train"
    assert train_args["train_ratio"] == 0.95
    assert train_args["max_samples"] == 400000
    if max_val_samples == 0:
        assert val_loader is None
        assert dataset.call_count == loader.call_count == 1
    else:
        assert val_loader is loader.return_value
        assert dataset.call_count == loader.call_count == 2
        assert dataset.call_args.kwargs["split"] == "val"
        assert dataset.call_args.kwargs["max_samples"] == max_val_samples


def test_validation_limit_rejects_negative_values():
    with pytest.raises(ValueError):
        DataArgs(max_val_samples=-1)


def test_worker_init_binds_to_local_rank(monkeypatch):
    set_num_threads = Mock()
    is_available = Mock(return_value=True)
    set_device_index = Mock()

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(dataloader.torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(dataloader.torch.accelerator, "is_available", is_available)
    monkeypatch.setattr(
        dataloader.torch.accelerator, "set_device_index", set_device_index
    )

    dataloader._worker_init_fn(worker_id=0)

    set_num_threads.assert_called_once_with(1)
    is_available.assert_called_once_with()
    set_device_index.assert_called_once_with(3)


def test_worker_init_skips_device_binding_without_accelerator(monkeypatch):
    set_num_threads = Mock()
    is_available = Mock(return_value=False)
    set_device_index = Mock()

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(dataloader.torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(dataloader.torch.accelerator, "is_available", is_available)
    monkeypatch.setattr(
        dataloader.torch.accelerator, "set_device_index", set_device_index
    )

    dataloader._worker_init_fn(worker_id=0)

    set_num_threads.assert_called_once_with(1)
    is_available.assert_called_once_with()
    set_device_index.assert_not_called()
