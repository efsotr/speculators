"""Unit tests for training dataloader setup."""

from unittest.mock import Mock

import pytest

from speculators.train import dataloader


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


@pytest.mark.parametrize("limit", [0, 10, None])
@pytest.mark.parametrize("ratio", [0.95, 1.0, 0.0, 1.1])
def test_validation_limit_controls_loader_construction(monkeypatch, limit, ratio):
    datasets = Mock(side_effect=lambda **kwargs: Mock(split=kwargs["split"]))
    loaders = Mock(side_effect=lambda dataset, *args, **kwargs: dataset.split)
    monkeypatch.setattr(dataloader, "ArrowDataset", datasets)
    monkeypatch.setattr(dataloader, "_setup_dataloader", loaders)
    kwargs = dict(
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
        hidden_size=8,
        num_target_layers=1,
        num_workers=0,
        prefetch_factor=4,
        preprocess=None,
        max_val_samples=limit,
        train_data_ratio=ratio,
    )
    if not 0 < ratio <= 1 or (ratio == 1 and limit != 0):
        with pytest.raises(ValueError, match="train_data_ratio"):
            dataloader.create_train_val_loaders(**kwargs)
        datasets.assert_not_called()
        loaders.assert_not_called()
        return
    train, val = dataloader.create_train_val_loaders(**kwargs)
    assert datasets.call_args_list[0].kwargs["train_ratio"] == ratio
    assert train == "train"
    assert val == (None if limit == 0 else "val")
    assert [call.kwargs["split"] for call in datasets.call_args_list] == (
        ["train"] if limit == 0 else ["train", "val"]
    )
    assert loaders.call_count == (1 if limit == 0 else 2)
    if limit != 0:
        assert datasets.call_args.kwargs["max_samples"] == limit
