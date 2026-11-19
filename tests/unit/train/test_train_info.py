import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators.train.cli import main
from speculators.train.config import TrainConfig
from speculators.train.train_info import TrainInfo


def _record_on_rank(rank: int, directory: Path) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        info = TrainInfo(rank, True, None)
        for step in range(2):
            info.add_step(
                {"global_step": step, "seconds": 1.0},
                {
                    "loss_sum": torch.tensor(rank + step + 1.0),
                    "loss_total": torch.tensor(1.0),
                },
                (rank + 2, 4, 1, 0),
            )
        info.flush()
        info.save(directory / str(rank), dataset={}, config={}, progress={})
    finally:
        dist.destroy_process_group()


@pytest.mark.smoke
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_all_ranks_contribute_only_rank_zero_writes(tmp_path: Path):
    mp.spawn(_record_on_rank, args=(tmp_path,), nprocs=2)
    assert not (tmp_path / "1").exists()
    saved = json.loads((tmp_path / "0" / "train_info.json").read_text())
    assert [row["metrics"]["loss"] for row in saved["train_steps"]] == [1.5, 2.5]
    assert saved["train_summary"]["input_tokens"] == 10
    assert saved["train_summary"]["packed_token_slots"] == 16


@pytest.mark.smoke
@pytest.mark.parametrize("validation_interval", [None, 2])
def test_cpu_dflash_training_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_interval: int | None
):
    """Run the training CLI with real DFlash, cached features and local checkpoints."""
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda **_: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    verifier = tmp_path / "verifier"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).save_pretrained(verifier)

    data = tmp_path / "data"
    generator = torch.Generator().manual_seed(121)
    tokens = torch.randint(0, 32, (6, 12), generator=generator)
    Dataset.from_dict(
        {
            "input_ids": tokens.tolist(),
            "loss_mask": [[1] * 12] * 6,
            "seq_len": [12] * 6,
        }
    ).with_format("torch").save_to_disk(str(data))
    hidden_states = data / "hidden_states"
    hidden_states.mkdir()
    for i, ids in enumerate(tokens):
        save_file(
            {
                "token_ids": ids.clone(),
                "hidden_states": torch.randn(12, 3, 16, generator=generator),
            },
            hidden_states / f"hs_{i}.safetensors",
        )

    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    config = {
        "speculator_type": "dflash",
        "verifier_name_or_path": str(verifier),
        "data_path": str(data),
        "save_path": str(checkpoints),
        "num_layers": 1,
        "target_layer_ids": [0, 1],
        "mask_token_id": 0,
        "draft_attn_impl": "eager",
        "loss_fn": "ce",
        "loss_implementation": "eager",
        "optimizer": "adamw",
        "block_size": 4,
        "max_anchors": 3,
        "hidden_states_dtype": "bfloat16",
        "total_seq_len": 16,
        "train_data_ratio": 0.7,
        "num_workers": 0,
        "on_missing": "raise",
        "epochs": 2,
        "log_freq": 3,
        "validation_interval_steps": validation_interval,
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(TrainConfig.from_flat(config).dump_yaml())
    argv = ["--config", str(config_path)]
    main(TrainConfig.resolve([*argv, "--max-try-steps", "2"]))
    mid = json.loads((checkpoints / "0" / "train_info.json").read_text())
    assert mid["progress"] == {"checkpoint": 0, "global_step": 2}
    assert len(mid["train_steps"]) == 2
    assert len(mid["validation_runs"]) == (0 if validation_interval is None else 1)

    main(TrainConfig.resolve(argv))
    first = json.loads((checkpoints / "0" / "train_info.json").read_text())
    last = json.loads((checkpoints / "1" / "train_info.json").read_text())
    full = json.loads((run / "train_info.json").read_text())
    assert first["progress"] == {"checkpoint": 0, "global_step": 4}
    assert last["progress"] == {"checkpoint": 1, "global_step": 8}
    assert last["dataset"] == {
        "train": {"samples": 4, "approx_tokens": 48, "batches_per_rank": 4},
        "valid": {"samples": 2, "approx_tokens": 24, "batches_per_rank": 2},
    }
    assert len(first["validation_runs"]) == (1 if validation_interval is None else 2)
    assert first["train_steps"] == last["train_steps"][:4]
    assert mid["train_steps"] == first["train_steps"][:2]
    assert (
        mid["validation_runs"]
        == first["validation_runs"][: len(mid["validation_runs"])]
    )
    assert [row["global_step"] for row in last["train_steps"]] == list(range(8))
    assert last["train_steps"][1]["profile"] is None
    assert all(row["metrics"]["loss"] > 0 for row in last["train_steps"])
    assert last["train_summary"]["input_tokens"] == 96
    assert last["train_summary"]["packing_utilization"] == 0.75
    assert last["train_summary"]["input_tokens_per_second"] > 0
    valid_steps = [4, 8] if validation_interval is None else [2, 4, 6, 8]
    assert [row["global_step"] for row in last["validation_runs"]] == valid_steps
    assert all(row["metrics"]["loss_epoch"] > 0 for row in last["validation_runs"])
    assert last["valid_summary"]["input_tokens"] == 24 * len(valid_steps)
    assert last["valid_summary"]["input_tokens_per_second"] > 0
    assert full["train_steps"] == last["train_steps"]
    assert full["validation_runs"] == last["validation_runs"]
    assert f"--config {config_path}" in (run / "train_command.txt").read_text()
    assert not (checkpoints / "train_command.txt").exists()
    for checkpoint in (checkpoints / "0", checkpoints / "1"):
        assert (checkpoint / "model.safetensors").is_file()
        assert not (checkpoint / "train_command.txt").exists()
        assert not (checkpoint / "run.yaml").exists()
        assert not list(checkpoint.glob("*.patch"))
    assert not (run / "run.yaml").exists()
    assert not list(run.glob("*.patch"))
