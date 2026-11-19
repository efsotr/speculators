"""Rank-zero training history with windowed reduction of per-step metrics."""

import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from speculators.provenance import atomic_write
from speculators.train.utils import normalize_counted_metrics

logger = logging.getLogger("speculators")
BATCH_COUNT_FIELDS = (
    "input_tokens",
    "packed_token_slots",
    "documents",
    "error_records",
)
FILENAME = "train_info.json"


def batch_counts(batch: dict) -> tuple[int, int, int, int]:
    # Document IDs are CPU metadata in the packed loader, even when the
    # verifier backend returns hidden states on the accelerator.
    docs = batch.get("document_ids")
    errors = int(batch["error_records"])
    if docs is None or docs.numel() == 0:
        return 0, 0, 0, errors
    valid = docs >= 0
    starts = valid[..., 1:] & (docs[..., 1:] != docs[..., :-1])
    return (
        int(valid.sum()),
        docs.numel(),
        int(valid[..., :1].sum() + starts.sum()),
        errors,
    )


def describe_split(loader: DataLoader | None) -> dict | None:
    if loader is None:
        return None
    dataset = loader.dataset
    lengths = getattr(dataset, "approx_lengths", None)
    return {
        "samples": len(dataset),
        "approx_tokens": int(sum(lengths)) if lengths is not None else None,
        "batches_per_rank": len(loader),
    }


def _json_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        key: value if math.isfinite(value) else None for key, value in metrics.items()
    }


def _summary(rows: list[dict]) -> dict:
    totals = {key: sum(row[key] for row in rows) for key in BATCH_COUNT_FIELDS}
    seconds = sum(row["seconds"] for row in rows)
    return {
        **totals,
        "batches_per_rank": sum(row["batches_per_rank"] for row in rows),
        "seconds": seconds,
        "input_tokens_per_second": totals["input_tokens"] / seconds
        if seconds
        else None,
        "packing_utilization": (
            totals["input_tokens"] / totals["packed_token_slots"]
            if totals["packed_token_slots"]
            else None
        ),
        "last_metrics": rows[-1]["metrics"] if rows else None,
    }


class TrainInfo:
    def __init__(self, rank: int, distributed: bool, resume_path: Path | None):
        self.rank = rank
        self.distributed = distributed
        self.pending: list[tuple[dict, dict[str, torch.Tensor], tuple[int, ...]]] = []
        self.train_steps: list[dict] = []
        self.validation_runs: list[dict] = []
        if rank == 0 and resume_path is not None:
            path = resume_path / FILENAME
            try:
                previous = json.loads(path.read_text())
                self.train_steps = previous["train_steps"]
                self.validation_runs = previous["validation_runs"]
            except FileNotFoundError:
                pass  # Older checkpoints predate training history.
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("Failed to load training history from %s: %s", path, exc)

    def add_step(
        self, row: dict, metrics: dict[str, torch.Tensor], counts: tuple[int, ...]
    ) -> None:
        self.pending.append((row, {k: v.detach() for k, v in metrics.items()}, counts))

    def flush(self) -> dict[str, float] | None:
        if not self.pending:
            return None
        started = time.perf_counter()
        # Batch the collective and host transfers. Flatten the window so
        # consecutive steps can report different metric keys.
        metrics = [
            value.reshape(())
            for _, values, _ in self.pending
            for value in values.values()
        ]
        packed = torch.stack(metrics).float()
        counts = torch.tensor([c for _, _, c in self.pending], dtype=torch.int64)
        if self.distributed:
            dist.all_reduce(packed)
            counts = counts.to(packed.device)
            dist.all_reduce(counts)
        latest = None
        if self.rank == 0:
            world_size = dist.get_world_size() if self.distributed else 1
            values = iter(packed.tolist())
            for (row, keys, _), batch in zip(
                self.pending, counts.tolist(), strict=True
            ):
                latest = normalize_counted_metrics(
                    {key: next(values) for key in keys}, world_size
                )
                self.train_steps.append(
                    {
                        **row,
                        **dict(zip(BATCH_COUNT_FIELDS, batch, strict=True)),
                        "batches_per_rank": 1,
                        "metrics": _json_metrics(latest),
                    }
                )
            # Include time spent waiting for asynchronous work at the flush.
            self.train_steps[-1]["seconds"] += time.perf_counter() - started
        self.pending.clear()
        return latest

    def add_validation(self, row: dict, metrics: dict[str, float]) -> None:
        if self.rank == 0:
            self.validation_runs.append({**row, "metrics": _json_metrics(metrics)})

    def save(self, path: Path, *, dataset: dict, config: dict, progress: dict) -> None:
        if self.rank != 0:
            return
        train_summary = {"steps": len(self.train_steps), **_summary(self.train_steps)}
        valid_summary = {
            "runs": len(self.validation_runs),
            **_summary(self.validation_runs),
        }
        payload = {
            "schema_version": 1,
            "progress": progress,
            "config": config,
            "dataset": dataset,
            "train_summary": train_summary,
            "valid_summary": valid_summary,
            "train_steps": self.train_steps,
            "validation_runs": self.validation_runs,
        }
        path.mkdir(parents=True, exist_ok=True)
        atomic_write(
            path / FILENAME, json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
