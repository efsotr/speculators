"""Opt-in per-rank CUDA peak records, including dataloader-side online forwards."""
import json
import os
import time

import torch

from speculators.train.distributed import get_rank


class CudaMemoryProfile:
    """Reset before fetching, record after each step; disabled by default.

    Resetting inside the loop body would miss CUDA work in next(dataloader),
    including lazy teacher initialization and online hidden-state extraction.
    Do not empty the allocator cache: reserved peaks must reflect real training.
    """

    def __init__(self, device, phase, epoch):
        self.enabled = os.environ.get('SPECULATORS_MEMORY_PROFILE') == '1' and torch.cuda.is_available()
        self.device = device
        self.phase = phase
        self.epoch = epoch
        self.phase_peak_allocated = 0
        self.phase_peak_reserved = 0
        if self.enabled:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def record(self, step, batch):
        if not self.enabled:
            return
        # Batch tensors originate on CPU; inspect them without allocating CUDA
        # temporaries that would distort the actual workload peak.
        fields = {}
        documents = batch.get('document_ids')
        if isinstance(documents, torch.Tensor):
            documents = documents.detach().cpu()
            valid = documents[documents >= 0]
            lengths = torch.unique(valid, return_counts=True)[1]
            fields.update(
                effective_tokens=int(valid.numel()),
                packed_token_footprint=int(documents.numel()),
                document_count=int(lengths.numel()),
                min_document_tokens=int(lengths.min()) if lengths.numel() else 0,
                max_document_tokens=int(lengths.max()) if lengths.numel() else 0,
                packed_shape=list(documents.shape),
            )
        loss_mask = batch.get('loss_mask')
        if isinstance(loss_mask, torch.Tensor):
            fields['supervised_tokens'] = int(loss_mask.detach().cpu().bool().sum())
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        self.phase_peak_allocated = max(self.phase_peak_allocated, peak_allocated)
        self.phase_peak_reserved = max(self.phase_peak_reserved, peak_reserved)
        record = dict(
            phase=self.phase, epoch=self.epoch, global_step=step,
            rank=get_rank(), local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            unix_seconds=time.time(),
            allocated_bytes=torch.cuda.memory_allocated(self.device),
            reserved_bytes=torch.cuda.memory_reserved(self.device),
            peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
            phase_peak_allocated_bytes=self.phase_peak_allocated,
            phase_peak_reserved_bytes=self.phase_peak_reserved,
            device_free_bytes=free, device_total_bytes=total, **fields,
        )
        print('SPECULATORS_MEMORY ' + json.dumps(record), flush=True)
        # This reset happens before the following next(loader), even when the
        # loader performs CUDA work. Existing live tensors remain in the baseline.
        torch.cuda.reset_peak_memory_stats(self.device)
