"""Persistent sparse indices for max-autotune, updated outside compiled forward.

Call only after the preceding batch's backward finishes. Shapes/dtype/device must
stay fixed; fail explicitly instead of silently reallocating and hiding captures.
The new mask object and its mask_mod closure remain batch-specific.
"""

import torch


INDEX_NAMES = (
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
)


class StableBlockMasks:
    """Keep full/sliding sparse indices at fixed addresses across sequential batches.

    These runtime tensors are intentionally outside module buffers: they are
    created on the input device and excluded from FSDP state and checkpoints.
    """

    def __init__(self):
        self.buffers = {}
        self.signature = None

    @torch.compiler.disable
    def update(self, prepared_attention_mask):
        """Update full/sliding indices; preserve anchors, validity and closures."""
        full, sliding, anchors, valid = prepared_attention_mask
        current = {}
        signature = {}
        for label, mask in (("full", full), ("sliding", sliding)):
            signature[label] = mask is not None
            if mask is None:
                continue
            for name in INDEX_NAMES:
                value = getattr(mask, name)
                key = (label, name)
                signature[key] = (
                    (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
                    if value is not None else None
                )
                if value is not None:
                    current[key] = value
        if self.signature is not None and signature != self.signature:
            raise ValueError(
                "Stable BlockMask requires unchanged index shapes/strides/dtype/device"
            )
        if self.signature is None:
            self.buffers = {key: value.clone() for key, value in current.items()}
            self.signature = signature
        else:
            for key, value in current.items():
                self.buffers[key].copy_(value)
        for label, mask in (("full", full), ("sliding", sliding)):
            if mask is not None:
                for name in INDEX_NAMES:
                    key = (label, name)
                    if key in self.buffers:
                        setattr(mask, name, self.buffers[key])
        return full, sliding, anchors, valid

    def pointers(self):
        """No device synchronization: record storage identity alongside captures."""
        return {
            f"{label}.{name}": value.data_ptr()
            for (label, name), value in self.buffers.items()
        }
