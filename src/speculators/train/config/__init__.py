"""Config-file-first configuration for ``scripts/train.py``.

One public type, :class:`~.schema.TrainConfig`, split across ``schema`` (fields),
``resolution`` (CLI + precedence), and ``artifacts`` (reproducibility I/O).
"""

# Register built-in shard backends before resolving backend configuration.
from speculators.train.shards import shard_backend as _shard_backend  # noqa: F401
from speculators.train import online_backend as _online_backend  # noqa: F401

from .schema import TrainConfig

__all__ = ["TrainConfig"]
