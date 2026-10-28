"""Config-file-first configuration for the ``speculators train`` command.

One public type, :class:`~.schema.TrainConfig`, split across ``schema`` (fields),
``resolution`` (CLI + precedence), and ``artifacts`` (reproducibility I/O).
"""

from speculators.train import online_backend as _online_backend  # noqa: F401
from speculators.train import online_speculators_backend as _online_speculators_backend  # noqa: F401

from .schema import TrainConfig

__all__ = ["TrainConfig"]
