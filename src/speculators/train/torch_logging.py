"""Configure compiler diagnostics and keep generated code out of run logs."""

import logging
import os
from pathlib import Path

import torch._logging._internal as torch_logging


def _artifact(record: logging.LogRecord) -> str | None:
    return getattr(logging.getLogger(record.name), "artifact_name", None)


class _ArtifactFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        artifact = _artifact(record)
        record.artifactprefix = f"[__{artifact}]" if artifact else ""
        return super().format(record)


class _CodeFilter(logging.Filter):
    def __init__(self, code_only: bool):
        super().__init__()
        self.code_only = code_only

    def filter(self, record: logging.LogRecord) -> bool:
        return (_artifact(record) == "output_code") == self.code_only


def setup_torch_logging(cfg) -> Path:
    """Apply flat logging config after root logging setup, before compilation.

    TORCH_LOGS / TORCH_LOGS_FORMAT override their config counterparts. PyTorch's
    internal registry is needed to cover artifacts created lazily during compile;
    no global Python root logger or structured TORCH_TRACE handler is changed.
    """
    settings = os.environ.get("TORCH_LOGS", cfg.torch_logs)
    fmt = os.environ.get("TORCH_LOGS_FORMAT", cfg.torch_logs_format)
    formatter = _ArtifactFormatter(fmt)
    destination = Path(cfg.torch_output_code_dir or Path(cfg.log_dir) / "torch_output_code")
    destination.mkdir(parents=True, exist_ok=True)
    code_path = destination / f"rank{os.environ.get('RANK', '0')}_torch_compiled_code.txt"

    # Remove only our previous file handlers on repeated setup.
    for name in torch_logging.log_registry.get_log_qnames():
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if getattr(handler, "_speculators_output_code", False):
                logger.removeHandler(handler)
                handler.close()

    previous = os.environ.get("TORCH_LOGS")
    os.environ["TORCH_LOGS"] = settings
    try:
        torch_logging._init_logs()
    finally:
        # Do not turn a config value into an environment override on later setup.
        if previous is None:
            os.environ.pop("TORCH_LOGS", None)
        else:
            os.environ["TORCH_LOGS"] = previous
    code_handler = logging.FileHandler(code_path)
    code_handler._speculators_output_code = True
    code_handler.setLevel(logging.DEBUG)
    code_handler.setFormatter(formatter)
    code_handler.addFilter(_CodeFilter(code_only=True))
    for name in torch_logging.log_registry.get_log_qnames():
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(_CodeFilter(code_only=False))
        # Registered torch loggers do not propagate to each other. Attaching at
        # each boundary also catches output_code artifacts instantiated later.
        logger.addHandler(code_handler)
    return code_path
