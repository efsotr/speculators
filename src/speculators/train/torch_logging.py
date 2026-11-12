"""Configure compiler diagnostics and route FX/generated code to rank files."""

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
    def __init__(self, artifact: str | None):
        super().__init__()
        self.artifact = artifact

    def filter(self, record: logging.LogRecord) -> bool:
        if self.artifact:
            return _artifact(record) == self.artifact
        return _artifact(record) not in {"graph_code", "output_code"}


def setup_torch_logging(cfg) -> Path:
    """Apply flat logging config after root logging setup, before compilation.

    TORCH_LOGS / TORCH_LOGS_FORMAT override their config counterparts. PyTorch's
    internal registry is needed to cover artifacts created lazily during compile;
    no global Python root logger or structured TORCH_TRACE handler is changed.
    """
    settings = os.environ.get("TORCH_LOGS", cfg.torch_logs)
    fmt = os.environ.get("TORCH_LOGS_FORMAT", cfg.torch_logs_format)
    formatter = _ArtifactFormatter(fmt)
    destination = Path(
        cfg.torch_graph_code_dir or Path(cfg.log_dir) / "torch_graph_code"
    )
    destination.mkdir(parents=True, exist_ok=True)
    code_path = destination / f"rank{os.environ.get('RANK', '0')}_graph_code.txt"

    # Remove only our previous file handlers on repeated setup.
    for name in torch_logging.log_registry.get_log_qnames():
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if getattr(handler, "_speculators_graph_code", False):
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
    code_handler._speculators_graph_code = True
    code_handler.setLevel(logging.DEBUG)
    code_handler.setFormatter(formatter)
    code_handler.addFilter(_CodeFilter("graph_code"))
    output_handler = logging.FileHandler(
        destination / f"rank{os.environ.get('RANK', '0')}_output_code.txt"
    )
    output_handler._speculators_graph_code = True
    output_handler.setLevel(logging.DEBUG)
    output_handler.setFormatter(formatter)
    output_handler.addFilter(_CodeFilter("output_code"))
    for name in torch_logging.log_registry.get_log_qnames():
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(_CodeFilter(None))
        # Registered torch loggers do not propagate to each other. Attaching at
        # each boundary also catches graph_code artifacts instantiated later.
        logger.addHandler(code_handler)
        logger.addHandler(output_handler)
    return code_path
