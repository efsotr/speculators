"""Focused checks for compiler diagnostics formatting and graph-code routing."""

import os
import subprocess
import sys
import textwrap

from speculators.train.config import TrainConfig
from speculators.train.config.resolution import resolve


def test_torch_logging_config_defaults_and_native_overrides(tmp_path):
    defaults = TrainConfig().logging
    assert defaults.torch_logs == "graph_breaks,recompiles,graph_code"
    assert defaults.torch_logs_format == "%(artifactprefix)s %(message)s"
    assert defaults.torch_graph_code_dir is None
    config = tmp_path / "config.yaml"
    config.write_text(
        "verifier:\n  verifier_name_or_path: test-model\n"
        "logging:\n  torch_logs: graph_breaks,graph_code\n"
        "  torch_logs_format: '%(message)s'\n"
        "  torch_graph_code_dir: generated\n"
    )
    cfg = resolve(
        TrainConfig, ["--config", str(config), "--torch-logs", "recompiles,graph_code"]
    )
    assert cfg.logging.torch_logs == "recompiles,graph_code"
    assert cfg.logging.torch_logs_format == "%(message)s"
    assert str(cfg.logging.torch_graph_code_dir) == "generated"
    assert cfg.flatten()["torch_logs"] == "recompiles,graph_code"


def test_compile_diagnostics_are_short_and_code_is_separate(tmp_path):
    """Exercise real graph breaks, recompiles and captured FX graph code."""
    script = textwrap.dedent(
        """
        import argparse
        import logging
        import os
        from pathlib import Path
        import torch
        from speculators.train.torch_logging import setup_torch_logging

        root = Path(os.environ['TEST_TORCH_LOG_ROOT'])
        cfg = argparse.Namespace(torch_logs='graph_breaks,recompiles,graph_code',
            torch_logs_format='%(artifactprefix)s %(message)s',
            torch_graph_code_dir=None, log_dir=str(root))
        setup_torch_logging(cfg)
        assert 'TORCH_LOGS' not in os.environ
        torch._logging.getArtifactLogger('torch._dynamo', 'graph_breaks').debug('FORMAT_SENTINEL')

        def broken(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y * 2
        compiled = torch.compile(broken, backend='eager', dynamic=False)
        assert torch.equal(compiled(torch.ones(2)), torch.full((2,), 4.))
        assert torch.equal(compiled(torch.ones(3)), torch.full((3,), 4.))
        native = torch.compile(lambda x: x.sin() + 1, backend='eager', fullgraph=True)
        x = torch.arange(8, dtype=torch.float)
        torch.testing.assert_close(native(x), x.sin() + 1)
        torch._logging.getArtifactLogger('torch._dynamo', 'graph_breaks').debug('MULTILINE_FIRST\\nMULTILINE_SECOND')
        cfg.torch_logs_format = '%(artifactprefix)s | %(message)s'
        setup_torch_logging(cfg)
        assert 'TORCH_LOGS' not in os.environ
        torch._logging.getArtifactLogger('torch._dynamo', 'graph_breaks').debug('CUSTOM_FORMAT_SENTINEL')
        logging.shutdown()
        """
    )
    env = os.environ.copy()
    for name in ("TORCH_LOGS", "TORCH_LOGS_FORMAT", "TORCH_LOGS_OUT", "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    env.update(
        TEST_TORCH_LOG_ROOT=str(tmp_path),
        TORCH_LOGS_OUT=str(tmp_path / "combined.log"),
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor_cache"),
        OMP_NUM_THREADS="1",
        RANK="3",
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True,
        capture_output=True, timeout=120,
    )
    console = result.stdout + result.stderr
    (tmp_path / "console.log").write_text(console)
    assert result.returncode == 0, console
    assert "[__graph_breaks] FORMAT_SENTINEL" in console
    sentinel_lines = [
        line.strip() for line in console.splitlines() if " FORMAT_SENTINEL" in line
    ]
    assert sentinel_lines == ["[__graph_breaks] FORMAT_SENTINEL"]
    custom_lines = [
        line.strip() for line in console.splitlines() if "CUSTOM_FORMAT_SENTINEL" in line
    ]
    assert custom_lines == ["[__graph_breaks] | CUSTOM_FORMAT_SENTINEL"]
    assert "[__graph_breaks] MULTILINE_FIRST\nMULTILINE_SECOND" in console
    assert "[__recompiles]" in console
    assert "Graph break" in console
    assert "def forward(" not in console
    assert "[__graph_code]" not in console
    output = (tmp_path / "torch_graph_code" / "rank3_torch_graph_code.txt").read_text()
    assert "def forward(" in output
    combined = (tmp_path / "combined.log").read_text()
    assert "FORMAT_SENTINEL" in combined
    assert "[__graph_code]" not in combined
    assert "def forward(" not in combined
