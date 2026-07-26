"""Regression tests for invoking the exporter exactly as CI does."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_exporter_is_runnable_by_file_path_from_any_directory(tmp_path):
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "export_onnx.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
