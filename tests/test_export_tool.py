"""Regression tests for invoking the exporter exactly as CI does."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tools.export_onnx import _quantize


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


def test_quantized_graphs_use_the_quality_preserving_profile(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("vision", "text"):
        (model_dir / f"{name}.onnx").write_bytes(b"fp32")

    calls = []

    def fake_quantize(source, target, **kwargs):
        calls.append((Path(source).name, kwargs))
        Path(target).write_bytes(b"int8")

    monkeypatch.setattr(
        "tools.export_onnx._late_vision_nodes", lambda source: ["late-layer"])
    _quantize(model_dir, quantizer=fake_quantize, weight_type="QUInt8")

    assert calls[0][0] == "vision.onnx"
    assert calls[0][1]["nodes_to_quantize"] == ["late-layer"]
    assert "op_types_to_quantize" not in calls[0][1]
    assert calls[1][1]["op_types_to_quantize"] == ["Gather"]
    assert "nodes_to_quantize" not in calls[1][1]
    assert all(call[1]["per_channel"] for call in calls)
    assert all(call[1]["weight_type"] == "QUInt8" for call in calls)
