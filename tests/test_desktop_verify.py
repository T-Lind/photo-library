from __future__ import annotations

import sys
from types import ModuleType


def test_desktop_verify_model_mode_exits_with_parity_status(tmp_path, monkeypatch, capsys):
    from photolib import desktop
    from photolib.embeddings import onnx_vision

    class FakeEmbedder:
        def __init__(self, model_dir):
            self.model_dir = model_dir

        def self_check(self):
            return {"checked": True, "ok": True, "text_matches": True}

    monkeypatch.setattr(onnx_vision, "OnnxVisionEmbedder", FakeEmbedder)

    # The frozen-binary smoke test imports the real InsightFace package. This
    # unit test only exercises exit/report plumbing, and importing two native
    # ONNX DLL stacks into the long-lived pytest process is unstable on
    # Windows, so keep that integration boundary explicit here.
    face_app = ModuleType("insightface.app")
    face_app.FaceAnalysis = object
    insightface = ModuleType("insightface")
    insightface.__path__ = []
    insightface.app = face_app
    monkeypatch.setitem(sys.modules, "insightface", insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", face_app)
    code = desktop.main([
        "--verify-model",
        "--data-dir",
        str(tmp_path / "data"),
    ])

    assert code == 0
    assert '"ok": true' in capsys.readouterr().out
