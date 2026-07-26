from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_tauri_bundles_pyinstaller_support_directory():
    config = json.loads(
        (ROOT / "desktop" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )
    bundle = config["bundle"]
    assert bundle["externalBin"] == ["binaries/photolib-server"]
    assert bundle["resources"]["binaries/_internal/"] == "_internal/"
    assert "msi" in bundle["targets"]
    assert "nsis" not in bundle["targets"]


def test_tracked_desktop_ui_is_the_real_app_not_a_placeholder():
    html = (ROOT / "desktop" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="folderPath"' in html
    assert 'src="/app.js"' in html
