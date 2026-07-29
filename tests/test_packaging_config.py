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


def test_frozen_build_bundles_the_video_decoder():
    """imageio-ffmpeg's data files carry the ffmpeg exe; losing them from
    the spec would silently ship a build that skips every video."""
    spec = (ROOT / "packaging" / "photolib.spec").read_text(encoding="utf-8")
    assert 'collect_data_files("imageio_ffmpeg")' in spec
    assert '"imageio_ffmpeg"' in spec  # hiddenimport


def test_windows_sfx_installs_from_a_unique_temporary_directory():
    comment = (
        ROOT / "packaging" / "windows" / "sfx-comment.txt"
    ).read_text(encoding="utf-8")
    installer = (
        ROOT / "packaging" / "windows" / "install-portable.ps1"
    ).read_text(encoding="utf-8")

    assert "TempMode=" in comment
    assert "Setup=powershell.exe" in comment
    assert 'Mutex]::new($false, "Local\\photolib-installer")' in installer
    assert 'Get-Process -Name "photolib", "photolib-server"' in installer
    assert 'Join-Path $env:LOCALAPPDATA "Programs"' in installer
    assert "robocopy.exe" in installer
