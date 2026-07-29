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
    assert "MODEL_FILES" in spec
    assert 'datas.append((str(MODEL_DIR), "models/siglip2-base"))' not in spec


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


def test_desktop_shell_stops_the_server_on_exit():
    shell = (
        ROOT / "desktop" / "src-tauri" / "src" / "main.rs"
    ).read_text(encoding="utf-8")
    assert "Mutex<Option<CommandChild>>" in shell
    assert "RunEvent::ExitRequested" in shell
    assert "RunEvent::Exit" in shell
    assert "child.kill()" in shell


def test_photo_navigation_replaces_the_displayed_media():
    app = (ROOT / "desktop" / "ui" / "app.js").read_text(encoding="utf-8")
    assert 'img.dataset.imageId !== String(imageId)' in app
    assert 'img.src = `${API}/images/${imageId}`' in app


def test_release_versions_stay_in_sync():
    assert 'version = "2.0.1"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.0.1"' in (
        ROOT / "desktop" / "src-tauri" / "Cargo.toml"
    ).read_text(encoding="utf-8")
    assert '"version": "2.0.1"' in (
        ROOT / "desktop" / "src-tauri" / "tauri.conf.json"
    ).read_text(encoding="utf-8")
    assert "photolib 2.0.1 setup" in (
        ROOT / "packaging" / "windows" / "sfx-comment.txt"
    ).read_text(encoding="utf-8")
