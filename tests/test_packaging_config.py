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
    assert bundle["targets"] == ["nsis", "dmg", "appimage", "deb"]


def test_windows_release_uses_a_clean_single_executable_installer():
    workflow = (
        ROOT / ".github" / "workflows" / "desktop.yml"
    ).read_text(encoding="utf-8")

    assert "bundles: nsis" in workflow
    assert "rm -rf desktop/src-tauri/binaries/_internal" in workflow
    assert "bundle/nsis/*-setup.exe" in workflow
    assert "bundle/**/*.msi" not in workflow


def test_tracked_desktop_ui_is_the_real_app_not_a_placeholder():
    html = (ROOT / "desktop" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="folderPath"' in html
    assert 'src="/app.js"' in html


def test_frozen_build_bundles_only_the_selected_model_and_video_decoder():
    """The standard package must not accidentally regain both model variants."""
    spec = (ROOT / "packaging" / "photolib.spec").read_text(encoding="utf-8")
    assert 'collect_data_files("imageio_ffmpeg")' in spec
    assert '"imageio_ffmpeg"' in spec
    assert '"int8": ("text.int8.onnx", "vision.int8.onnx")' in spec
    assert '"fp32": ("text.onnx", "vision.onnx")' in spec
    assert 'PHOTOLIB_MODEL_VARIANT' in spec
    assert 'datas.append((str(MODEL_DIR), "models/siglip2-base"))' not in spec


def test_desktop_shell_owns_the_complete_server_lifecycle():
    shell = (
        ROOT / "desktop" / "src-tauri" / "src" / "main.rs"
    ).read_text(encoding="utf-8")
    cargo = (
        ROOT / "desktop" / "src-tauri" / "Cargo.toml"
    ).read_text(encoding="utf-8")

    assert "Mutex<ServerState>" in shell
    assert "PHOTOLIB_SHUTDOWN" in shell
    assert "MAX_SERVER_RESTARTS" in shell
    assert "child.write(SHUTDOWN_COMMAND)" in shell
    assert "child.kill()" in shell
    assert "tauri_plugin_single_instance::init" in shell
    assert 'tauri-plugin-single-instance = "2"' in cargo


def test_photo_navigation_replaces_the_displayed_media():
    app = (ROOT / "desktop" / "ui" / "app.js").read_text(encoding="utf-8")
    assert 'img.dataset.imageId !== String(imageId)' in app
    assert 'img.src = `${API}/images/${imageId}`' in app


def test_release_versions_stay_in_sync():
    assert 'version = "2.0.2"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.0.2"' in (
        ROOT / "desktop" / "src-tauri" / "Cargo.toml"
    ).read_text(encoding="utf-8")
    assert '"version": "2.0.2"' in (
        ROOT / "desktop" / "src-tauri" / "tauri.conf.json"
    ).read_text(encoding="utf-8")
