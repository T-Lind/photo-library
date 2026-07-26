"""Model asset management: status, downloads, and offline enforcement.

The real face-model download is ~288 MB from GitHub, so the download path is
exercised against a local HTTP server serving a small archive. That covers
everything that can actually go wrong locally — streaming, extraction,
archive traversal, checksum mismatch, partial state — without the network.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import threading
import zipfile
from pathlib import Path

import pytest

from photolib.models import (DownloadError, FACE_MODELS, ModelSpec, OfflineError,
                             ensure_face_model, enforce_offline_env, offline,
                             status)


def _zip_bytes(members: dict, top_level: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(f"{top_level}/{name}" if top_level else name, data)
    return buffer.getvalue()


@pytest.fixture
def file_server(tmp_path):
    """Serves tmp_path over HTTP on a free port."""
    directory = tmp_path / "served"
    directory.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield directory, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def fake_model(file_server, monkeypatch, settings, tmp_path):
    """Register a small downloadable model in place of buffalo_l."""
    directory, base = file_server
    payload = _zip_bytes({"det_10g.onnx": b"detector", "w600k_r50.onnx": b"recogniser"},
                         top_level="buffalo_l")
    (directory / "buffalo_l.zip").write_bytes(payload)

    spec = ModelSpec(
        name="buffalo_l", kind="face", url=f"{base}/buffalo_l.zip",
        members=["det_10g.onnx", "w600k_r50.onnx"], approx_bytes=len(payload),
        licence="test")
    monkeypatch.setitem(FACE_MODELS, "buffalo_l", spec)
    # The shared `settings` fixture configures stub models; this fixture is
    # specifically about the real InsightFace download path.
    monkeypatch.setenv("PHOTO_FACE_BACKEND", "insightface")
    monkeypatch.setenv("PHOTO_FACE_MODEL", "buffalo_l")
    monkeypatch.setenv("PHOTO_FACE_MODEL_ROOT", str(tmp_path / "insightface"))

    from photolib.config import get_settings, reset_settings_cache

    reset_settings_cache()
    yield spec, get_settings(), directory
    reset_settings_cache()


# -- status ----------------------------------------------------------------

def test_status_reports_a_missing_model(fake_model):
    _, cfg, _ = fake_model
    report = status(cfg)

    assert report["ready"] is False
    assert "buffalo_l" in report["missing"]
    assert report["download_bytes"] > 0
    entry = next(m for m in report["models"] if m["name"] == "buffalo_l")
    # The URL and licence are surfaced so "what is it downloading?" is answerable.
    assert entry["url"].endswith("buffalo_l.zip")
    assert entry["licence"]


def test_status_is_ready_once_installed(fake_model):
    _, cfg, _ = fake_model
    ensure_face_model(cfg)
    assert status(cfg)["ready"] is True


def test_stub_backend_needs_no_models(settings):
    # The test/CI configuration must never claim it needs a download.
    assert status(settings)["ready"] is True


# -- download --------------------------------------------------------------

def test_download_extracts_the_expected_files(fake_model):
    spec, cfg, _ = fake_model
    dest = ensure_face_model(cfg)

    assert (dest / "det_10g.onnx").read_bytes() == b"detector"
    assert (dest / "w600k_r50.onnx").read_bytes() == b"recogniser"


def test_download_reports_progress(fake_model):
    _, cfg, _ = fake_model
    seen = []
    ensure_face_model(cfg, lambda name, done, total: seen.append((name, done, total)))

    assert seen
    assert seen[-1][1] == seen[-1][2]  # finished at 100%


def test_second_call_does_not_redownload(fake_model):
    spec, cfg, directory = fake_model
    ensure_face_model(cfg)

    # Remove the archive: a second call must be satisfied from disk alone.
    (directory / "buffalo_l.zip").unlink()
    assert ensure_face_model(cfg).exists()


def test_archive_without_a_top_level_folder(fake_model, file_server):
    spec, cfg, directory = fake_model
    (directory / "buffalo_l.zip").write_bytes(
        _zip_bytes({"det_10g.onnx": b"d", "w600k_r50.onnx": b"r"}))

    dest = ensure_face_model(cfg)
    assert (dest / "det_10g.onnx").exists()


def test_incomplete_archive_is_rejected(fake_model, file_server):
    spec, cfg, directory = fake_model
    (directory / "buffalo_l.zip").write_bytes(
        _zip_bytes({"det_10g.onnx": b"only the detector"}, top_level="buffalo_l"))

    with pytest.raises(DownloadError, match="missing expected files"):
        ensure_face_model(cfg)


def test_checksum_mismatch_is_rejected(fake_model, monkeypatch):
    spec, cfg, _ = fake_model
    monkeypatch.setattr(spec, "sha256", hashlib.sha256(b"wrong").hexdigest())

    with pytest.raises(DownloadError, match="checksum"):
        ensure_face_model(cfg)


def test_unreachable_url_gives_an_actionable_error(fake_model, monkeypatch):
    spec, cfg, _ = fake_model
    monkeypatch.setattr(spec, "url", "http://127.0.0.1:9/nothing.zip")

    with pytest.raises(DownloadError, match="internet connection"):
        ensure_face_model(cfg)


def test_zip_slip_is_refused(fake_model, file_server):
    """A downloaded archive is untrusted input, even from a trusted URL."""
    spec, cfg, directory = fake_model
    (directory / "buffalo_l.zip").write_bytes(
        _zip_bytes({"../../escaped.onnx": b"pwned"}))

    with pytest.raises(DownloadError, match="escapes the destination"):
        ensure_face_model(cfg)


def test_unknown_model_name_lists_the_known_ones(fake_model, monkeypatch):
    _, cfg, _ = fake_model
    monkeypatch.setenv("PHOTO_FACE_MODEL", "buffalo_xxl")

    from photolib.config import get_settings, reset_settings_cache

    reset_settings_cache()
    try:
        with pytest.raises(DownloadError, match="buffalo_l"):
            ensure_face_model(get_settings())
    finally:
        reset_settings_cache()


# -- offline ---------------------------------------------------------------

def test_offline_build_refuses_to_download(fake_model, monkeypatch):
    _, cfg, _ = fake_model
    monkeypatch.setenv("PHOTO_OFFLINE", "1")

    assert offline() is True
    with pytest.raises(OfflineError, match="pinned offline"):
        ensure_face_model(cfg)


def test_offline_build_is_fine_once_installed(fake_model, monkeypatch):
    _, cfg, _ = fake_model
    ensure_face_model(cfg)
    monkeypatch.setenv("PHOTO_OFFLINE", "1")

    assert ensure_face_model(cfg).exists()


def test_enforce_offline_env_pins_the_ml_libraries(monkeypatch):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)

    enforce_offline_env()

    import os
    # transformers will otherwise contact the hub even with a local copy.
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_enforce_offline_env_does_not_override_an_explicit_setting(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    enforce_offline_env()

    import os
    assert os.environ["HF_HUB_OFFLINE"] == "0"


# -- API surface -----------------------------------------------------------

def test_models_endpoint_reports_readiness(client):
    body = client.get("/api/v1/admin/models").json()
    assert body["ready"] is True
    assert body["missing"] == []
