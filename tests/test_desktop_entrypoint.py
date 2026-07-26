"""Exercise the desktop file the same way PyInstaller executes it."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_desktop_entrypoint_works_without_package_context(tmp_path, monkeypatch):
    for key in (
        "PHOTO_DB_URI",
        "PHOTO_THUMBNAIL_CACHE_DIR",
        "PHOTO_STATE_DIR",
        "PHOTO_FACES_DIR",
        "PHOTO_FACE_MODEL_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)

    root = Path(__file__).resolve().parent.parent
    namespace = runpy.run_path(
        str(root / "photolib" / "desktop.py"),
        run_name="pyinstaller_entrypoint",
    )

    data = namespace["configure_environment"](tmp_path / "appdata")
    assert data == tmp_path / "appdata"
    assert (data / "logs").is_dir()
