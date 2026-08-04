"""The packaged desktop entry point.

The Tauri shell depends on two things being exactly right: the server
announcing its port on stdout, and every writable path landing in the user's
data directory rather than next to a read-only executable. Both are easy to
break silently, so both are tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from photolib.folder_picker import FolderChoice, choose_photo_folder

from photolib.desktop import (READY_PREFIX, configure_environment, free_port,
                              user_data_dir)


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("PHOTO_DB_URI", "PHOTO_THUMBNAIL_CACHE_DIR", "PHOTO_STATE_DIR",
                "PHOTO_FACES_DIR", "PHOTO_FACE_MODEL_ROOT", "PHOTO_WEB_DIR",
                "PHOTO_EMBED_BACKEND", "PHOTO_ONNX_MODEL_DIR",
                "PHOTO_ONNX_INT8",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    from photolib.config import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


def test_all_writable_paths_go_to_the_data_directory(tmp_path, clean_env):
    import os

    data = configure_environment(tmp_path / "appdata")

    assert data == tmp_path / "appdata"
    for key in ("PHOTO_DB_URI", "PHOTO_THUMBNAIL_CACHE_DIR", "PHOTO_STATE_DIR",
                "PHOTO_FACES_DIR", "PHOTO_FACE_MODEL_ROOT"):
        # A frozen binary may live under Program Files; nothing may try to
        # write next to it.
        assert Path(os.environ[key]).is_relative_to(data), key


def test_the_log_directory_is_created(tmp_path, clean_env):
    data = configure_environment(tmp_path / "appdata")
    assert (data / "logs").is_dir()


def test_network_access_is_pinned_off(tmp_path, clean_env):
    import os

    configure_environment(tmp_path / "appdata")

    # transformers otherwise contacts the model hub even with local weights.
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_explicit_environment_is_never_overridden(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("PHOTO_DB_URI", "/somewhere/else")
    configure_environment(tmp_path / "appdata")

    import os

    assert os.environ["PHOTO_DB_URI"] == "/somewhere/else"


def test_a_bundled_model_selects_the_onnx_backend(tmp_path, clean_env, monkeypatch):
    import os

    bundle = tmp_path / "bundle"
    model = bundle / "models" / "siglip2-base"
    model.mkdir(parents=True)
    (model / "preprocess.json").write_text("{}")
    monkeypatch.setattr("photolib.desktop.bundle_dir", lambda: bundle)

    configure_environment(tmp_path / "appdata")

    assert os.environ["PHOTO_EMBED_BACKEND"] == "onnx"
    assert os.environ["PHOTO_ONNX_MODEL_DIR"] == str(model)
def test_an_int8_only_bundle_selects_quantized_graphs(tmp_path, clean_env,
                                                       monkeypatch):
    import os

    bundle = tmp_path / "bundle"
    model = bundle / "models" / "siglip2-base"
    model.mkdir(parents=True)
    (model / "preprocess.json").write_text("{}")
    (model / "text.int8.onnx").write_bytes(b"int8")
    (model / "vision.int8.onnx").write_bytes(b"int8")
    monkeypatch.setattr("photolib.desktop.bundle_dir", lambda: bundle)

    configure_environment(tmp_path / "appdata")

    assert os.environ["PHOTO_ONNX_INT8"] == "1"




def test_no_bundled_model_leaves_the_backend_alone(tmp_path, clean_env, monkeypatch):
    import os

    monkeypatch.setattr("photolib.desktop.bundle_dir", lambda: tmp_path / "empty")
    configure_environment(tmp_path / "appdata")

    assert "PHOTO_EMBED_BACKEND" not in os.environ


def test_free_port_returns_a_usable_port():
    import socket

    port = free_port()
    assert 1024 < port < 65536
    # It must actually be bindable — that is the whole point of asking.
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_user_data_dir_is_platform_appropriate():
    path = user_data_dir("photolib-test")
    assert path.name == "photolib-test"
    assert path.is_absolute()
def test_shutdown_channel_requests_uvicorn_exit():
    from io import StringIO

    from photolib.desktop import watch_stdin_for_shutdown

    class Server:
        should_exit = False

    server = Server()
    thread = watch_stdin_for_shutdown(
        server, StringIO("ignored\nPHOTOLIB_SHUTDOWN\n"))
    thread.join(timeout=1)

    assert server.should_exit is True




def test_ready_callback_fires_once_on_startup(settings):
    """The desktop shell blocks until this line appears; if it never fires,
    the app hangs on a blank screen with no diagnosis."""
    from fastapi.testclient import TestClient

    from photolib.api.app import create_app
    from photolib.api.deps import set_service

    calls = []
    set_service(None)
    try:
        app = create_app(settings, on_ready=lambda: calls.append(1))
        assert calls == [], "must not fire before the server starts"
        with TestClient(app) as client:
            assert client.get("/api/v1/health").status_code == 200
        assert calls == [1]
    finally:
        set_service(None)


def test_a_failing_ready_callback_does_not_stop_the_server(settings):
    from fastapi.testclient import TestClient

    from photolib.api.app import create_app
    from photolib.api.deps import set_service

    def boom():
        raise RuntimeError("no stdout")

    set_service(None)
    try:
        with TestClient(create_app(settings, on_ready=boom)) as client:
            assert client.get("/api/v1/health").status_code == 200
    finally:
        set_service(None)


def test_ready_line_is_machine_readable():
    payload = json.dumps({"url": "http://127.0.0.1:1234", "data_dir": "/tmp/x"})
    line = f"{READY_PREFIX}{payload}"

    assert line.startswith("PHOTOLIB_READY ")
    parsed = json.loads(line[len(READY_PREFIX):])
    assert parsed["url"] == "http://127.0.0.1:1234"


def test_module_level_app_is_built_lazily_and_once():
    """Importing the module must not construct an application.

    It used to, which meant the desktop launcher built two apps and mounted
    the web UI twice.
    """
    import photolib.api.app as module

    module._app = None
    assert module._app is None

    first = module.app
    assert module._app is first
    assert module.app is first


def test_folder_picker_returns_an_absolute_existing_directory(tmp_path, monkeypatch):
    class Result:
        returncode = 0
        stdout = str(tmp_path)
        stderr = ""

    monkeypatch.setattr("photolib.folder_picker.sys.platform", "win32")
    monkeypatch.setattr("photolib.folder_picker.shutil.which",
                        lambda name: "powershell.exe")
    monkeypatch.setattr("photolib.folder_picker.subprocess.run",
                        lambda *args, **kwargs: Result())

    choice = choose_photo_folder()
    assert choice == FolderChoice(path=str(tmp_path.resolve()))


def test_folder_picker_cancellation_is_not_an_error(monkeypatch):
    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("photolib.folder_picker.sys.platform", "win32")
    monkeypatch.setattr("photolib.folder_picker.shutil.which",
                        lambda name: "powershell.exe")
    monkeypatch.setattr("photolib.folder_picker.subprocess.run",
                        lambda *args, **kwargs: Result())

    assert choose_photo_folder().cancelled is True
