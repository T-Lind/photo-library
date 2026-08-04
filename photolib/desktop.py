"""Entry point for the packaged desktop application.

This is what PyInstaller freezes and what the Tauri shell launches. It has
one job: bring up the server with settings that suit a desktop install, and
tell the caller where it is listening.

Differences from ``photolib serve``:

* Data lives in a per-user application directory, not the working directory,
  because a frozen binary's working directory is wherever the user's shortcut
  happened to point.
* It binds to a free port on loopback only, and prints that port on stdout in
  a machine-readable form so the shell knows what to load. Hardcoding 8000
  fails the moment anything else on the machine is already using it.
* Models resolve to files bundled next to the executable.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

APP_NAME = "photolib"
# Printed on stdout once the server is accepting connections. The Tauri shell
# waits for this line rather than polling a guessed port.
READY_PREFIX = "PHOTOLIB_READY "


def user_data_dir(app: str = APP_NAME) -> Path:
    """Per-user writable directory, following each platform's convention."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / app


def bundle_dir() -> Path:
    """Directory holding bundled read-only assets (models, web UI).

    Under PyInstaller this is the unpacked ``_MEIPASS`` temp directory; from
    a source checkout it is the repository root.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def configure_environment(data_dir: Optional[Path] = None) -> Path:
    """Point every configurable path at the per-user data directory.

    Uses ``setdefault`` throughout so an advanced user can still override
    any of it from the environment.
    """
    data = Path(data_dir) if data_dir else user_data_dir()
    bundle = bundle_dir()

    (data / "logs").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PHOTO_DB_URI", str(data / "library"))
    os.environ.setdefault("PHOTO_THUMBNAIL_CACHE_DIR", str(data / "thumbnails"))
    os.environ.setdefault("PHOTO_STATE_DIR", str(data / "state"))
    os.environ.setdefault("PHOTO_FACES_DIR", str(data / "faces"))
    # Downloaded on first run; belongs with the user's data, not the
    # read-only bundle, which may be under Program Files.
    os.environ.setdefault("PHOTO_FACE_MODEL_ROOT", str(data / "models" / "insightface"))

    # The image/text model ships with the app and needs no network at all.
    bundled_model = bundle / "models" / "siglip2-base"
    if (bundled_model / "preprocess.json").exists():
        os.environ.setdefault("PHOTO_EMBED_BACKEND", "onnx")
        os.environ.setdefault("PHOTO_ONNX_MODEL_DIR", str(bundled_model))
        # Standard installers contain only the smaller INT8 graphs. Source
        # checkouts and maximum-quality builds may contain FP32 instead.
        if (bundled_model / "text.int8.onnx").exists() and not (
                bundled_model / "text.onnx").exists():
            os.environ.setdefault("PHOTO_ONNX_INT8", "1")

    web = bundle / "web"
    if (web / "index.html").exists():
        os.environ.setdefault("PHOTO_WEB_DIR", str(web))

    # Nothing may reach the network except the explicit, user-initiated
    # model download in photolib.models.
    from photolib.models import enforce_offline_env

    enforce_offline_env()

    return data


def setup_logging(data_dir: Path) -> None:
    """Log to a file — a windowed app has nowhere useful to print."""
    handlers: list[logging.Handler] = [
        logging.FileHandler(data_dir / "logs" / "photolib.log", encoding="utf-8")
    ]
    if not frozen() or os.environ.get("PHOTO_CONSOLE_LOG"):
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def watch_stdin_for_shutdown(server, stream=None) -> threading.Thread:
    """Ask uvicorn to drain active requests when the shell closes."""
    stream = stream or sys.stdin

    def watch() -> None:
        try:
            for line in stream:
                if line.strip() == "PHOTOLIB_SHUTDOWN":
                    logger.info("Desktop shell requested a graceful shutdown")
                    server.should_exit = True
                    return
        except (OSError, ValueError):
            # stdin can disappear abruptly during OS shutdown. The desktop
            # shell retains a timed hard-kill fallback for that case.
            logger.debug("Desktop shutdown channel closed", exc_info=True)

    thread = threading.Thread(
        target=watch, name="photolib-shutdown", daemon=True)
    thread.start()
    return thread

def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="photolib-desktop")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--verify-model", action="store_true",
                        help="Verify bundled ONNX preprocessing and exit")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open a browser (the desktop shell does it)")
    args = parser.parse_args(argv)

    data = configure_environment(args.data_dir)
    setup_logging(data)

    import uvicorn

    from photolib.config import get_settings

    port = args.port or free_port()
    settings = get_settings()
    settings.ensure_dirs()

    if args.verify_model:
        from photolib.embeddings.onnx_vision import OnnxVisionEmbedder

        report = OnnxVisionEmbedder(
            settings.onnx_model_dir,
            prefer_int8=settings.onnx_int8,
        ).self_check()
        try:
            from insightface.app import FaceAnalysis  # noqa: F401
            report["face_runtime"] = True
        except Exception as exc:
            report["face_runtime"] = False
            report["face_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(report, indent=2), flush=True)
        return 0 if (report.get("checked") and report.get("ok")
                     and report.get("face_runtime")) else 1

    logger.info("photolib desktop starting: data=%s bundle=%s port=%d",
                data, bundle_dir(), port)

    from photolib.api.app import create_app
    from photolib.readiness import run_when_ready

    url = f"http://{args.host}:{port}"

    def announce() -> None:
        # stdout is the handshake with the desktop shell; flush so the reader
        # is not left waiting on a buffer.
        print(f"{READY_PREFIX}{json.dumps({'url': url, 'data_dir': str(data)})}",
              flush=True)
        if not args.no_browser:
            import webbrowser

            webbrowser.open(url)

    run_when_ready(url, announce)
    config = uvicorn.Config(
        create_app(settings), host=args.host, port=port,
        log_level="warning", access_log=False,
    )
    server = uvicorn.Server(config)
    watch_stdin_for_shutdown(server)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
