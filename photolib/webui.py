"""Serve the built web UI from the API process.

Running a Python API and a Node server side by side is fine for development
and a non-starter for handing the app to a family member. The UI is exported
as static files (``next build`` with ``output: 'export'``), which means the
API can serve it directly: one process, one port, no Node.js at runtime, and
nothing for anyone to start in the right order.

Because the UI and API then share an origin, the browser makes same-origin
requests to ``/api/v1`` and CORS stops being involved at all.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Where a built UI may live, in priority order.
CANDIDATES = (
    "PHOTO_WEB_DIR",           # explicit override
)
BUNDLED_DIRNAME = "web"


def find_web_dir(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the exported UI, or None if this is an API-only deployment."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if (path / "index.html").exists() else None

    env = os.environ.get("PHOTO_WEB_DIR")
    if env:
        path = Path(env).expanduser()
        if (path / "index.html").exists():
            return path
        logger.warning("PHOTO_WEB_DIR=%s has no index.html; ignoring", env)

    roots = [Path(__file__).resolve().parent.parent]
    # PyInstaller unpacks bundled data to _MEIPASS at runtime.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.insert(0, Path(meipass))

    for root in roots:
        candidate = root / BUNDLED_DIRNAME
        if (candidate / "index.html").exists():
            return candidate
    return None


class SpaStaticFiles(StaticFiles):
    """Static files with the fallbacks a statically-exported Next app needs.

    ``next build --output export`` writes ``people.html`` rather than
    ``people/index.html`` for a route like ``/people``, and client-side
    routes such as ``/person?id=3`` have no file at all. Both need to resolve
    to real HTML instead of a 404.
    """

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
            # With html=True, StaticFiles *returns* its own 404.html rather
            # than raising, so a missing file has to be detected by status.
            if response.status_code != 404:
                return response
        except Exception:
            response = None

        root = Path(self.directory)

        # /people -> people.html
        candidate = (root / f"{path.strip('/')}.html") if path.strip("/") else None
        if candidate is not None and candidate.is_file() and _within(candidate, root):
            return FileResponse(candidate, media_type="text/html")

        # A client-side route with no file of its own falls back to the
        # shell; a missing asset (anything with an extension) stays a 404.
        if "." not in Path(path).name:
            index = root / "index.html"
            if index.is_file():
                return FileResponse(index, media_type="text/html")

        return response if response is not None else Response(status_code=404)


def _within(path: Path, root: Path) -> bool:
    """Guard the ``.html`` fallback against traversal out of the web root."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def mount_web_ui(app: FastAPI, api_prefix: str,
                 explicit_dir: Optional[str] = None) -> Optional[Path]:
    """Mount the UI at ``/`` if a build is present. Returns the directory used.

    Mounting at ``/`` must happen after the API routers are registered, so
    that ``/api/v1/...`` still resolves to the API rather than being
    swallowed by the static handler.
    """
    web_dir = find_web_dir(explicit_dir)
    if web_dir is None:
        logger.info("No built web UI found; serving the API only")
        return None

    app.mount("/", SpaStaticFiles(directory=str(web_dir), html=True), name="web")
    logger.info("Serving the web UI from %s", web_dir)
    return web_dir
