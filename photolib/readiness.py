"""Reachability checks used by the native desktop launch handshake."""

from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

logger = logging.getLogger(__name__)


def wait_for_server(url: str, attempts: int = 600, interval: float = 0.05) -> bool:
    """Return only after the loopback API has accepted and answered a request."""
    health_url = f"{url.rstrip('/')}/api/v1/health"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(interval)
    return False


def run_when_ready(url: str, callback: Callable[[], None]) -> threading.Thread:
    """Run ``callback`` in a daemon thread once the server is truly reachable."""
    def probe() -> None:
        if wait_for_server(url):
            callback()
        else:
            logger.error("Server did not become reachable at %s", url)

    thread = threading.Thread(target=probe, name="desktop-readiness", daemon=True)
    thread.start()
    return thread
