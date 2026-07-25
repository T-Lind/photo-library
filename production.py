"""Production server: gunicorn with uvicorn workers.

Worker count defaults to ``min(4, cpu_count)`` rather than ``2 * cpu + 1``.
Each worker holds its own copy of the embedding model and the browse index,
so oversubscribing multiplies memory use for no throughput gain — the work
here is model inference, not blocking IO.
"""

from __future__ import annotations

import multiprocessing
import os

import gunicorn.app.base

from photolib.api.app import app
from photolib.config import get_settings


class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, application, options=None):
        self.options = options or {}
        self.application = application
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()

    workers = int(os.environ.get("PHOTO_WORKERS", 0)) or max(
        1, min(4, multiprocessing.cpu_count()))

    StandaloneApplication(app, {
        "bind": f"0.0.0.0:{settings.port}",
        "workers": workers,
        "worker_class": "uvicorn.workers.UvicornWorker",
        # Indexing requests return immediately (they enqueue a job), but a
        # first-time thumbnail of a 60MP RAW can still take a while.
        "timeout": 300,
        "graceful_timeout": 30,
        "keepalive": 5,
        "errorlog": "-",
        "accesslog": "-",
        "access_log_format": '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)ss',
    }).run()


if __name__ == "__main__":
    main()
