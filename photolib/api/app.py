"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Settings, get_settings
from .deps import get_service
from .routers import admin, faces, images, people, search

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"

DESCRIPTION = """
A local, private photo library. Natural-language search over your photos,
face search and recognition, and date/location browsing — all running on your
own machine with no network calls.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Warm the browse index if a library already exists, so the first
        # request isn't the one that pays for it. Models stay lazy: starting
        # the server should not load a gigabyte of weights.
        try:
            service = get_service()
            if service.ready:
                service.index.ensure_fresh()
                logger.info("Library ready: %d images", service.index.count)
            else:
                logger.info("No library indexed yet at %s", settings.db_uri)
        except Exception as exc:
            logger.warning("Startup warm-up skipped: %s", exc)
        yield

    app = FastAPI(
        title="photolib",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # The UI reads pagination totals from the body, but exposing these
        # lets a client see timing without parsing JSON.
        expose_headers=["X-Response-Time"],
    )
    # JSON search responses compress well; images are already compressed and
    # are skipped by the minimum-size threshold.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Response-Time"] = \
            f"{(time.perf_counter() - started) * 1000:.1f}ms"
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # pragma: no cover
        # Log the traceback locally; return a generic message so an internal
        # path never leaks into a response body.
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error"})

    for router in (search.router, images.router, people.router,
                   faces.router, admin.router):
        app.include_router(router, prefix=API_PREFIX)

    @app.get("/")
    def root():
        return {
            "name": "photolib",
            "version": __version__,
            "docs": "/docs",
            "api": API_PREFIX,
        }

    return app


app = create_app()
