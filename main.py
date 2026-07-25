"""Compatibility entry point: ``uvicorn main:app``.

The application now lives in :mod:`photolib.api.app`. This module exists so
that existing deployment commands and Dockerfiles keep working.
"""

from photolib.api.app import app  # noqa: F401

__all__ = ["app"]
