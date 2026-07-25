"""Shared dependencies for the API routers."""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import HTTPException

from ..config import get_settings
from ..service import NotFound, PhotoService

_service: Optional[PhotoService] = None
_lock = threading.Lock()


def get_service() -> PhotoService:
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = PhotoService(get_settings())
    return _service


def set_service(service: Optional[PhotoService]) -> None:
    """Inject a service instance (used by tests)."""
    global _service
    with _lock:
        _service = service


def translate_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValueError, KeyError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))
