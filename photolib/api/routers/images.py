"""Image delivery: originals, thumbnails, metadata, and similarity."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from ...config import get_settings
from ...service import PhotoService
from ...thumbnails import SIZES
from ..deps import get_service, translate_errors

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])

MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/heic",
    ".heif": "image/heif", ".tif": "image/tiff", ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}


def _cached_file(path: Path, request: Request, media_type: str,
                 max_age: int, filename: Optional[str] = None) -> Response:
    """Serve a file with a strong ETag so repeat views are 304s.

    A photo grid re-requests the same thumbnails constantly as the user
    scrolls; without validators the browser re-downloads every one.
    """
    try:
        stat = path.stat()
    except OSError:
        raise HTTPException(status_code=404, detail="File not found")

    etag = '"{}"'.format(
        hashlib.sha1(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest())
    headers = {
        "ETag": etag,
        "Cache-Control": f"private, max-age={max_age}",
    }
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)

    return FileResponse(path, media_type=media_type, headers=headers,
                        filename=filename)


@router.get("/{image_id}")
def get_original(image_id: int, request: Request,
                 download: bool = Query(False),
                 service: PhotoService = Depends(get_service)):
    """The original file, straight off disk."""
    try:
        path = Path(service.image_path(image_id))
    except Exception as exc:
        raise translate_errors(exc)
    if not path.exists():
        raise HTTPException(
            status_code=410,
            detail=f"Indexed file is missing from disk: {path}")

    media_type = MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    return _cached_file(path, request, media_type,
                        get_settings().thumbnail_max_age,
                        filename=path.name if download else None)


@router.get("/{image_id}/thumbnail")
def get_thumbnail(image_id: int, request: Request,
                  size: str = Query("grid", enum=list(SIZES)),
                  format: str = Query("webp", enum=["webp", "jpeg"]),
                  service: PhotoService = Depends(get_service)):
    """A cached thumbnail, generated on first request."""
    try:
        source = service.image_path(image_id)
    except Exception as exc:
        raise translate_errors(exc)
    if not os.path.exists(source):
        raise HTTPException(status_code=410, detail="Original file is missing")

    try:
        path = service.thumbs.get_thumbnail(image_id, source, size, format)
    except Exception as exc:
        logger.warning("Thumbnail generation failed for %s: %s", source, exc)
        raise HTTPException(status_code=500, detail=f"Thumbnail failed: {exc}")

    media_type = "image/webp" if format == "webp" else "image/jpeg"
    return _cached_file(path, request, media_type, get_settings().thumbnail_max_age)


@router.get("/{image_id}/details")
def get_details(image_id: int, service: PhotoService = Depends(get_service)):
    """Full metadata, the people in the photo, and every detected face box."""
    try:
        return service.image_details(image_id)
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{image_id}/similar")
def get_similar(image_id: int, limit: int = Query(24, ge=1, le=200),
                service: PhotoService = Depends(get_service)):
    """Visually similar photos, using the stored image embedding."""
    try:
        return {"results": service.similar_images(image_id, limit)}
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{image_id}/faces")
def get_faces(image_id: int, service: PhotoService = Depends(get_service)):
    try:
        return {"faces": service.faces_in_image(image_id)}
    except Exception as exc:
        raise translate_errors(exc)
