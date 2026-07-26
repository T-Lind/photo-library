"""Albums: user-curated collections with similarity-powered suggestions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, Query

from ...service import PhotoService
from ..deps import get_service, translate_errors
from ..schemas import AlbumCreateRequest, AlbumItemsRequest, AlbumRenameRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/albums", tags=["albums"])


@router.get("")
def list_albums(service: PhotoService = Depends(get_service)):
    try:
        return {"albums": service.list_albums()}
    except Exception as exc:
        raise translate_errors(exc)


@router.post("")
def create_album(req: AlbumCreateRequest,
                 service: PhotoService = Depends(get_service)):
    try:
        return service.create_album(req.name)
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{album_id}")
def album_detail(album_id: int, limit: int = Query(500, ge=1, le=2000),
                 service: PhotoService = Depends(get_service)):
    """The album plus its photos, newest addition first."""
    try:
        return service.album_detail(album_id, limit)
    except Exception as exc:
        raise translate_errors(exc)


@router.patch("/{album_id}")
def rename_album(album_id: int, req: AlbumRenameRequest = Body(...),
                 service: PhotoService = Depends(get_service)):
    try:
        return service.rename_album(album_id, req.name)
    except Exception as exc:
        raise translate_errors(exc)


@router.delete("/{album_id}")
def delete_album(album_id: int, service: PhotoService = Depends(get_service)):
    """Delete the album. The photos in it are untouched."""
    try:
        return service.delete_album(album_id)
    except Exception as exc:
        raise translate_errors(exc)


@router.post("/{album_id}/items")
def add_items(album_id: int, req: AlbumItemsRequest,
              service: PhotoService = Depends(get_service)):
    try:
        return service.add_album_items(album_id, req.image_ids)
    except Exception as exc:
        raise translate_errors(exc)


@router.post("/{album_id}/items/remove")
def remove_items(album_id: int, req: AlbumItemsRequest,
                 service: PhotoService = Depends(get_service)):
    try:
        return service.remove_album_items(album_id, req.image_ids)
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{album_id}/suggestions")
def suggestions(album_id: int, limit: int = Query(24, ge=1, le=100),
                service: PhotoService = Depends(get_service)):
    """Photos that look like this album, for one-click adding."""
    try:
        return {"suggestions": service.album_suggestions(album_id, limit)}
    except Exception as exc:
        raise translate_errors(exc)
