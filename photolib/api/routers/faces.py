"""Face-level endpoints: crops, search-by-face, and manual corrections."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import (APIRouter, Depends, File, HTTPException, Query, Request,
                     UploadFile)

from ...config import get_settings
from ...service import PhotoService
from ..deps import get_service, translate_errors
from ..schemas import AssignFacesRequest, DetachFacesRequest, FaceSearchRequest
from .images import _cached_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/faces", tags=["faces"])

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


@router.get("/unassigned")
def unassigned(limit: int = Query(120, ge=1, le=500),
               min_quality: float = Query(0.3, ge=0.0, le=1.0),
               service: PhotoService = Depends(get_service)):
    """Faces with no identity yet, best-quality first — the review queue."""
    try:
        return {"faces": service.unassigned_faces(limit, min_quality)}
    except Exception as exc:
        raise translate_errors(exc)


@router.post("/search")
def search_faces(req: FaceSearchRequest,
                 service: PhotoService = Depends(get_service)):
    """Find faces that look like a given face or person.

    This is the "search by face" path: it queries the face vector index
    directly, so it finds people the clusterer never linked, including in
    photos where they were never tagged.
    """
    try:
        if req.face_id is not None:
            return {"faces": service.search_faces_by_face(
                req.face_id, req.limit, req.min_similarity)}
        if req.person_id is not None:
            return {"faces": service.person_suggestions(req.person_id, req.limit)}
        raise ValueError("Provide either face_id or person_id")
    except Exception as exc:
        raise translate_errors(exc)


@router.post("/search/by-upload")
async def search_faces_by_upload(
    file: UploadFile = File(...),
    limit: int = Query(60, ge=1, le=300),
    min_similarity: float = Query(0.3, ge=-1.0, le=1.0),
    service: PhotoService = Depends(get_service),
):
    """"Who is this?" — upload a photo and match its largest face."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="photolib-face-"))
    tmp_path = tmp_dir / (Path(file.filename or "upload").name or "upload")
    try:
        written = 0
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload too large")
                out.write(chunk)
        return {"faces": service.search_faces_in_upload(
            tmp_path, limit, min_similarity)}
    except HTTPException:
        raise
    except Exception as exc:
        raise translate_errors(exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/assign")
def assign_faces(req: AssignFacesRequest,
                 service: PhotoService = Depends(get_service)):
    """Attach faces to a person, or create a new person from them.

    Assignments made here are marked confirmed, so a later recluster treats
    them as ground truth rather than undoing the correction.
    """
    try:
        return service.assign_faces(req.face_ids, req.person_id, req.name)
    except Exception as exc:
        raise translate_errors(exc)


@router.post("/detach")
def detach_faces(req: DetachFacesRequest,
                 service: PhotoService = Depends(get_service)):
    """"That isn't them" — return faces to the unassigned pool."""
    try:
        return service.detach_faces(req.face_ids)
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{face_id}")
def get_face(face_id: int, service: PhotoService = Depends(get_service)):
    try:
        return service.get_face(face_id)
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{face_id}/crop")
def get_face_crop(face_id: int, request: Request,
                  service: PhotoService = Depends(get_service)):
    """The cropped face, cut from the original photo and cached on first view."""
    try:
        face = service.get_face(face_id)
        source = service.image_path(face["image_id"])
    except Exception as exc:
        raise translate_errors(exc)

    if not Path(source).exists():
        raise HTTPException(status_code=410, detail="Original file is missing")

    try:
        path = service.thumbs.get_face_crop(face_id, source, tuple(face["bbox"]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Face crop failed: {exc}")

    return _cached_file(path, request, "image/jpeg", get_settings().thumbnail_max_age)
