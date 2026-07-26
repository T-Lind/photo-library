"""Search endpoints: natural language, reverse image, and browsing."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from ...browse import Filters
from ...service import PhotoService
from ..deps import get_service, translate_errors
from ..schemas import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def _filters(req: SearchRequest) -> Filters:
    return Filters(
        start_date=req.start_date,
        end_date=req.end_date,
        people_ids=req.people_ids,
        people_mode=req.people_mode,
        has_location=req.has_location,
        has_faces=req.has_faces,
        folder=req.folder,
        camera=req.camera,
        untagged_only=req.untagged_only,
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, service: PhotoService = Depends(get_service)):
    """Combined semantic + person + date + location search.

    With no query this is a plain browse, which is why an empty search is
    cheap: it never touches the embedding model or the ANN index.
    """
    try:
        page = service.search(req.query, _filters(req), sort=req.sort,
                              page=req.page, per_page=req.per_page,
                              min_score=req.min_score)
    except Exception as exc:
        logger.exception("Search failed")
        raise translate_errors(exc)
    return SearchResponse(**page.__dict__)


@router.get("/search", response_model=SearchResponse)
def search_get(
    q: Optional[str] = Query(None, description="Natural-language query"),
    page: int = Query(1, ge=1),
    per_page: Optional[int] = Query(None, ge=1, le=1000),
    sort: str = Query("relevance"),
    people: Optional[str] = Query(None, description="Comma-separated person ids"),
    service: PhotoService = Depends(get_service),
):
    """GET form of /search, so a search is a shareable, bookmarkable URL."""
    people_ids = [int(p) for p in people.split(",") if p.strip()] if people else []
    req = SearchRequest(query=q, page=page, per_page=per_page,
                        sort=sort, people_ids=people_ids)  # type: ignore[arg-type]
    return search(req, service)


@router.post("/search/by-image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    page: int = Form(1),
    per_page: Optional[int] = Form(None),
    service: PhotoService = Depends(get_service),
):
    """Reverse image search: find photos that look like the uploaded one."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="photolib-upload-"))
    tmp_path = tmp_dir / (Path(file.filename or "upload").name or "upload")
    try:
        written = 0
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload too large")
                out.write(chunk)

        result = service.search_by_image(tmp_path, Filters(), page=page,
                                         per_page=per_page)
        return SearchResponse(**result.__dict__)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Reverse image search failed")
        raise translate_errors(exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/timeline")
def timeline(service: PhotoService = Depends(get_service)):
    """Photos per month — drives the timeline scrubber."""
    try:
        return {"months": service.timeline()}
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/folders")
def folders(service: PhotoService = Depends(get_service)):
    try:
        return {"folders": service.folders()}
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/cameras")
def cameras(service: PhotoService = Depends(get_service)):
    """Distinct camera models with photo counts — the camera filter options."""
    try:
        return {"cameras": service.cameras()}
    except Exception as exc:
        raise translate_errors(exc)
