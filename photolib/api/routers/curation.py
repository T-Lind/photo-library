"""User curation and library maintenance endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from ..deps import get_service, translate_errors
from ..schemas import SearchRequest
from ...service import PhotoService
from ... import catalog

router = APIRouter(tags=["curation"])


class Annotation(BaseModel):
    favorite: Optional[bool] = None
    rating: Optional[int] = Field(None, ge=0, le=5)


class SavedSearch(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    request: SearchRequest


class Relocate(BaseModel):
    old_folder: str
    new_folder: str
    verification: Optional[str] = None


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise translate_errors(exc)


@router.patch("/images/{image_id}/annotation")
def annotate(image_id: int, req: Annotation, service: PhotoService = Depends(get_service)):
    return call(service.annotate, image_id, **req.model_dump())


@router.get("/images/{image_id}/burst")
def burst(image_id: int, service: PhotoService = Depends(get_service)):
    return call(service.burst_candidates, image_id)


@router.get("/saved-searches")
def searches(service: PhotoService = Depends(get_service)):
    return {"searches": call(service.saved_searches)}


@router.post("/saved-searches")
def save_search(req: SavedSearch, service: PhotoService = Depends(get_service)):
    return call(service.save_search, req.name, req.request.model_dump(mode="json"))


@router.delete("/saved-searches/{search_id}")
def delete_search(search_id: str, service: PhotoService = Depends(get_service)):
    call(catalog.delete, service.library, f"search:{search_id}")
    return {"deleted": search_id}


@router.post("/admin/roots/relocate")
def relocate(req: Relocate, service: PhotoService = Depends(get_service)):
    return call(service.relocate_root, **req.model_dump())


@router.get("/admin/failures")
def failures(service: PhotoService = Depends(get_service)):
    return {"files": call(service.failures)}


@router.post("/admin/retry")
def retry(service: PhotoService = Depends(get_service)):
    return call(service.start_retry_job).to_dict()


@router.post("/admin/quality")
def scan_quality(service: PhotoService = Depends(get_service)):
    return call(service.start_quality_job).to_dict()


@router.post("/admin/curation/verify")
def verify_backup(data: dict, service: PhotoService = Depends(get_service)):
    from ...backup import preview
    return call(preview, service, data)
