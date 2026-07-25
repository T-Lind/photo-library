"""Library administration: indexing, reclustering, stats, maintenance."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from ...config import get_settings
from ...service import PhotoService
from ..deps import get_service, translate_errors
from ..schemas import IndexRequest, JobOut, ReclusterRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])


@router.get("/stats")
def stats(service: PhotoService = Depends(get_service)):
    """Library-wide counts, date coverage, and which models built the index."""
    try:
        return service.stats()
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/health")
def health(service: PhotoService = Depends(get_service)):
    """Liveness plus whether a library has been indexed yet.

    Deliberately does not touch the models — a health check must not pay for
    loading a couple of gigabytes of weights.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "ready": service.ready,
        "db_uri": settings.db_uri,
        "embed_backend": settings.embed_backend,
        "face_backend": settings.face_backend,
    }


@router.post("/admin/index", response_model=JobOut)
def start_index(req: IndexRequest, service: PhotoService = Depends(get_service)):
    """Index (or re-index) a folder in the background.

    Returns immediately with a job id; poll ``/admin/jobs/{id}`` for progress.
    Indexing is incremental, so pointing this at the same folder after adding
    photos only costs model time for the new files.
    """
    try:
        job = service.start_index_job(req.folder, req.rebuild, req.prune_missing)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise translate_errors(exc)
    return JobOut(**job.to_dict())


@router.post("/admin/recluster", response_model=JobOut)
def start_recluster(req: ReclusterRequest,
                    service: PhotoService = Depends(get_service)):
    """Rebuild every person from scratch from the stored face embeddings.

    Useful after tuning thresholds. Manually confirmed faces are preserved.
    """
    try:
        job = service.start_recluster_job(req.threshold, req.knn)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise translate_errors(exc)
    return JobOut(**job.to_dict())


@router.post("/admin/compact", response_model=JobOut)
def start_compact(service: PhotoService = Depends(get_service)):
    """Merge fragments and rebuild indexes. Worth running after a big import."""
    try:
        job = service.start_compact_job()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise translate_errors(exc)
    return JobOut(**job.to_dict())


@router.get("/admin/jobs", response_model=List[JobOut])
def list_jobs(service: PhotoService = Depends(get_service)):
    return [JobOut(**job.to_dict()) for job in service.jobs.list()]


@router.get("/admin/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, service: PhotoService = Depends(get_service)):
    job = service.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobOut(**job.to_dict())


@router.delete("/admin/jobs/{job_id}")
def cancel_job(job_id: str, service: PhotoService = Depends(get_service)):
    if not service.jobs.cancel(job_id):
        raise HTTPException(status_code=404,
                            detail="Job not found or already finished")
    return {"cancelled": job_id}


@router.get("/admin/duplicates")
def duplicates(max_distance: int = Query(6, ge=0, le=20),
               limit: int = Query(200, ge=1, le=1000),
               service: PhotoService = Depends(get_service)):
    """Groups of identical or near-identical photos, for reclaiming space."""
    try:
        return {"groups": service.duplicates(max_distance, limit)}
    except Exception as exc:
        raise translate_errors(exc)
