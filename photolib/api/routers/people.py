"""Person management: naming, merging, hiding, and review suggestions."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Body, Depends, Query

from ...service import PhotoService
from ..deps import get_service, translate_errors
from ..schemas import (HidePersonRequest, MergePeopleRequest, PersonOut,
                       RenamePersonRequest)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/people", tags=["people"])


@router.get("", response_model=List[PersonOut])
def list_people(include_hidden: bool = Query(False),
                named_only: bool = Query(False),
                min_photos: int = Query(1, ge=0),
                service: PhotoService = Depends(get_service)):
    """Everyone the library knows about, most-photographed first.

    ``min_photos`` is the useful knob on a real library: auto-clustering will
    always produce a long tail of strangers who appear in exactly one photo,
    and hiding them makes the page usable.
    """
    try:
        return service.list_people(include_hidden, named_only, min_photos)
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{person_id}", response_model=PersonOut)
def get_person(person_id: int, service: PhotoService = Depends(get_service)):
    try:
        return service.get_person(person_id)
    except Exception as exc:
        raise translate_errors(exc)


@router.patch("/{person_id}", response_model=PersonOut)
def rename_person(person_id: int, req: RenamePersonRequest = Body(...),
                  service: PhotoService = Depends(get_service)):
    try:
        return service.rename_person(person_id, req.name.strip())
    except Exception as exc:
        raise translate_errors(exc)


@router.post("/{person_id}/hidden", response_model=PersonOut)
def set_hidden(person_id: int, req: HidePersonRequest = Body(...),
               service: PhotoService = Depends(get_service)):
    """Hide a person without deleting them — for strangers and false clusters."""
    try:
        return service.set_person_hidden(person_id, req.hidden)
    except Exception as exc:
        raise translate_errors(exc)


@router.post("/merge", response_model=PersonOut)
def merge_people(req: MergePeopleRequest, service: PhotoService = Depends(get_service)):
    """Fold one identity into another when clustering split the same person."""
    try:
        return service.merge_people(req.source_id, req.target_id)
    except Exception as exc:
        raise translate_errors(exc)


@router.delete("/{person_id}")
def delete_person(person_id: int, service: PhotoService = Depends(get_service)):
    """Forget an identity. The faces stay, returned to the unassigned pool."""
    try:
        return service.delete_person(person_id)
    except Exception as exc:
        raise translate_errors(exc)


@router.get("/{person_id}/suggestions")
def suggestions(person_id: int, limit: int = Query(60, ge=1, le=300),
                service: PhotoService = Depends(get_service)):
    """Unassigned faces that look like this person, for one-click confirmation."""
    try:
        return {"suggestions": service.person_suggestions(person_id, limit)}
    except Exception as exc:
        raise translate_errors(exc)
