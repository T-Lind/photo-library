"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SortOption = Literal["relevance", "date_desc", "date_asc", "added_desc", "random"]
PeopleMode = Literal["any", "all"]


class SearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Natural-language description")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    people_ids: List[int] = Field(default_factory=list)
    people_mode: PeopleMode = "any"
    has_location: Optional[bool] = None
    has_faces: Optional[bool] = None
    folder: Optional[str] = None
    camera: Optional[str] = None
    untagged_only: bool = False
    near_lat: Optional[float] = Field(None, ge=-90, le=90)
    near_lon: Optional[float] = Field(None, ge=-180, le=180)
    near_km: float = Field(1.0, gt=0, le=20000)
    sort: SortOption = "relevance"
    page: int = Field(1, ge=1)
    per_page: Optional[int] = Field(None, ge=1, le=1000)
    min_score: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Drop semantic matches below this cosine similarity")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _accept_z_suffix(cls, v):
        if isinstance(v, str) and v.endswith("Z"):
            return v[:-1] + "+00:00"
        return v


class ImageSummary(BaseModel):
    image_id: int
    filename: str = ""
    taken_at: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    place: str = ""
    people_ids: List[int] = Field(default_factory=list)
    face_count: int = 0
    width: int = 0
    height: int = 0
    score: Optional[float] = None


class SearchResponse(BaseModel):
    total: int
    page: int
    per_page: int
    took_ms: float = 0.0
    scored: bool = False
    results: List[ImageSummary]


class FaceOut(BaseModel):
    face_id: int
    image_id: int
    person_id: Optional[int] = None
    bbox: List[int]
    quality: Optional[float] = None
    det_score: Optional[float] = None
    confirmed: Optional[bool] = None
    similarity: Optional[float] = None


class PersonOut(BaseModel):
    person_id: int
    name: str = ""
    photo_count: int = 0
    face_count: int = 0
    cover_face_id: int = -1
    hidden: bool = False


class RenamePersonRequest(BaseModel):
    name: str = Field(..., max_length=200)


class HidePersonRequest(BaseModel):
    hidden: bool = True


class MergePeopleRequest(BaseModel):
    source_id: int
    target_id: int


class AssignFacesRequest(BaseModel):
    face_ids: List[int] = Field(..., min_length=1)
    person_id: Optional[int] = Field(
        None, description="Omit to create a new person from these faces")
    name: Optional[str] = Field(None, description="Name for a newly created person")


class DetachFacesRequest(BaseModel):
    face_ids: List[int] = Field(..., min_length=1)


class FaceSearchRequest(BaseModel):
    face_id: Optional[int] = None
    person_id: Optional[int] = None
    limit: int = Field(60, ge=1, le=500)
    min_similarity: float = Field(0.3, ge=-1.0, le=1.0)


class IndexRequest(BaseModel):
    folder: str = Field(..., description="Absolute path to a folder of photos")
    rebuild: bool = Field(False, description="Drop the library and start over")
    prune_missing: bool = Field(
        False, description="Remove indexed photos whose files are gone")


class RootRequest(BaseModel):
    folder: str = Field(..., description="Absolute path to a photo folder")


class ReclusterRequest(BaseModel):
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    knn: Optional[int] = Field(None, ge=2, le=200)


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    phase: str = ""
    current: int = 0
    total: int = 0
    percent: float = 0.0
    elapsed: float = 0.0
    detail: dict = Field(default_factory=dict)
    result: Optional[dict] = None
    error: Optional[str] = None
