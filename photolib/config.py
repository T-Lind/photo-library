"""Runtime configuration.

Every setting is overridable with a ``PHOTO_``-prefixed environment
variable, so a deployment never needs to edit code. Defaults are chosen to
work on a laptop with no GPU and no configuration at all.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PHOTO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- storage -------------------------------------------------------
    db_uri: str = Field("data/library", description="LanceDB database location")
    faces_dir: str = Field("data/faces", description="Cropped face images")
    thumbnail_cache_dir: str = Field("data/thumbnails", description="Thumbnail cache")
    state_dir: str = Field("data/state", description="Job state and scratch files")

    # ---- embedding model ----------------------------------------------
    # SigLIP 2 beats OpenAI CLIP on essentially every zero-shot retrieval
    # benchmark at a comparable size, and it is Apache-2.0. "stub" is a
    # deterministic hash embedder used by the test suite so CI needs no
    # model weights.
    embed_backend: Literal["siglip", "clip", "open_clip", "stub"] = "siglip"
    embed_model: str = "google/siglip2-base-patch16-224"
    embed_batch_size: int = Field(16, ge=1, le=512)
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    # float16 on GPU roughly halves memory and speeds up indexing; ignored on CPU.
    embed_fp16: bool = True

    # ---- face model ----------------------------------------------------
    # InsightFace buffalo_l = RetinaFace detection + ArcFace w600k_r50
    # recognition. Substantially more accurate than dlib/face_recognition,
    # especially on non-frontal faces, children, and low light.
    face_backend: Literal["insightface", "dlib", "stub", "none"] = "insightface"
    face_model: str = "buffalo_l"
    face_det_size: int = Field(640, ge=160, le=1600)
    face_min_det_score: float = Field(0.5, ge=0.0, le=1.0)
    # Faces smaller than this (in pixels, longest side) are detected but not
    # used for identity — they are too low-resolution to embed reliably.
    face_min_size: int = Field(40, ge=0)

    # Cosine-similarity thresholds on normalised ArcFace embeddings.
    # 0.0 = unrelated, 1.0 = identical.
    face_match_threshold: float = Field(
        0.38, ge=0.0, le=1.0,
        description="Minimum similarity for a face to join an existing person",
    )
    face_strong_match_threshold: float = Field(
        0.55, ge=0.0, le=1.0,
        description="Similarity above which a single neighbour is enough",
    )
    face_cluster_threshold: float = Field(
        0.42, ge=0.0, le=1.0,
        description="Similarity threshold for offline mutual-kNN reclustering",
    )
    face_cluster_knn: int = Field(20, ge=2, le=200)
    face_min_cluster_size: int = Field(2, ge=1)

    # ---- search --------------------------------------------------------
    # ANN search knobs. Higher = more accurate, slower.
    nprobes: int = Field(24, ge=1)
    refine_factor: int = Field(8, ge=1)
    # Semantic search ranks the whole library; we materialise at most this
    # many ranked candidates and paginate inside them.
    max_candidates: int = Field(1000, ge=50, le=20000)
    default_page_size: int = Field(60, ge=1, le=500)
    max_page_size: int = Field(200, ge=1, le=1000)

    # ---- indexing ------------------------------------------------------
    ingest_workers: int = Field(0, ge=0, description="0 = os.cpu_count()")
    write_batch_size: int = Field(256, ge=1)
    pregenerate_thumbnails: bool = True
    ann_min_rows: int = Field(4096, ge=0, description="Skip ANN index below this")
    follow_symlinks: bool = False

    # ---- server --------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    # NoDecode stops pydantic-settings from trying to JSON-parse the env var,
    # so PHOTO_CORS_ORIGINS can be the natural comma-separated list.
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"])
    # Serving originals means the API can read any file it indexed. Keep the
    # allowlist tight if the API is exposed beyond localhost.
    thumbnail_max_age: int = Field(604800, ge=0, description="Cache-Control seconds")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ---- derived -------------------------------------------------------
    @property
    def worker_count(self) -> int:
        return self.ingest_workers or (os.cpu_count() or 4)

    def ensure_dirs(self) -> None:
        for p in (self.db_uri, self.faces_dir, self.thumbnail_cache_dir, self.state_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests that mutate the environment)."""
    get_settings.cache_clear()
