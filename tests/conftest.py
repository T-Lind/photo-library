"""Shared fixtures.

The suite runs against a real LanceDB database with synthetic photos, using
the stub embedding and face backends. That keeps CI free of model downloads
while still exercising the real database, indexer, clustering, and HTTP
layers rather than mocks of them.

Synthetic photos carry their content in two places:

* the **filename**, whose words the stub embedder turns into an embedding, so
  ``beach-sunset-01.jpg`` genuinely ranks first for "sunset at the beach";
* a row of **coloured marker blocks** along the top edge, which the stub face
  backend reads as faces — one distinct colour per person.

Capture dates come from the filename too, via the same ``YYYYMMDD`` fallback
that rescues real photos stripped of EXIF by messaging apps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pytest

PERSON_COLOURS = {
    "alice": (220, 20, 60),
    "bob": (30, 144, 255),
    "carol": (34, 177, 76),
    "dave": (255, 165, 0),
}


def make_photo(path: Path, people: Sequence[str] = (),
               size: Tuple[int, int] = (240, 180),
               tint: Tuple[int, int, int] = (200, 200, 200)) -> Path:
    """Write a synthetic JPEG with face markers for ``people``."""
    from PIL import Image

    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[:, :] = tint
    # Give the body some texture so the perceptual hash isn't degenerate.
    array[16:, ::7] = np.clip(np.array(tint, dtype=np.int16) + 40, 0, 255)
    # Top strip: white background, one 16px block per person.
    array[:16, :] = 255
    for i, person in enumerate(people):
        array[:16, i * 16:(i + 1) * 16] = PERSON_COLOURS[person]

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, "JPEG", quality=95)
    return path


@pytest.fixture
def photo_dir(tmp_path: Path) -> Path:
    """A small library: 8 photos, 4 people, nested folders, known dates."""
    root = tmp_path / "photos"
    make_photo(root / "beach-sunset-holiday-20180704.jpg", ["alice"], tint=(255, 200, 120))
    make_photo(root / "beach-sunset-holiday-20180705.jpg", ["alice", "bob"], tint=(250, 195, 118))
    make_photo(root / "mountain-hiking-trail-20190812.jpg", ["bob"], tint=(120, 160, 120))
    make_photo(root / "birthday-cake-candles-20200103.jpg", ["alice", "carol"], tint=(240, 220, 240))
    make_photo(root / "nested" / "garden-flowers-spring-20210419.jpg", ["carol"], tint=(150, 220, 150))
    make_photo(root / "nested" / "deeper" / "snow-winter-skiing-20220211.jpg", ["dave"], tint=(230, 230, 255))
    make_photo(root / "city-street-night-lights.jpg", [], tint=(40, 40, 60))
    make_photo(root / "dog-park-running-20230530.jpg", ["bob", "dave"], tint=(180, 200, 160))
    return root


@pytest.fixture
def settings(tmp_path: Path, monkeypatch):
    """Isolated settings pointing at temp storage with stub models."""
    from photolib.config import get_settings, reset_settings_cache

    monkeypatch.setenv("PHOTO_DB_URI", str(tmp_path / "db"))
    monkeypatch.setenv("PHOTO_THUMBNAIL_CACHE_DIR", str(tmp_path / "thumbs"))
    monkeypatch.setenv("PHOTO_FACES_DIR", str(tmp_path / "faces"))
    monkeypatch.setenv("PHOTO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PHOTO_EMBED_BACKEND", "stub")
    monkeypatch.setenv("PHOTO_EMBED_MODEL", "stub-v1")
    monkeypatch.setenv("PHOTO_FACE_BACKEND", "stub")
    monkeypatch.setenv("PHOTO_FACE_MODEL", "stub-face-v1")
    # Brute-force vector search: an IVF_PQ index over 8 rows is meaningless
    # and its training would fail.
    monkeypatch.setenv("PHOTO_ANN_MIN_ROWS", "1000000")
    monkeypatch.setenv("PHOTO_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("PHOTO_INGEST_WORKERS", "2")
    # Stub identity vectors are random unit vectors, so same-person similarity
    # is 1.0 and different-person is ~0. Any threshold in between works.
    monkeypatch.setenv("PHOTO_FACE_MATCH_THRESHOLD", "0.6")
    monkeypatch.setenv("PHOTO_FACE_STRONG_MATCH_THRESHOLD", "0.8")
    monkeypatch.setenv("PHOTO_FACE_CLUSTER_THRESHOLD", "0.6")

    reset_settings_cache()
    s = get_settings()
    s.ensure_dirs()
    yield s
    reset_settings_cache()


@pytest.fixture
def library(settings):
    from photolib.db import Library

    return Library(settings.db_uri)


@pytest.fixture
def service(settings, library):
    from photolib.service import PhotoService

    return PhotoService(settings=settings, library=library)


@pytest.fixture
def indexer(service):
    from photolib.indexer import Indexer

    return Indexer(service.library, service.settings, service.embedder,
                   service.face_backend, service.thumbs)


@pytest.fixture
def indexed_service(service, indexer, photo_dir):
    """A service with the sample library already indexed."""
    indexer.index_directory(photo_dir)
    service.index.invalidate()
    return service


@pytest.fixture
def client(indexed_service):
    from fastapi.testclient import TestClient

    from photolib.api.app import create_app
    from photolib.api.deps import set_service

    set_service(indexed_service)
    app = create_app(indexed_service.settings)
    with TestClient(app) as c:
        yield c
    set_service(None)
