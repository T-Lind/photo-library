"""OCR: index-time extraction, backfill, and text-first search ranking.

Uses the stub OCR backend, which "reads" a photo's filename words as its
text — the same convention as the stub embedder, so no models run in CI.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def ocr_settings(settings, monkeypatch):
    monkeypatch.setenv("PHOTO_OCR_BACKEND", "stub")
    from photolib.config import get_settings, reset_settings_cache

    reset_settings_cache()
    s = get_settings()
    s.ensure_dirs()
    yield s
    reset_settings_cache()


@pytest.fixture
def ocr_service(ocr_settings, library):
    from photolib.service import PhotoService

    return PhotoService(settings=ocr_settings, library=library)


@pytest.fixture
def ocr_indexed(ocr_service, photo_dir):
    from photolib.indexer import Indexer

    indexer = Indexer(ocr_service.library, ocr_service.settings,
                      ocr_service.embedder, ocr_service.face_backend,
                      ocr_service.thumbs)
    indexer.index_directory(photo_dir)
    ocr_service.index.invalidate()
    return ocr_service


def _wait(job, timeout=30):
    deadline = time.time() + timeout
    while job.status in ("pending", "running") and time.time() < deadline:
        time.sleep(0.05)
    return job


def test_indexing_writes_one_ocr_row_per_image(ocr_indexed):
    library = ocr_indexed.library
    assert library.has_ocr()
    assert library.ocr.count_rows(None) == library.images.count_rows(None)


def test_text_matches_rank_ahead_of_semantic_results(ocr_indexed):
    from photolib.browse import Filters

    page = ocr_indexed.search("city street night", Filters(), sort="relevance")
    assert page.results
    top = page.results[0]
    assert top["filename"] == "city-street-night-lights.jpg"
    assert top.get("text_match") is True


def test_ocr_text_appears_in_image_details(ocr_indexed):
    from photolib.browse import Filters

    page = ocr_indexed.search("beach sunset holiday", Filters(), sort="relevance")
    details = ocr_indexed.image_details(page.results[0]["image_id"])
    assert "beach" in details["ocr_text"]


def test_backfill_scans_only_missing_images(settings, library, photo_dir,
                                            monkeypatch):
    """A library indexed before OCR existed gains text without re-indexing."""
    from photolib.indexer import Indexer
    from photolib.service import PhotoService

    service = PhotoService(settings=settings, library=library)
    # Index with OCR explicitly off, as an old build would have.
    service.settings.ocr_backend = "off"
    indexer = Indexer(library, settings, service.embedder,
                      service.face_backend, service.thumbs)
    indexer.index_directory(photo_dir)
    service.index.invalidate()
    assert not library.has_ocr() or library.ocr.count_rows(None) == 0

    images_version = library.images.version

    service.settings.ocr_backend = "stub"
    job = _wait(service.start_ocr_backfill_job())
    assert job.status == "done", job.error
    assert job.result["scanned"] == 8
    assert job.result["with_text"] == 8

    # The images table was never rewritten — no re-embedding happened.
    assert library.images.version == images_version

    # A second run has nothing to do.
    job = _wait(service.start_ocr_backfill_job())
    assert job.status == "done"
    assert job.result["scanned"] == 0
    assert job.result["already_scanned"] == 8

    from photolib.browse import Filters

    page = service.search("snow winter skiing", Filters(), sort="relevance")
    assert page.results[0]["filename"] == "snow-winter-skiing-20220211.jpg"
    assert page.results[0].get("text_match") is True


def test_stats_report_ocr_coverage(ocr_indexed):
    stats = ocr_indexed.stats()
    assert stats["ocr"]["scanned"] == 8
    assert stats["ocr"]["available"] is True
