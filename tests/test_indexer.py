"""Indexing: recursion, incrementality, metadata, and face extraction."""

from __future__ import annotations

import numpy as np
import pytest

from photolib.db import UNASSIGNED
from tests.conftest import make_photo


def test_indexes_every_photo_including_subfolders(indexer, service, photo_dir):
    stats = indexer.index_directory(photo_dir)

    assert stats.scanned == 8
    assert stats.added == 8
    assert stats.failed == 0
    # The old flat os.listdir scan missed everything under nested/.
    assert service.library.images.count_rows(None) == 8

    paths = service.library.images.to_lance().to_table(columns=["path"])["path"].to_pylist()
    assert any("nested/deeper" in p for p in paths)


def test_embeddings_are_normalised(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    vectors = service.library.images.to_lance().to_table(columns=["vector"])["vector"]
    norms = [float(np.linalg.norm(np.asarray(v, dtype=np.float32)))
             for v in vectors.to_pylist()]
    assert all(abs(n - 1.0) < 1e-4 for n in norms)


def test_second_run_skips_unchanged_files(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    stats = indexer.index_directory(photo_dir)

    assert stats.added == 0
    assert stats.skipped == 8
    assert service.library.images.count_rows(None) == 8


def test_new_photos_are_added_incrementally(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    make_photo(photo_dir / "lake-canoe-summer-20240701.jpg", ["alice"])

    stats = indexer.index_directory(photo_dir)

    assert stats.added == 1
    assert stats.skipped == 8
    assert service.library.images.count_rows(None) == 9


def test_changed_file_is_reindexed_not_duplicated(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    target = photo_dir / "mountain-hiking-trail-20190812.jpg"
    make_photo(target, ["bob", "carol"], tint=(10, 90, 10))
    # Force a different mtime even on coarse-resolution filesystems.
    import os
    import time
    os.utime(target, (time.time() + 10, time.time() + 10))

    stats = indexer.index_directory(photo_dir)

    assert stats.updated == 1
    assert service.library.images.count_rows(None) == 8
    assert service.library.images.count_rows(f"path = '{target}'") == 1


def test_prune_removes_deleted_files(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    (photo_dir / "city-street-night-lights.jpg").unlink()

    stats = indexer.index_directory(photo_dir, prune_missing=True)

    assert stats.removed == 1
    assert service.library.images.count_rows(None) == 7


def test_dates_are_extracted_and_missing_dates_stay_null(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    table = service.library.images.to_lance().to_table(columns=["filename", "taken_at"])
    dates = dict(zip(table["filename"].to_pylist(), table["taken_at"].to_pylist()))

    assert dates["beach-sunset-holiday-20180704.jpg"].year == 2018
    assert dates["beach-sunset-holiday-20180704.jpg"].month == 7
    # No date anywhere: stays NULL rather than becoming 1970.
    assert dates["city-street-night-lights.jpg"] is None


def test_faces_are_detected_and_linked_to_images(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)

    faces = service.library.faces.to_lance().to_table(
        columns=["image_id", "person_id"])
    # 1+2+1+2+1+1+0+2 markers across the sample set.
    assert faces.num_rows == 10
    assert all(p != UNASSIGNED for p in faces["person_id"].to_pylist())


def test_people_ids_are_denormalised_onto_images(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    table = service.library.images.to_lance().to_table(
        columns=["filename", "people_ids", "face_count"])
    rows = {f: (p, c) for f, p, c in zip(table["filename"].to_pylist(),
                                         table["people_ids"].to_pylist(),
                                         table["face_count"].to_pylist())}

    assert len(rows["beach-sunset-holiday-20180705.jpg"][0]) == 2
    assert rows["beach-sunset-holiday-20180705.jpg"][1] == 2
    assert rows["city-street-night-lights.jpg"][0] == []


def test_broken_file_is_reported_not_fatal(indexer, service, photo_dir):
    (photo_dir / "corrupt-photo.jpg").write_bytes(b"this is not a JPEG")

    stats = indexer.index_directory(photo_dir)

    assert stats.failed == 1
    assert stats.added == 8
    assert any("corrupt-photo" in e for e in stats.errors)


def test_thumbnails_are_pregenerated(indexer, service, photo_dir):
    indexer.index_directory(photo_dir)
    ids = service.library.images.to_lance().to_table(
        columns=["image_id"])["image_id"].to_pylist()
    assert service.thumbs.thumbnail_path(int(ids[0]), "grid").exists()


def test_model_change_is_refused_rather_than_silently_corrupting(indexer, service,
                                                                photo_dir):
    from photolib.db import SchemaMismatch
    from photolib.embeddings.stub import StubEmbedder

    indexer.index_directory(photo_dir)
    indexer.embedder = StubEmbedder(dim=128, model_name="different-model")

    with pytest.raises(SchemaMismatch, match="different-model"):
        indexer.index_directory(photo_dir)
