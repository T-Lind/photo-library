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
    assert any("nested/deeper" in p.replace("\\", "/") for p in paths)


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


def test_face_boxes_are_stored_in_original_image_coordinates(indexer, service,
                                                             tmp_path):
    """Detection runs on a downscaled buffer; boxes must be scaled back.

    Otherwise a face crop taken from the full-resolution original lands on

    the wrong part of the photo, and the UI's overlay is offset too.
    """
    from photolib.faces.stub import MARKER_WIDTH
    from photolib.indexer import WORK_MAX_SIDE
    from tests.conftest import make_photo

    scale = 2
    big = tmp_path / "big"
    # Twice the working buffer's long edge, so a scale factor is applied.
    make_photo(big / "huge-photo.jpg", ["alice"],
               size=(WORK_MAX_SIDE * scale, 1200))
    indexer.index_directory(big)

    faces = service.library.faces.to_lance().to_table(
        columns=["x", "w", "h"]).to_pylist()
    images = service.library.images.to_lance().to_table(
        columns=["width"]).to_pylist()

    assert faces, "no face detected in the oversized photo"
    assert images[0]["width"] == WORK_MAX_SIDE * scale
    # The stub reports a MARKER_WIDTH-wide box in *buffer* coordinates.
    # Stored unscaled it would be MARKER_WIDTH; correctly scaled it is
    # MARKER_WIDTH * scale.
    assert faces[0]["w"] == MARKER_WIDTH * scale


def test_merged_person_captures_future_photos(indexer, service, photo_dir):
    """A merge must hold for photos indexed later, not just existing ones."""
    indexer.index_directory(photo_dir)
    service.index.invalidate()

    # Locate the person in the solo beach photo and split one face off into
    # a second identity, as a bad clustering run would.
    paths = service.library.images.to_lance().to_table(
        columns=["image_id", "path"]).to_pylist()
    beach = next(r for r in paths
                 if r["path"].endswith("beach-sunset-holiday-20180704.jpg"))
    face = service.faces_in_image(beach["image_id"])[0]
    original = face["person_id"]

    split = service.assign_faces([face["face_id"]], person_id=None)["person_id"]
    assert split != original

    service.merge_people(split, original)

    # A new photo of the same person must land on the merged identity.
    make_photo(photo_dir / "picnic-blanket-park-20240102.jpg", ["alice"])
    stats = indexer.index_directory(photo_dir)
    assert stats.added == 1

    new = next(r for r in service.library.images.to_lance().to_table(
        columns=["image_id", "path"]).to_pylist()
        if r["path"].endswith("picnic-blanket-park-20240102.jpg"))
    new_faces = service.faces_in_image(new["image_id"])
    assert new_faces
    assert all(f["person_id"] == original for f in new_faces)


def test_indexing_reuses_the_working_decode_for_hash_and_thumbnails(
        indexer, service, tmp_path, monkeypatch):
    from photolib import imageio

    root = tmp_path / "single-decode"
    make_photo(root / "photo.jpg", ["alice"])

    calls = 0
    real_open = imageio.open_image

    def counting_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(imageio, "open_image", counting_open)
    indexer.index_directory(root)

    assert calls == 1
    image_id = int(service.library.images.to_lance().to_table(
        columns=["image_id"])["image_id"][0].as_py())
    assert service.thumbs.thumbnail_path(image_id, "small").exists()
    assert service.thumbs.thumbnail_path(image_id, "grid").exists()
