from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from photolib import catalog, quality
from photolib.browse import Filters
from photolib.db import Library, UNASSIGNED
from tests.conftest import make_photo


def image(service, filename):
    return next(r for r in service.library.images.to_lance().to_table().to_pylist() if r["filename"] == filename)


def test_failed_reindex_keeps_photo_album_and_faces(indexed_service, indexer, photo_dir):
    service = indexed_service
    original = image(service, "beach-sunset-holiday-20180705.jpg")
    album = service.create_album("Keep")
    service.add_album_items(album["album_id"], [original["image_id"]])
    face = service.faces_in_image(original["image_id"])[0]
    service.assign_faces([face["face_id"]], face["person_id"])
    service.annotate(original["image_id"], True, 5)
    target = Path(original["path"])
    target.write_bytes(b"corrupted image")
    stats = indexer.index_directory(photo_dir)
    assert stats.failed == 1 and stats.updated == 0
    assert service.library.images.count_rows(None) == 8
    assert service.album_image_ids(album["album_id"]) == [original["image_id"]]
    assert service.get_face(face["face_id"])["confirmed"]
    assert service.annotations()[original["image_id"]]["rating"] == 5
    assert service.failures()[0]["path"] == str(target)


def test_reindex_preserves_stable_ids_and_confirmed_unassigned(indexed_service, indexer, photo_dir):
    service = indexed_service
    original = image(service, "beach-sunset-holiday-20180705.jpg")
    faces = service.faces_in_image(original["image_id"])
    service.assign_faces([faces[0]["face_id"]], faces[0]["person_id"])
    service.detach_faces([faces[1]["face_id"]])
    stats = indexer.index_directory(photo_dir, rebuild=True)
    assert stats.updated == 8
    assert image(service, original["filename"])["image_id"] == original["image_id"]
    assert image(service, original["filename"])["added_at"] == original["added_at"]
    assert service.get_face(faces[0]["face_id"])["confirmed"]
    assert service.get_face(faces[1]["face_id"])["person_id"] == UNASSIGNED
    assert service.get_face(faces[1]["face_id"])["confirmed"]


def test_changed_image_preserves_matching_corrections(indexed_service, indexer, photo_dir):
    service = indexed_service
    original = image(service, "beach-sunset-holiday-20180705.jpg")
    faces = service.faces_in_image(original["image_id"])
    for face in faces:
        service.assign_faces([face["face_id"]], face["person_id"])
    make_photo(Path(original["path"]), ["alice", "bob"], tint=(100, 120, 130))
    os.utime(original["path"], (original["mtime"]+10, original["mtime"]+10))
    stats = indexer.index_directory(photo_dir)
    assert stats.updated == 1
    for face in faces:
        restored = service.get_face(face["face_id"])
        assert restored["confirmed"] and restored["person_id"] == face["person_id"]


def test_retry_queue_survives_reopening(indexed_service, indexer, photo_dir):
    bad = photo_dir / "broken.jpg"
    bad.write_bytes(b"broken")
    indexer.index_directory(photo_dir)
    assert catalog.records(Library(indexed_service.library.uri), "failure:")
    make_photo(bad, ["alice"])
    report = indexer.retry_failed()
    assert report["added"] == 1
    assert indexed_service.failures() == []
    assert indexed_service.library.images.count_rows(None) == 9


def test_atomic_rolls_back_all_tables(indexed_service):
    service = indexed_service
    with pytest.raises(RuntimeError, match="injected"):
        with catalog.atomic(service.library):
            service.library.images.delete("image_id = 0")
            service.library.faces.delete("image_id = 0")
            catalog.put(service.library, "test:new", {"value": 1})
            raise RuntimeError("injected")
    assert service.library.images.count_rows(None) == 8
    assert service.library.faces.count_rows(None) == 10
    assert catalog.records(service.library, "test:") == {}


def test_deleted_ids_are_never_reused(indexed_service, indexer, photo_dir):
    service = indexed_service
    old_max = max(r["image_id"] for r in service.library.images.to_lance().to_table(
        columns=["image_id"]).to_pylist())
    victim = image(service, "city-street-night-lights.jpg")
    indexer.remove_images([victim["image_id"]])
    replacement = make_photo(photo_dir / "replacement-20260101.jpg")
    indexer.index_directory(photo_dir)
    added = image(service, replacement.name)
    assert added["image_id"] > old_max


def test_restart_rolls_back_interrupted_edit(indexed_service):
    lib = indexed_service.library
    journal = Path(lib.uri) / ".pending-edit.json"
    journal.write_text(json.dumps({n: lib.table(n).version for n in lib.table_names()}))
    lib.images.delete("image_id = 0")
    reopened = Library(lib.uri)
    assert reopened.images.count_rows(None) == 8
    assert not journal.exists()


def test_annotations_and_saved_search_persist_and_filter(client, indexed_service):
    service = indexed_service
    photo = service.search(None, Filters()).results[0]
    response = client.patch(f"/api/v1/images/{photo['image_id']}/annotation", json={"favorite": True, "rating": 4})
    assert response.status_code == 200
    filtered = client.post("/api/v1/search", json={"favorites_only": True, "min_rating": 4}).json()
    assert [r["image_id"] for r in filtered["results"]] == [photo["image_id"]]
    assert client.post("/api/v1/search", json={"min_rating": 5}).json()["total"] == 0
    assert client.patch(f"/api/v1/images/{photo['image_id']}/annotation", json={"rating": 6}).status_code == 422
    saved = client.post("/api/v1/saved-searches", json={"name":"Best sunsets", "request":{
        "query":"sunset", "people_ids":photo["people_ids"], "people_mode":"all", "favorites_only":True,
        "sort":"quality"}}).json()
    assert saved["request"]["people_mode"] == "all"
    assert catalog.records(Library(service.library.uri), "search:")[saved["id"]]["request"]["sort"] == "quality"


def test_quality_is_measured_at_ingest(indexed_service):
    scores = quality.get(indexed_service.library)
    assert len(scores) == 8
    assert all(0 <= r["score"] <= 1 for r in scores.values())


def test_quality_rules_penalize_blur_and_clipping():
    rng = np.random.default_rng(42)
    textured = rng.integers(30, 220, (512,512,3), dtype=np.uint8)
    flat = np.full_like(textured, 128)
    white = np.full_like(textured, 255)
    a = quality.measure(textured, 4000, 3000)
    b = quality.measure(flat, 4000, 3000)
    c = quality.measure(white, 4000, 3000)
    assert a["sharpness"] > b["sharpness"]
    assert c["exposure"] < b["exposure"]
    assert quality.measure(textured, 200, 150)["resolution"] < a["resolution"]


def test_quality_ranking_uses_weaker_selected_face(indexed_service):
    service = indexed_service
    photos = service.search(None, Filters()).results
    pair = next(r for r in photos if r["filename"] == "beach-sunset-holiday-20180705.jpg")
    face_rows = service.faces_in_image(pair["image_id"])
    for face, value in zip(face_rows, [.95, .1]):
        service.library.faces.update(where=f"face_id = {face['face_id']}", values={"quality": value})
    ranked = service.quality_scores([pair["image_id"]], pair["people_ids"])[pair["image_id"]]
    assert any("Weakest selected face 10%" in reason for reason in ranked["quality_reasons"])
    base = quality.get(service.library, [pair["image_id"]])[pair["image_id"]]["score"]
    assert ranked["quality_score"] == pytest.approx(.5 * base + .05, abs=1e-4)
    result = service.search("sunset", Filters(people_ids=pair["people_ids"], people_mode="all"), sort="quality")
    assert result.total == 1
    assert result.results[0]["quality_score"] is not None


def test_verified_relocation_preserves_every_id(indexed_service, photo_dir, tmp_path):
    service = indexed_service
    service.add_root(str(photo_dir))
    source_id = service.list_roots()[0]["source_id"]
    target = tmp_path / "replacement-drive"
    shutil.copytree(photo_dir, target)
    before = service.library.images.to_lance().to_table(columns=["image_id", "path"]).to_pylist()
    preview = service.relocate_root(str(photo_dir), str(target))
    assert preview["photos"] == 8
    assert service.image_path(before[0]["image_id"]) == before[0]["path"]
    result = service.relocate_root(str(photo_dir), str(target), preview["verification"])
    assert result["relocated"] == 8
    assert service.list_roots()[0]["source_id"] == source_id
    for row in before:
        assert service.image_path(row["image_id"]) == str(target / Path(row["path"]).relative_to(photo_dir))


def test_relocation_rejects_wrong_content_without_changes(indexed_service, photo_dir, tmp_path):
    service = indexed_service
    service.add_root(str(photo_dir))
    target = tmp_path / "wrong-drive"
    shutil.copytree(photo_dir, target)
    (target / "city-street-night-lights.jpg").write_bytes(b"different")
    with pytest.raises(ValueError, match="Content differs"):
        service.relocate_root(str(photo_dir), str(target))
    assert service.list_roots()[0]["path"] == str(photo_dir)


def test_backup_preserves_split_and_confirmed_unassigned(indexed_service):
    service = indexed_service
    photo = image(service, "beach-sunset-holiday-20180705.jpg")
    faces = service.faces_in_image(photo["image_id"])
    split = service.assign_faces([faces[0]["face_id"]], None, "A separate identity")["person_id"]
    service.detach_faces([faces[1]["face_id"]])
    service.annotate(photo["image_id"], True, 5)
    backup = service.export_curation()
    assert backup["version"] == 2 and backup["checksum"]
    from photolib.backup import preview
    assert preview(service, backup)["faces_matched"] == 2
    service.library.faces.update(where=f"image_id = {photo['image_id']}", values={"person_id":faces[0]["person_id"], "confirmed":False})
    service.annotate(photo["image_id"], False, 0)
    report = service.import_curation(backup)
    assert report["faces_matched"] == 2
    assert service.get_face(faces[0]["face_id"])["person_id"] == split
    assert service.get_face(faces[1]["face_id"])["person_id"] == UNASSIGNED
    assert service.get_face(faces[1]["face_id"])["confirmed"]
    assert service.annotations()[photo["image_id"]]["rating"] == 5


def test_backup_checksum_prevents_partial_restore(indexed_service):
    service = indexed_service
    data = service.export_curation()
    data["albums"].append({"name":"Tampered", "photos":[]})
    with pytest.raises(ValueError, match="checksum"):
        service.import_curation(data)
    assert service.list_albums() == []


def test_burst_comparison_keeps_all_originals(indexed_service):
    service = indexed_service
    a = image(service, "beach-sunset-holiday-20180704.jpg")
    b = image(service, "beach-sunset-holiday-20180705.jpg")
    service.library.images.update(where=f"image_id = {b['image_id']}", values={"taken_at":a["taken_at"],"phash":a["phash"]})
    service.index.invalidate()
    burst = service.burst_candidates(a["image_id"])
    assert {a["image_id"], b["image_id"]}.issubset({r["image_id"] for r in burst["images"]})
    assert service.library.images.count_rows(None) == 8
    assert Path(a["path"]).exists() and Path(b["path"]).exists()
