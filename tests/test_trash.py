"""Trashing photos: Recycle Bin delivery and complete row cleanup.

The OS trash itself is stubbed (tests must not touch the real Recycle
Bin); everything else — row deletion, person recomputation, album
cleanup — runs against the real database.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

API = "/api/v1"


@pytest.fixture
def fake_trash(monkeypatch, tmp_path):
    """Capture files "sent to the Recycle Bin" in an inspectable folder."""
    bin_dir = tmp_path / "recycle-bin"
    bin_dir.mkdir()
    sent = []

    def fake(path: str) -> None:
        shutil.move(path, bin_dir / Path(path).name)
        sent.append(Path(path))

    monkeypatch.setattr("photolib.service._send_to_trash", fake)
    return sent


def _search_ids(client, query=None, **kwargs):
    body = client.post(f"{API}/search",
                       json={"query": query, "per_page": 100, **kwargs}).json()
    return [r["image_id"] for r in body["results"]], body["total"]


def test_trash_moves_file_and_forgets_the_photo(client, indexed_service,
                                                fake_trash):
    ids, total = _search_ids(client)
    victim = ids[0]
    path = Path(indexed_service.image_path(victim))
    assert path.exists()

    body = client.post(f"{API}/images/trash",
                       json={"image_ids": [victim]}).json()
    assert body == {"trashed": 1, "missing": 0, "removed": 1, "failed": []}

    # The file went to the (fake) bin, not into oblivion.
    assert not path.exists()
    assert fake_trash == [path]

    ids_after, total_after = _search_ids(client)
    assert victim not in ids_after
    assert total_after == total - 1
    assert client.get(f"{API}/images/{victim}/details").status_code == 404
    assert indexed_service.library.faces.count_rows(
        f"image_id = {victim}") == 0


def test_trash_recomputes_people_counts(client, fake_trash):
    people = client.get(f"{API}/people?min_photos=1").json()
    person = max(people, key=lambda p: p["photo_count"])
    assert person["photo_count"] >= 2

    ids, _ = _search_ids(client, people_ids=[person["person_id"]])
    client.post(f"{API}/images/trash", json={"image_ids": [ids[0]]})

    after = client.get(f"{API}/people/{person['person_id']}").json()
    assert after["photo_count"] == person["photo_count"] - 1
    assert after["face_count"] == person["face_count"] - 1


def test_trash_cleans_a_row_whose_file_is_already_gone(client,
                                                       indexed_service,
                                                       fake_trash):
    ids, _ = _search_ids(client)
    victim = ids[-1]
    os.remove(indexed_service.image_path(victim))

    body = client.post(f"{API}/images/trash",
                       json={"image_ids": [victim]}).json()
    assert body["missing"] == 1
    assert body["trashed"] == 0
    assert body["removed"] == 1
    assert fake_trash == []

    ids_after, _ = _search_ids(client)
    assert victim not in ids_after


def test_trash_removes_album_membership(client, fake_trash):
    ids, _ = _search_ids(client)
    album = client.post(f"{API}/albums", json={"name": "Keepers"}).json()
    client.post(f"{API}/albums/{album['album_id']}/items",
                json={"image_ids": ids[:2]})

    client.post(f"{API}/images/trash", json={"image_ids": [ids[0]]})

    detail = client.get(f"{API}/albums/{album['album_id']}").json()
    assert detail["photo_count"] == 1
    assert {img["image_id"] for img in detail["images"]} == {ids[1]}


def test_trash_requires_at_least_one_id(client, fake_trash):
    assert client.post(f"{API}/images/trash",
                       json={"image_ids": []}).status_code == 422


def test_unknown_ids_are_ignored_quietly(client, fake_trash):
    body = client.post(f"{API}/images/trash",
                       json={"image_ids": [987654]}).json()
    assert body == {"trashed": 0, "missing": 0, "removed": 0, "failed": []}


def test_duplicates_carry_file_sizes(client, indexed_service, indexer,
                                     photo_dir):
    # Plant an exact duplicate, then re-index incrementally.
    original = photo_dir / "beach-sunset-holiday-20180704.jpg"
    shutil.copy(original, photo_dir / "beach-copy.jpg")
    indexer.index_directory(photo_dir)
    indexed_service.index.invalidate()

    groups = client.get(f"{API}/admin/duplicates").json()["groups"]
    identical = [g for g in groups if g["kind"] == "identical"]
    assert identical, "the planted copy must be found"
    group = identical[0]
    assert [i["image_id"] for i in group["items"]] == group["image_ids"]
    assert all(i["file_size"] > 0 for i in group["items"])
