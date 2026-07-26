"""Getting files out: Show in Explorer and export-copies-to-folder.

Neither operation may ever touch an original: reveal only points the OS file
manager at it, and export writes new copies with collision-safe names.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

API = "/api/v1"


@pytest.fixture
def reveal_calls(monkeypatch):
    calls: list = []
    monkeypatch.setattr("photolib.service._reveal_in_file_manager", calls.append)
    return calls


def _ids(client, n=3):
    body = client.post(f"{API}/search", json={"per_page": 20}).json()
    return [r["image_id"] for r in body["results"][:n]]


def _path_of(client, image_id):
    details = client.get(f"{API}/images/{image_id}/details").json()
    return Path(details["folder"]) / details["filename"]


def test_reveal_points_at_the_original(client, reveal_calls):
    image_id = _ids(client, 1)[0]
    r = client.post(f"{API}/images/{image_id}/reveal")
    assert r.status_code == 200
    assert reveal_calls == [str(_path_of(client, image_id))]


def test_reveal_unknown_image_is_404(client, reveal_calls):
    assert client.post(f"{API}/images/987654/reveal").status_code == 404
    assert reveal_calls == []


def test_reveal_missing_file_is_404(client, reveal_calls):
    image_id = _ids(client, 1)[0]
    os.remove(_path_of(client, image_id))
    assert client.post(f"{API}/images/{image_id}/reveal").status_code == 404
    assert reveal_calls == []


def test_export_copies_and_never_moves(client, tmp_path):
    ids = _ids(client)
    sources = [_path_of(client, i) for i in ids]
    dest = tmp_path / "export-out"
    dest.mkdir()

    body = client.post(f"{API}/images/export",
                       json={"image_ids": ids, "folder": str(dest)}).json()
    assert body["copied"] == 3
    assert body["missing"] == 0 and body["failed"] == []

    copied = sorted(p.name for p in dest.iterdir())
    assert copied == sorted(s.name for s in sources)
    for source in sources:
        assert source.exists()  # originals untouched


def test_export_renames_instead_of_overwriting(client, tmp_path):
    ids = _ids(client)
    dest = tmp_path / "export-out"
    dest.mkdir()

    first = client.post(f"{API}/images/export",
                        json={"image_ids": ids, "folder": str(dest)}).json()
    second = client.post(f"{API}/images/export",
                         json={"image_ids": ids, "folder": str(dest)}).json()
    assert first["copied"] == 3 and second["copied"] == 3
    names = [p.name for p in dest.iterdir()]
    assert len(names) == 6
    assert sum(1 for n in names if "(2)" in n) == 3


def test_export_counts_missing_originals(client, tmp_path):
    ids = _ids(client)
    os.remove(_path_of(client, ids[0]))
    dest = tmp_path / "export-out"
    dest.mkdir()

    body = client.post(f"{API}/images/export",
                       json={"image_ids": ids, "folder": str(dest)}).json()
    assert body["copied"] == 2
    assert body["missing"] == 1


def test_export_into_a_missing_folder_is_400(client, tmp_path):
    ids = _ids(client, 1)
    r = client.post(f"{API}/images/export",
                    json={"image_ids": ids,
                          "folder": str(tmp_path / "nope" / "nowhere")})
    assert r.status_code == 400


def test_export_into_the_source_folder_makes_no_duplicate(client):
    """Exporting a photo to its own folder is a no-op, not a '(2)' copy."""
    image_id = _ids(client, 1)[0]
    source = _path_of(client, image_id)
    before = sorted(p.name for p in source.parent.iterdir())

    body = client.post(f"{API}/images/export",
                       json={"image_ids": [image_id],
                             "folder": str(source.parent)}).json()
    assert body["copied"] == 1
    assert sorted(p.name for p in source.parent.iterdir()) == before
