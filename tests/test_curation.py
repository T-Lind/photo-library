"""Curation backup: export and restore of names, hidden flags, and albums.

The export is keyed by content_hash, so a restore must work even when
image ids and person ids differ — here that is simulated by wiping the
curated data in place and importing the backup back.
"""

from __future__ import annotations

API = "/api/v1"


def _people(client):
    return client.get(f"{API}/people?min_photos=1&include_hidden=true").json()


def _search_ids(client, n=8):
    body = client.post(f"{API}/search", json={"per_page": n}).json()
    return [r["image_id"] for r in body["results"]]


def _curate(client):
    """Name the two biggest people, hide the third, make an album."""
    people = sorted(_people(client), key=lambda p: -p["photo_count"])
    named = []
    for person, name in zip(people, ["Alice", "Bob"]):
        client.patch(f"{API}/people/{person['person_id']}",
                     json={"name": name})
        named.append((person["person_id"], name))
    hidden_id = people[2]["person_id"]
    client.post(f"{API}/people/{hidden_id}/hidden", json={"hidden": True})

    image_ids = _search_ids(client)[:3]
    album = client.post(f"{API}/albums", json={"name": "Trip"}).json()
    client.post(f"{API}/albums/{album['album_id']}/items",
                json={"image_ids": image_ids})
    return named, hidden_id, image_ids


def test_export_contains_the_hand_made_data(client):
    named, hidden_id, image_ids = _curate(client)

    backup = client.get(f"{API}/admin/curation").json()
    assert backup["format"] == "photolib-curation"
    assert backup["version"] == 1

    names = {p["name"] for p in backup["people"] if p["name"]}
    assert names == {"Alice", "Bob"}
    assert any(p["hidden"] for p in backup["people"])
    # Photos are content hashes, not ids — ids do not survive a rebuild.
    for person in backup["people"]:
        assert all(isinstance(h, str) and h for h in person["photos"])

    albums = {a["name"]: a for a in backup["albums"]}
    assert len(albums["Trip"]["photos"]) == len(image_ids)


def test_round_trip_restores_albums_and_people(client, indexed_service):
    named, hidden_id, image_ids = _curate(client)
    backup = client.get(f"{API}/admin/curation").json()

    # Wipe the curation in place: forget names, unhide, drop the album.
    for person_id, _ in named:
        indexed_service.rename_person(person_id, "")
    indexed_service.set_person_hidden(hidden_id, False)
    album_id = client.get(f"{API}/albums").json()["albums"][0]["album_id"]
    client.delete(f"{API}/albums/{album_id}")
    assert not any(p["name"] for p in _people(client))

    report = client.post(f"{API}/admin/curation", json=backup).json()
    assert report["albums_created"] == 1
    assert report["album_items_added"] == len(image_ids)
    assert report["people_restored"] == 3  # Alice, Bob, and the hidden one
    assert report["people_skipped"] == 0

    restored = {p["person_id"]: p for p in _people(client)}
    for person_id, name in named:
        assert restored[person_id]["name"] == name
    assert restored[hidden_id]["hidden"] is True

    album = client.get(f"{API}/albums").json()["albums"][0]
    detail = client.get(f"{API}/albums/{album['album_id']}").json()
    assert detail["name"] == "Trip"
    assert {img["image_id"] for img in detail["images"]} == set(image_ids)


def test_import_is_idempotent(client):
    _curate(client)
    backup = client.get(f"{API}/admin/curation").json()

    report = client.post(f"{API}/admin/curation", json=backup).json()
    assert report["albums_created"] == 0
    assert report["album_items_added"] == 0

    albums = client.get(f"{API}/albums").json()["albums"]
    assert [a["name"] for a in albums] == ["Trip"]


def test_import_never_overwrites_a_different_name(client, indexed_service):
    named, _, _ = _curate(client)
    backup = client.get(f"{API}/admin/curation").json()

    # The user renamed Alice to Alicia after the backup was taken.
    alice_id = next(pid for pid, name in named if name == "Alice")
    indexed_service.rename_person(alice_id, "Alicia")

    report = client.post(f"{API}/admin/curation", json=backup).json()
    assert report["people_skipped"] >= 1
    assert client.get(f"{API}/people/{alice_id}").json()["name"] == "Alicia"


def test_import_rejects_foreign_json(client):
    assert client.post(f"{API}/admin/curation",
                       json={"format": "something-else"}).status_code == 400
