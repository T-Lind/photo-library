"""Albums: CRUD, membership, and similarity-based suggestions."""

from __future__ import annotations

API = "/api/v1"


def _ids(client, query=None, n=8):
    body = client.post(f"{API}/search", json={"query": query, "per_page": n}).json()
    return [r["image_id"] for r in body["results"]]


def test_album_lifecycle(client):
    created = client.post(f"{API}/albums", json={"name": "Holidays"}).json()
    album_id = created["album_id"]
    assert created["name"] == "Holidays"
    assert created["photo_count"] == 0

    image_ids = _ids(client)[:3]
    added = client.post(f"{API}/albums/{album_id}/items",
                        json={"image_ids": image_ids}).json()
    assert added["added"] == 3
    assert added["photo_count"] == 3

    # Adding the same photos again is a no-op, not a duplicate.
    again = client.post(f"{API}/albums/{album_id}/items",
                        json={"image_ids": image_ids}).json()
    assert again["added"] == 0
    assert again["photo_count"] == 3

    detail = client.get(f"{API}/albums/{album_id}").json()
    assert {img["image_id"] for img in detail["images"]} == set(image_ids)

    renamed = client.patch(f"{API}/albums/{album_id}",
                           json={"name": "Trips"}).json()
    assert renamed["name"] == "Trips"

    removed = client.post(f"{API}/albums/{album_id}/items/remove",
                          json={"image_ids": [image_ids[0]]}).json()
    assert removed["photo_count"] == 2

    listing = client.get(f"{API}/albums").json()["albums"]
    assert any(a["album_id"] == album_id and a["cover_image_id"] >= 0
               for a in listing)

    client.delete(f"{API}/albums/{album_id}")
    assert client.get(f"{API}/albums/{album_id}").status_code == 404


def test_album_suggestions_exclude_members(client):
    beach = _ids(client, "beach sunset holiday", 2)
    album = client.post(f"{API}/albums", json={"name": "Beach"}).json()
    client.post(f"{API}/albums/{album['album_id']}/items",
                json={"image_ids": beach})

    body = client.get(
        f"{API}/albums/{album['album_id']}/suggestions?limit=10").json()
    suggested = {s["image_id"] for s in body["suggestions"]}
    assert suggested.isdisjoint(set(beach))
    assert all("score" in s for s in body["suggestions"])


def test_unknown_album_is_404(client):
    assert client.get(f"{API}/albums/9999").status_code == 404
    assert client.post(f"{API}/albums/9999/items",
                       json={"image_ids": [1]}).status_code == 404
