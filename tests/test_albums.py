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


def test_album_pages_reach_every_photo_beyond_500(client, indexed_service, monkeypatch):
    import pyarrow as pa

    service = indexed_service
    template = service.library.images.search().limit(1).to_arrow().to_pylist()[0]
    new_rows = [{**template, "image_id": i, "path": f"sample-{i}.jpg"}
                for i in range(1000, 1505)]
    service.library.images.add(pa.Table.from_pylist(new_rows, schema=service.library.images.schema))
    service.index.invalidate()
    album_id = service.create_album("Large album")["album_id"]
    service.add_album_items(album_id, list(range(1000, 1505)))
    hydrated_sizes = []
    hydrate = service.hydrate

    def track(ids):
        hydrated_sizes.append(len(ids))
        return hydrate(ids)

    monkeypatch.setattr(service, "hydrate", track)
    collected = []
    for page in range(1, 7):
        response = client.get(f"{API}/albums/{album_id}",
                              params={"page": page, "per_page": 100})
        assert response.status_code == 200
        detail = response.json()
        assert detail["total"] == detail["photo_count"] == 505
        assert detail["page"] == page
        assert detail["has_more"] is (page < 6)
        collected.extend(r["image_id"] for r in detail["images"])
    assert collected == list(range(1504, 999, -1))
    assert hydrated_sizes == [100, 100, 100, 100, 100, 5]
    assert len(client.get(f"{API}/albums/{album_id}?limit=3").json()["images"]) == 3
    assert client.get(f"{API}/albums/{album_id}?page=99&per_page=100").json()["images"] == []


def test_album_pagination_validates_bounds(client):
    for params in ({"page": 0}, {"per_page": 0}, {"per_page": 201}):
        assert client.get(f"{API}/albums/0", params=params).status_code == 422
