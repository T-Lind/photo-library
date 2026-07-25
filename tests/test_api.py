"""HTTP layer: contracts, status codes, caching, and job handling."""

from __future__ import annotations

import io
import time

import pytest

API = "/api/v1"


def test_health_reports_readiness_without_loading_models(client):
    body = client.get(f"{API}/health").json()
    assert body["status"] == "ok"
    assert body["ready"] is True


def test_stats_endpoint(client):
    body = client.get(f"{API}/stats").json()
    assert body["total_images"] == 8
    assert body["total_people"] == 4


def test_search_post_returns_paginated_results(client):
    body = client.post(f"{API}/search",
                       json={"query": "beach sunset", "per_page": 3}).json()

    assert body["total"] > 0
    assert body["per_page"] == 3
    assert len(body["results"]) <= 3
    assert body["scored"] is True
    assert body["results"][0]["filename"].startswith("beach-sunset")


def test_search_get_is_bookmarkable(client):
    body = client.get(f"{API}/search", params={"q": "snow winter skiing"}).json()
    assert "snow-winter-skiing" in body["results"][0]["filename"]


def test_search_accepts_z_suffixed_dates(client):
    """The frontend sends Date.toISOString(), which always ends in Z."""
    response = client.post(f"{API}/search", json={
        "start_date": "2018-01-01T00:00:00.000Z",
        "end_date": "2019-12-31T23:59:59.000Z",
    })
    assert response.status_code == 200
    assert response.json()["total"] == 3


def test_search_rejects_a_bad_sort_option(client):
    response = client.post(f"{API}/search", json={"sort": "by_vibes"})
    assert response.status_code == 422


def test_people_listing_is_sorted_and_typed(client):
    people = client.get(f"{API}/people").json()

    assert len(people) == 4
    assert people == sorted(people, key=lambda p: (not p["name"], -p["photo_count"],
                                                   p["person_id"]))
    assert all("cover_face_id" in p for p in people)


def test_min_photos_filters_the_long_tail(client):
    assert len(client.get(f"{API}/people", params={"min_photos": 3}).json()) == 2


def test_rename_person_round_trips(client):
    person = client.get(f"{API}/people").json()[0]
    updated = client.patch(f"{API}/people/{person['person_id']}",
                           json={"name": "Grandma"}).json()

    assert updated["name"] == "Grandma"
    assert updated["photo_count"] == person["photo_count"]
    assert client.get(f"{API}/people/{person['person_id']}").json()["name"] == "Grandma"


def test_unknown_person_is_404_not_500(client):
    assert client.get(f"{API}/people/9999").status_code == 404
    assert client.patch(f"{API}/people/9999", json={"name": "x"}).status_code == 404


def test_merging_a_person_into_themselves_is_a_400(client):
    response = client.post(f"{API}/people/merge", json={"source_id": 0, "target_id": 0})
    assert response.status_code == 400


def test_hide_person(client):
    person = client.get(f"{API}/people").json()[0]
    client.post(f"{API}/people/{person['person_id']}/hidden", json={"hidden": True})

    visible = client.get(f"{API}/people").json()
    assert person["person_id"] not in [p["person_id"] for p in visible]
    assert len(client.get(f"{API}/people",
                          params={"include_hidden": True}).json()) == 4


def test_image_details_include_faces_and_people(client):
    image_id = client.post(f"{API}/search", json={"has_faces": True}).json()[
        "results"][0]["image_id"]
    details = client.get(f"{API}/images/{image_id}/details").json()

    assert details["image_id"] == image_id
    assert details["faces"]
    assert details["people"]
    assert "bbox" in details["faces"][0]


def test_thumbnail_is_served_and_cached(client):
    image_id = client.post(f"{API}/search", json={}).json()["results"][0]["image_id"]

    first = client.get(f"{API}/images/{image_id}/thumbnail", params={"size": "grid"})
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/webp"
    assert "ETag" in first.headers

    # A scrolling photo grid re-requests the same thumbnails constantly.
    again = client.get(f"{API}/images/{image_id}/thumbnail",
                       params={"size": "grid"},
                       headers={"If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304


def test_thumbnail_sizes_differ(client):
    image_id = client.post(f"{API}/search", json={}).json()["results"][0]["image_id"]
    small = client.get(f"{API}/images/{image_id}/thumbnail", params={"size": "small"})
    large = client.get(f"{API}/images/{image_id}/thumbnail", params={"size": "large"})
    assert len(small.content) != len(large.content)


def test_jpeg_thumbnail_format_is_available(client):
    image_id = client.post(f"{API}/search", json={}).json()["results"][0]["image_id"]
    response = client.get(f"{API}/images/{image_id}/thumbnail",
                          params={"format": "jpeg"})
    assert response.headers["content-type"] == "image/jpeg"


def test_original_image_download(client):
    image_id = client.post(f"{API}/search", json={}).json()["results"][0]["image_id"]
    response = client.get(f"{API}/images/{image_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_missing_image_is_404(client):
    assert client.get(f"{API}/images/999999").status_code == 404
    assert client.get(f"{API}/images/999999/thumbnail").status_code == 404


def test_similar_images_endpoint(client):
    image_id = client.post(f"{API}/search", json={}).json()["results"][0]["image_id"]
    results = client.get(f"{API}/images/{image_id}/similar",
                         params={"limit": 4}).json()["results"]
    assert 0 < len(results) <= 4
    assert all(r["image_id"] != image_id for r in results)


def test_face_crop_is_generated_on_demand(client):
    person = client.get(f"{API}/people").json()[0]
    response = client.get(f"{API}/faces/{person['cover_face_id']}/crop")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_face_search_by_face_id(client):
    person = client.get(f"{API}/people").json()[0]
    body = client.post(f"{API}/faces/search", json={
        "face_id": person["cover_face_id"], "min_similarity": 0.5}).json()

    assert body["faces"]
    assert all(f["person_id"] == person["person_id"] for f in body["faces"])


def test_face_search_needs_a_subject(client):
    assert client.post(f"{API}/faces/search", json={}).status_code == 400


def test_face_search_by_upload(client, photo_dir):
    target = photo_dir / "beach-sunset-holiday-20180704.jpg"
    with open(target, "rb") as fh:
        response = client.post(f"{API}/faces/search/by-upload",
                               files={"file": ("q.jpg", fh, "image/jpeg")},
                               params={"min_similarity": 0.5})

    assert response.status_code == 200
    assert response.json()["faces"]


def test_reverse_image_search_endpoint(client, photo_dir):
    target = photo_dir / "dog-park-running-20230530.jpg"
    with open(target, "rb") as fh:
        # The stub embedder reads content from the filename, so the upload
        # keeps its original name.
        response = client.post(f"{API}/search/by-image",
                               files={"file": (target.name, fh, "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["results"][0]["filename"] == target.name


def test_assign_and_detach_faces(client):
    people = client.get(f"{API}/people").json()
    source, target = people[0], people[1]

    faces = client.post(f"{API}/faces/search", json={
        "face_id": source["cover_face_id"], "min_similarity": 0.5}).json()["faces"]
    face_ids = [f["face_id"] for f in faces] + [source["cover_face_id"]]

    assigned = client.post(f"{API}/faces/assign", json={
        "face_ids": face_ids, "person_id": target["person_id"]}).json()
    assert assigned["updated"] == len(face_ids)

    detached = client.post(f"{API}/faces/detach",
                           json={"face_ids": [face_ids[0]]}).json()
    assert detached["updated"] == 1

    unassigned = client.get(f"{API}/faces/unassigned",
                            params={"min_quality": 0.0}).json()["faces"]
    assert face_ids[0] in [f["face_id"] for f in unassigned]


def test_assign_can_create_a_new_person(client):
    before = len(client.get(f"{API}/people").json())
    face = client.get(f"{API}/people").json()[0]["cover_face_id"]

    body = client.post(f"{API}/faces/assign",
                       json={"face_ids": [face], "name": "New Person"}).json()

    assert "person_id" in body
    people = {p["person_id"]: p for p in client.get(f"{API}/people").json()}
    assert people[body["person_id"]]["name"] == "New Person"
    assert len(people) >= before


def test_timeline_and_folders_endpoints(client):
    months = client.get(f"{API}/timeline").json()["months"]
    folders = client.get(f"{API}/folders").json()["folders"]

    assert sum(m["count"] for m in months) == 7
    assert len(folders) == 3  # root, nested, nested/deeper


def test_duplicates_endpoint_finds_the_copied_photo(client, photo_dir, indexed_service,
                                                    indexer):
    import shutil

    shutil.copy(photo_dir / "birthday-cake-candles-20200103.jpg",
                photo_dir / "birthday-cake-candles-copy.jpg")
    indexer.index_directory(photo_dir)
    indexed_service.index.invalidate()

    groups = client.get(f"{API}/admin/duplicates").json()["groups"]
    assert any(len(g["image_ids"]) >= 2 for g in groups)


def test_index_job_runs_in_the_background(client, tmp_path, photo_dir):
    from tests.conftest import make_photo

    make_photo(photo_dir / "late-addition-20250101.jpg", ["alice"])

    job = client.post(f"{API}/admin/index", json={"folder": str(photo_dir)}).json()
    assert job["status"] in ("pending", "running", "done")

    for _ in range(100):
        job = client.get(f"{API}/admin/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)

    assert job["status"] == "done", job.get("error")
    assert job["result"]["added"] == 1
    assert client.get(f"{API}/stats").json()["total_images"] == 9


def test_indexing_a_missing_folder_is_a_400(client):
    response = client.post(f"{API}/admin/index", json={"folder": "/no/such/folder"})
    assert response.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get(f"{API}/admin/jobs/deadbeef").status_code == 404


def test_gzip_is_applied_to_large_json(client):
    response = client.post(f"{API}/search", json={"per_page": 100},
                           headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert "X-Response-Time" in response.headers
