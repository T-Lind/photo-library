"""Face identity: clustering quality, incremental assignment, corrections."""

from __future__ import annotations

import numpy as np
import pytest

from photolib.db import UNASSIGNED
from photolib.faces.base import score_quality
from photolib.faces.cluster import FaceAssigner, FaceObservation, mutual_knn_components
from photolib.faces.stub import identity_vector
from tests.conftest import make_photo


# ---------------------------------------------------------------------------
# Clustering algorithm
# ---------------------------------------------------------------------------

def _cluster_blob(centre: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = centre + rng.standard_normal((n, centre.size)).astype(np.float32) * spread
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_mutual_knn_separates_distinct_identities():
    """Three well-separated identities must never share a cluster.

    Individual outliers are allowed to fall out as singletons — that is the
    desired behaviour, and it is what feeds the "unassigned faces" review
    queue. What must not happen is one label spanning two identities.
    """
    dim = 64
    blobs = [_cluster_blob(identity_vector(name, dim), 20, 0.10, seed=i)
             for i, name in enumerate("abc", start=1)]
    labels = mutual_knn_components(np.concatenate(blobs), k=8, threshold=0.5)

    groups = [labels[i * 20:(i + 1) * 20] for i in range(3)]
    dominant = [np.bincount(g).argmax() for g in groups]

    assert len(set(dominant)) == 3, "distinct identities were merged"
    for g, main in zip(groups, dominant):
        assert (g == main).sum() >= 18, "an identity was fragmented"
    for i, g in enumerate(groups):
        others = set().union(*(set(o.tolist()) for j, o in enumerate(groups) if j != i))
        assert dominant[i] not in others


def test_mutual_knn_resists_chaining_through_a_bridge_face():
    """A single ambiguous face between two people must not merge them.

    This is the exact failure mode of DBSCAN with a loose eps, which is what
    the previous implementation used.
    """
    dim = 64
    a_centre = identity_vector("chain-a", dim)
    b_centre = identity_vector("chain-b", dim)
    a = _cluster_blob(a_centre, 15, 0.08, seed=11)
    b = _cluster_blob(b_centre, 15, 0.08, seed=12)
    bridge = (a_centre + b_centre)
    bridge = (bridge / np.linalg.norm(bridge)).astype(np.float32)[None, :]

    labels = mutual_knn_components(np.concatenate([a, b, bridge]), k=6, threshold=0.5)

    assert labels[0] != labels[15], "two distinct people were merged"


def test_mutual_knn_handles_trivial_inputs():
    assert mutual_knn_components(np.zeros((0, 8), dtype=np.float32), 5, 0.5).size == 0
    assert mutual_knn_components(np.ones((1, 8), dtype=np.float32), 5, 0.5).tolist() == [0]


def test_quality_score_prefers_large_sharp_faces():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)

    big_sharp = score_quality(sharp, (0, 0, 200, 200), 0.99)
    small_blurry = score_quality(flat, (0, 0, 30, 30), 0.6)

    assert big_sharp > 0.8
    assert small_blurry < 0.35
    assert big_sharp > small_blurry


# ---------------------------------------------------------------------------
# Incremental assignment
# ---------------------------------------------------------------------------

def _observe(image_id: int, identity: str, face_id: int, dim: int = 32,
             quality: float = 0.9, jitter: float = 0.0, seed: int = 0):
    vec = identity_vector(identity, dim)
    if jitter:
        rng = np.random.default_rng(seed)
        vec = vec + rng.standard_normal(dim).astype(np.float32) * jitter
        vec = vec / np.linalg.norm(vec)
    return FaceObservation(image_id=image_id, embedding=vec.astype(np.float32),
                           bbox=(0, 0, 100, 100), det_score=0.99,
                           quality=quality, face_id=face_id)


@pytest.fixture
def empty_library(library, settings):
    from photolib.db import LibraryMeta, SCHEMA_VERSION

    library.create(LibraryMeta(
        schema_version=SCHEMA_VERSION, image_dim=64, face_dim=32,
        embed_backend="stub", embed_model="stub-v1",
        face_backend="stub", face_model="stub-face-v1"))
    return library


def test_same_person_across_photos_gets_one_identity(empty_library):
    assigner = FaceAssigner(empty_library, dim=32,
                            match_threshold=0.6, strong_threshold=0.8)
    faces = [_observe(i, "alice", i, jitter=0.05, seed=i) for i in range(5)]

    assigner.assign(faces)

    assert len({f.person_id for f in faces}) == 1


def test_different_people_get_different_identities(empty_library):
    assigner = FaceAssigner(empty_library, dim=32,
                            match_threshold=0.6, strong_threshold=0.8)
    faces = [_observe(0, "alice", 0), _observe(1, "bob", 1), _observe(2, "carol", 2)]

    assigner.assign(faces)

    assert len({f.person_id for f in faces}) == 3


def test_two_faces_in_one_photo_are_never_the_same_person(empty_library):
    """Same identity vector, same photo — physically impossible, so split them."""
    assigner = FaceAssigner(empty_library, dim=32,
                            match_threshold=0.6, strong_threshold=0.8)
    faces = [_observe(7, "alice", 0, quality=0.9),
             _observe(7, "alice", 1, quality=0.5)]

    assigner.assign(faces)

    assert faces[0].person_id != faces[1].person_id


def test_low_quality_face_never_invents_a_new_person(empty_library):
    assigner = FaceAssigner(empty_library, dim=32,
                            match_threshold=0.6, strong_threshold=0.8)
    junk = _observe(0, "stranger", 0, quality=0.05)

    assigner.assign([junk])

    assert junk.person_id == UNASSIGNED
    assert not assigner.people


def test_low_quality_face_still_joins_a_confident_match(empty_library):
    assigner = FaceAssigner(empty_library, dim=32,
                            match_threshold=0.6, strong_threshold=0.8)
    good = _observe(0, "alice", 0, quality=0.9)
    assigner.assign([good])

    blurry = _observe(1, "alice", 1, quality=0.05)
    assigner.assign([blurry])

    assert blurry.person_id == good.person_id


def test_assignments_persist_and_reload(empty_library):
    assigner = FaceAssigner(empty_library, dim=32,
                            match_threshold=0.6, strong_threshold=0.8)
    first = _observe(0, "alice", 0)
    assigner.assign([first])
    created, _ = assigner.flush()
    assert created == 1

    # A fresh assigner (i.e. the next indexing run) must recognise her.
    reloaded = FaceAssigner(empty_library, dim=32,
                            match_threshold=0.6, strong_threshold=0.8)
    later = _observe(1, "alice", 1, jitter=0.05, seed=9)
    reloaded.assign([later])

    assert later.person_id == first.person_id


# ---------------------------------------------------------------------------
# End-to-end through the indexer + service
# ---------------------------------------------------------------------------

def test_indexing_groups_each_person_once(indexed_service):
    people = indexed_service.list_people()
    # Four distinct markers across the sample library.
    assert len(people) == 4
    counts = sorted(p["photo_count"] for p in people)
    assert counts == [2, 2, 3, 3]


def test_incremental_run_recognises_a_known_person(indexed_service, indexer, photo_dir):
    alice = max(indexed_service.list_people(), key=lambda p: p["photo_count"])
    before = alice["photo_count"]

    make_photo(photo_dir / "picnic-park-20240808.jpg", ["alice"])
    indexer.index_directory(photo_dir)
    indexed_service.index.invalidate()

    after = {p["person_id"]: p for p in indexed_service.list_people()}
    assert len(after) == 4, "a known face should not create a fifth person"
    # Alice is whoever gained the photo.
    gained = [p for p in after.values() if p["photo_count"] > before]
    assert gained or after[alice["person_id"]]["photo_count"] == before + 1


def test_face_search_finds_the_same_person_elsewhere(indexed_service):
    people = indexed_service.list_people()
    person = next(p for p in people if p["photo_count"] > 1)
    faces = indexed_service.search_faces_by_face(person["cover_face_id"], limit=10,
                                                 min_similarity=0.5)

    assert faces, "searching by face returned nothing"
    assert all(f["person_id"] == person["person_id"] for f in faces)


def test_manual_assignment_moves_faces_and_updates_photos(indexed_service):
    people = {p["person_id"]: p for p in indexed_service.list_people()}
    source, target = list(people)[0], list(people)[1]

    face_ids = [f["face_id"] for f in indexed_service.search_faces_by_face(
        people[source]["cover_face_id"], limit=50, min_similarity=0.5)]
    face_ids.append(people[source]["cover_face_id"])

    result = indexed_service.assign_faces(face_ids, target)
    indexed_service.index.invalidate()

    assert result["updated"] == len(face_ids)
    remaining = {p["person_id"] for p in indexed_service.list_people()}
    assert source not in remaining or indexed_service.get_person(source)["photo_count"] == 0


def test_detaching_a_face_removes_the_person_from_the_photo(indexed_service):
    person = next(p for p in indexed_service.list_people() if p["photo_count"] >= 2)
    faces = indexed_service.search_faces_by_face(
        person["cover_face_id"], limit=50, min_similarity=0.5)
    victim = faces[0]

    indexed_service.detach_faces([victim["face_id"]])
    indexed_service.index.invalidate()

    details = indexed_service.image_details(victim["image_id"])
    assert person["person_id"] not in details["people_ids"]


def test_merging_people_reattributes_every_photo(indexed_service):
    people = indexed_service.list_people()
    source, target = people[0], people[1]
    # Photos where both already appear must not be double-counted.
    expected = len({
        img for person in (source, target)
        for img in _photo_ids_of(indexed_service, person["person_id"])})

    merged = indexed_service.merge_people(source["person_id"], target["person_id"])
    indexed_service.index.invalidate()

    assert merged["person_id"] == target["person_id"]
    assert indexed_service.get_person(target["person_id"])["photo_count"] == expected
    assert len(indexed_service.list_people()) == 3


def _photo_ids_of(service, person_id: int) -> set:
    from photolib.browse import Filters

    page = service.search(None, Filters(people_ids=[person_id]),
                          sort="date_desc", page=1, per_page=200)
    return {r["image_id"] for r in page.results}


def test_deleting_a_person_frees_their_faces(indexed_service):
    person = indexed_service.list_people()[0]

    indexed_service.delete_person(person["person_id"])
    indexed_service.index.invalidate()

    assert len(indexed_service.list_people()) == 3
    unassigned = indexed_service.library.faces.count_rows(f"person_id = {UNASSIGNED}")
    assert unassigned > 0


def test_recluster_preserves_confirmed_identities(indexed_service):
    from photolib.faces.cluster import recluster

    person = next(p for p in indexed_service.list_people() if p["photo_count"] >= 2)
    indexed_service.rename_person(person["person_id"], "Alice Confirmed")
    indexed_service.assign_faces([person["cover_face_id"]], person["person_id"])

    recluster(indexed_service.library, dim=32, threshold=0.6, knn=8,
              min_cluster_size=1)
    indexed_service._resync_all_image_people()
    indexed_service.index.invalidate()
    indexed_service._people_cache = None

    names = {p["person_id"]: p["name"] for p in indexed_service.list_people()}
    assert names.get(person["person_id"]) == "Alice Confirmed"
