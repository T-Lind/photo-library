"""Search and browse behaviour: ranking, filtering, sorting, pagination."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from photolib.browse import Filters


def _names(page):
    return [r["filename"] for r in page.results]


def test_natural_language_query_ranks_the_right_photo_first(indexed_service):
    page = indexed_service.search("sunset at the beach on holiday", Filters(),
                                  sort="relevance")

    assert page.scored
    assert page.results[0]["filename"].startswith("beach-sunset-holiday")
    assert page.results[0]["score"] > page.results[-1]["score"]


def test_different_queries_surface_different_photos(indexed_service):
    snow = indexed_service.search("skiing in winter snow", Filters(), sort="relevance")
    cake = indexed_service.search("birthday cake with candles", Filters(),
                                  sort="relevance")

    assert "snow-winter-skiing" in snow.results[0]["filename"]
    assert "birthday-cake-candles" in cake.results[0]["filename"]


def test_empty_query_browses_everything_newest_first(indexed_service):
    page = indexed_service.search(None, Filters(), sort="date_desc")

    assert page.total == 8
    assert not page.scored
    dated = [r["taken_at"] for r in page.results if r["taken_at"]]
    assert dated == sorted(dated, reverse=True)
    # The photo with no capture date sorts last, not to 1970.
    assert page.results[-1]["taken_at"] is None


def test_date_ascending_reverses_the_order(indexed_service):
    desc = indexed_service.search(None, Filters(), sort="date_desc")
    asc = indexed_service.search(None, Filters(), sort="date_asc")

    desc_dated = [r["filename"] for r in desc.results if r["taken_at"]]
    asc_dated = [r["filename"] for r in asc.results if r["taken_at"]]
    assert asc_dated == list(reversed(desc_dated))


def test_date_range_filter_excludes_undated_photos(indexed_service):
    page = indexed_service.search(
        None,
        Filters(start_date=datetime(2018, 1, 1), end_date=datetime(2019, 12, 31)),
        sort="date_desc")

    assert page.total == 3
    # An undated photo cannot be shown to fall inside a range, so it is out.
    # The previous implementation's "OR date IS NULL" matched it every time.
    assert all(r["taken_at"] for r in page.results)


def test_person_filter_returns_only_their_photos(indexed_service):
    person = max(indexed_service.list_people(), key=lambda p: p["photo_count"])
    page = indexed_service.search(None, Filters(people_ids=[person["person_id"]]))

    assert page.total == person["photo_count"]
    assert all(person["person_id"] in r["people_ids"] for r in page.results)


def test_people_mode_all_requires_every_person_present(indexed_service):
    people = indexed_service.list_people()
    pair = [people[0]["person_id"], people[1]["person_id"]]

    any_page = indexed_service.search(None, Filters(people_ids=pair, people_mode="any"))
    all_page = indexed_service.search(None, Filters(people_ids=pair, people_mode="all"))

    assert all_page.total <= any_page.total
    assert all(set(pair).issubset(r["people_ids"]) for r in all_page.results)


def test_has_faces_filter(indexed_service):
    with_faces = indexed_service.search(None, Filters(has_faces=True))
    without = indexed_service.search(None, Filters(has_faces=False))

    assert with_faces.total == 7
    assert without.total == 1
    assert without.results[0]["filename"] == "city-street-night-lights.jpg"


def test_folder_filter_matches_subtrees(indexed_service, photo_dir):
    page = indexed_service.search(None, Filters(folder=str(photo_dir / "nested")))

    assert page.total == 2  # includes nested/deeper


def test_semantic_search_respects_filters(indexed_service):
    person = max(indexed_service.list_people(), key=lambda p: p["photo_count"])
    page = indexed_service.search("beach sunset", Filters(people_ids=[person["person_id"]]),
                                  sort="relevance")

    assert page.total > 0
    assert all(person["person_id"] in r["people_ids"] for r in page.results)


def test_empty_filter_does_not_load_model(indexed_service, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Empty filter must not embed or search vectors")

    monkeypatch.setattr(indexed_service.embedder, "embed_texts", unexpected)
    page = indexed_service.search("sunset", Filters(people_ids=[999999]),
                                  sort="relevance")
    assert page.total == 0


def test_person_pair_and_text_rank_within_the_subset(indexed_service):
    service = indexed_service
    photos = service.search(None, Filters()).results
    pair = next(p["people_ids"] for p in photos
                if p["filename"] == "beach-sunset-holiday-20180705.jpg")
    page = service.search("sunset portrait", Filters(people_ids=pair,
                          people_mode="all"), sort="relevance")
    assert page.total == 1
    assert page.results[0]["filename"] == "beach-sunset-holiday-20180705.jpg"


def test_subset_ranking_matches_exact_cosine(indexed_service):
    service = indexed_service
    allowed = service.index.select(Filters(has_faces=True))
    vector = service.embedder.embed_texts(["beach sunset"])[0]
    rows, scores = service._vector_rows(vector, allowed, limit=3)
    stored = service.library.images.to_lance().to_table(
        columns=["image_id", "vector"]).to_pylist()
    allowed_ids = set(service.index.ids_of(allowed))
    expected = sorted([
        (r["image_id"], float(np.dot(r["vector"], vector) /
                              (np.linalg.norm(r["vector"]) * np.linalg.norm(vector))))
        for r in stored if r["image_id"] in allowed_ids
    ], key=lambda item: (-item[1], item[0]))[:3]
    assert service.index.ids_of(rows) == [i for i, _ in expected]
    assert np.allclose(scores, [s for _, s in expected], atol=1e-6)

    filtered, filtered_scores = service._vector_rows(
        vector, allowed, min_score=float(scores[-1]) - 1e-5,
        exclude_image_id=expected[0][0], limit=2)
    assert service.index.ids_of(filtered) == [i for i, _ in expected[1:]]
    assert len(filtered_scores) == 2


def test_random_sort_returns_a_reusable_seed(indexed_service):
    first = indexed_service.search(None, Filters(), sort="random", page=1, per_page=4)
    assert first.seed is not None

    # Reusing the returned seed keeps a shuffle stable while paging.
    again = indexed_service.search(None, Filters(), sort="random", page=1,
                                   per_page=4, seed=first.seed)
    assert [r["image_id"] for r in first.results] == \
           [r["image_id"] for r in again.results]
    # It is a shuffle, not a re-sort: same set, generally different order.
    every = indexed_service.search(None, Filters(), sort="date_desc", per_page=50)
    assert {r["image_id"] for r in first.results} <= \
           {r["image_id"] for r in every.results}


def test_annotation_filter_reflects_a_new_favorite_without_reindex(indexed_service):
    target = indexed_service.search(None, Filters()).results[0]["image_id"]

    indexed_service.annotate(target, favorite=True, rating=5)

    favorites = indexed_service.search(None, Filters(favorites_only=True))
    assert [r["image_id"] for r in favorites.results] == [target]
    rated = indexed_service.search(None, Filters(min_rating=5))
    assert target in [r["image_id"] for r in rated.results]


def test_search_mode_semantic_skips_text_matching(indexed_service, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("text search must not run in semantic mode")

    monkeypatch.setattr(indexed_service, "_text_rows", boom)
    page = indexed_service.search("beach sunset", Filters(), sort="relevance",
                                  search_mode="semantic")

    assert page.total > 0
    assert all(not r.get("text_match") for r in page.results)


def test_search_reports_a_clear_model_mismatch(indexed_service, monkeypatch):
    import pytest as _pytest

    from photolib.db import SchemaMismatch

    class WrongEmbedder:
        backend = "stub"
        model_name = "some-other-model"
        dim = 3

        def embed_texts(self, texts):  # pragma: no cover - must not be reached
            raise AssertionError("a mismatched model must not be used")

    monkeypatch.setattr(indexed_service, "_embedder", WrongEmbedder())
    with _pytest.raises(SchemaMismatch) as exc:
        indexed_service.search("beach", Filters(), sort="relevance")
    assert "indexed with" in str(exc.value)


def test_unknown_search_mode_is_rejected(indexed_service):
    import pytest as _pytest

    with _pytest.raises(ValueError):
        indexed_service.search("beach", Filters(), search_mode="magic")


def test_pagination_is_stable_and_non_overlapping(indexed_service):
    first = indexed_service.search(None, Filters(), sort="date_desc",
                                   page=1, per_page=3)
    second = indexed_service.search(None, Filters(), sort="date_desc",
                                    page=2, per_page=3)
    third = indexed_service.search(None, Filters(), sort="date_desc",
                                   page=3, per_page=3)

    assert first.total == second.total == 8
    assert len(first.results) == len(second.results) == 3
    assert len(third.results) == 2
    ids = [r["image_id"] for r in first.results + second.results + third.results]
    assert len(set(ids)) == 8


def test_page_past_the_end_is_empty_not_an_error(indexed_service):
    page = indexed_service.search(None, Filters(), page=99, per_page=20)
    assert page.results == []
    assert page.total == 8


def test_min_score_drops_weak_matches(indexed_service):
    loose = indexed_service.search("beach", Filters(), sort="relevance")
    strict = indexed_service.search("beach", Filters(), sort="relevance",
                                    min_score=0.5)

    assert strict.total < loose.total


def test_similar_images_excludes_the_query_image(indexed_service):
    page = indexed_service.search(None, Filters())
    target = page.results[0]["image_id"]

    similar = indexed_service.similar_images(target, limit=5)

    assert similar
    assert all(r["image_id"] != target for r in similar)


def test_similar_images_finds_the_matching_pair(indexed_service):
    page = indexed_service.search("beach sunset holiday", Filters(), sort="relevance")
    first = page.results[0]

    similar = indexed_service.similar_images(first["image_id"], limit=3)

    assert similar[0]["filename"].startswith("beach-sunset-holiday")


def test_reverse_image_search_finds_the_original(indexed_service, photo_dir):
    target = photo_dir / "mountain-hiking-trail-20190812.jpg"
    page = indexed_service.search_by_image(target, Filters(), per_page=5)

    assert page.results[0]["filename"] == target.name


def test_timeline_buckets_by_month(indexed_service):
    months = indexed_service.timeline()

    assert ("2018-07", 2) in [(m["month"], m["count"]) for m in months]
    assert sum(m["count"] for m in months) == 7  # one photo has no date


def test_stats_reflect_the_library(indexed_service):
    stats = indexed_service.stats()

    assert stats["ready"] is True
    assert stats["total_images"] == 8
    assert stats["total_people"] == 4
    assert stats["total_faces"] == 10
    assert stats["images_without_date"] == 1
    assert stats["embed_model"] == "stub:stub-v1"


def test_empty_library_reports_not_ready(service):
    assert service.stats() == {"ready": False, "total_images": 0,
                               "total_people": 0, "total_faces": 0}
