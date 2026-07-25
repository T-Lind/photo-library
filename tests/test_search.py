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
