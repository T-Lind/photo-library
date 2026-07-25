"""Scaling behaviour of the browse index at library sizes that matter.

These exercise the NumPy paths directly with a synthetic 200k-photo library.
They are the check that "page 3000 of a 200k library" is a constant-time
operation rather than something that degrades as you scroll — the failure the
previous ``limit(offset + per_page)[offset:]`` approach had.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pytest

from photolib.browse import Filters, LibraryIndex

N = 200_000
PEOPLE = 400


@pytest.fixture(scope="module")
def big_index() -> LibraryIndex:
    """A 200k-photo index populated directly, without a database round trip."""
    rng = np.random.default_rng(1234)
    index = LibraryIndex.__new__(LibraryIndex)

    import threading
    from types import SimpleNamespace

    index._lock = threading.RLock()
    # A stub library whose version never changes, so ensure_fresh() is a no-op
    # and the arrays below stand in for a real 200k-row table.
    index._library = SimpleNamespace(images=SimpleNamespace(version=1))
    index._version = 1
    index._built = True

    index.image_ids = np.arange(N, dtype=np.int64)
    # Twenty years of photos, with 5% missing a capture date.
    base = datetime(2005, 1, 1).timestamp()
    index.taken_ts = base + rng.uniform(0, 20 * 365 * 86400, N)
    index.taken_ts[rng.random(N) < 0.05] = np.nan
    index.added_ts = np.full(N, datetime(2024, 1, 1).timestamp())
    index.lat = np.where(rng.random(N) < 0.4, rng.uniform(-60, 60, N), np.nan)
    index.face_count = rng.integers(0, 4, N).astype(np.int32)
    index.folders = ["/photos/%04d" % (i % 500) for i in range(N)]
    index._row_of_id = {i: i for i in range(N)}

    # Zipf-ish: a few family members dominate, a long tail of strangers.
    weights = 1.0 / np.arange(1, PEOPLE + 1)
    weights /= weights.sum()
    index._rows_by_person = {}
    for person in range(PEOPLE):
        count = max(2, int(weights[person] * N * 2))
        index._rows_by_person[person] = np.unique(
            rng.integers(0, N, count).astype(np.int64))
    index._person_counts = {k: len(v) for k, v in index._rows_by_person.items()}

    keys = np.where(np.isnan(index.taken_ts), -np.inf, index.taken_ts)
    index._order_date_desc = np.lexsort((index.image_ids, keys))[::-1].astype(np.int64)
    return index


def _time(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def test_unfiltered_browse_ordering_is_fast(big_index):
    rows, elapsed = _time(big_index.select, Filters())
    assert rows.shape[0] == N
    assert elapsed < 0.5

    ordered, elapsed = _time(big_index.order, rows, "date_desc")
    assert ordered.shape[0] == N
    assert elapsed < 0.5


def test_deep_pagination_costs_the_same_as_the_first_page(big_index):
    ordered = big_index.order(big_index.select(Filters()), "date_desc")

    def page(n, size=60):
        return ordered[(n - 1) * size:n * size]

    _, first = _time(page, 1)
    _, deep = _time(page, 3000)

    assert page(3000).shape[0] == 60
    # Slicing a NumPy view is O(1); the old approach was O(offset).
    assert deep < max(first * 20, 0.01)


def test_person_filter_at_scale(big_index):
    rows, elapsed = _time(big_index.select, Filters(people_ids=[0]))

    assert rows.shape[0] == big_index._person_counts[0]
    assert elapsed < 0.2


def test_multi_person_all_mode_at_scale(big_index):
    rows, elapsed = _time(
        big_index.select, Filters(people_ids=[0, 1, 2], people_mode="all"))

    assert elapsed < 0.3
    for person in (0, 1, 2):
        assert np.isin(rows, big_index._rows_by_person[person]).all()


def test_date_range_filter_at_scale(big_index):
    rows, elapsed = _time(big_index.select, Filters(
        start_date=datetime(2015, 1, 1), end_date=datetime(2016, 1, 1)))

    assert 0 < rows.shape[0] < N
    assert elapsed < 0.2
    assert not np.isnan(big_index.taken_ts[rows]).any()


def test_combined_filter_and_sort_stays_interactive(big_index):
    def query():
        rows = big_index.select(Filters(
            start_date=datetime(2012, 1, 1),
            people_ids=[0, 1],
            has_location=True))
        return big_index.order(rows, "date_desc")[:60]

    _, elapsed = _time(query)
    # A user-facing search must not feel laggy on a full-size library.
    assert elapsed < 0.5


def test_month_histogram_at_scale(big_index):
    months, elapsed = _time(big_index.month_histogram)

    assert len(months) > 200  # ~20 years
    assert sum(c for _, c in months) == int((~np.isnan(big_index.taken_ts)).sum())
    assert elapsed < 0.5


def test_index_memory_footprint_is_modest(big_index):
    """The whole point is that this fits comfortably in RAM."""
    arrays = (big_index.image_ids, big_index.taken_ts, big_index.added_ts,
              big_index.lat, big_index.face_count, big_index._order_date_desc)
    total = sum(a.nbytes for a in arrays)
    total += sum(a.nbytes for a in big_index._rows_by_person.values())

    assert total < 32 * 1024 * 1024, f"browse index uses {total / 1e6:.1f} MB"


def test_near_duplicate_grouping_scales(big_index):
    """Banded matching must not degenerate into an O(n^2) sweep."""
    from photolib.hashing import group_near_duplicates

    rng = np.random.default_rng(7)
    items = []
    for i in range(50_000):
        h = int(rng.integers(-(2 ** 62), 2 ** 62))
        items.append((i, h))
    # Plant 100 duplicate pairs.
    for i in range(0, 200, 2):
        items[i + 1] = (items[i + 1][0], items[i][1] ^ 0b11)

    groups, elapsed = _time(group_near_duplicates, items, 6)

    assert len(groups) >= 100
    assert elapsed < 10.0
