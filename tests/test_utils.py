"""Hashing, EXIF parsing, thumbnails, and config."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from photolib.exif import date_from_filename, read_metadata
from photolib.hashing import content_hash, group_near_duplicates, hamming, phash_file
from tests.conftest import make_photo


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def test_content_hash_matches_for_identical_bytes(tmp_path):
    a = make_photo(tmp_path / "a.jpg", ["alice"])
    b = tmp_path / "b.jpg"
    b.write_bytes(a.read_bytes())

    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(make_photo(tmp_path / "c.jpg", ["bob"]))


def test_phash_survives_recompression(tmp_path):
    from PIL import Image

    original = make_photo(tmp_path / "orig.jpg", ["alice"], tint=(180, 90, 40))
    recompressed = tmp_path / "recompressed.jpg"
    with Image.open(original) as img:
        img.resize((img.width // 2, img.height // 2)).save(
            recompressed, "JPEG", quality=40)

    # A re-encoded, half-size copy is the classic "shared via chat" duplicate.
    assert hamming(phash_file(original), phash_file(recompressed)) <= 6


def test_phash_differs_for_different_pictures(tmp_path):
    a = make_photo(tmp_path / "a.jpg", ["alice"], tint=(255, 0, 0))
    b = make_photo(tmp_path / "b.jpg", ["bob"], tint=(0, 0, 255),
                   size=(240, 180))

    assert hamming(phash_file(a), phash_file(b)) > 6


def test_phash_handles_the_int64_sign_boundary():
    """Hashes with the top bit set are stored negative; comparisons must cope."""
    negative = -6172840429334713785
    assert hamming(negative, negative) == 0
    assert hamming(negative, negative ^ 0b111) == 3


def test_group_near_duplicates_uses_banding():
    base = 0x0123456789ABCDEF
    items = [(1, base), (2, base ^ 0b11), (3, 0x7EDCBA9876543210), (4, base ^ 0b1)]

    groups = group_near_duplicates(items, max_distance=6)

    assert len(groups) == 1
    assert groups[0] == [1, 2, 4]


def test_group_near_duplicates_on_empty_input():
    assert group_near_duplicates([]) == []


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stem,expected", [
    ("IMG_20180704_153000", datetime(2018, 7, 4, 15, 30, 0)),
    ("2019-08-12 09.15.30", datetime(2019, 8, 12, 9, 15, 30)),
    ("PXL_20220211_101112345", datetime(2022, 2, 11, 10, 11, 12)),
    ("holiday-20200103", datetime(2020, 1, 3, 0, 0, 0)),
])
def test_dates_are_recovered_from_filenames(stem, expected, tmp_path):
    assert date_from_filename(Path(f"{stem}.jpg")) == expected


@pytest.mark.parametrize("stem", ["vacation", "IMG_1234", "20189901", "DSC00042"])
def test_non_dates_in_filenames_are_ignored(stem):
    assert date_from_filename(Path(f"{stem}.jpg")) is None


def test_metadata_of_a_photo_without_exif(tmp_path):
    path = make_photo(tmp_path / "plain-photo.jpg")
    meta = read_metadata(path)

    assert meta.taken_at is None
    assert meta.lat is None and meta.lon is None
    assert not meta.has_location


def test_metadata_never_raises_on_a_broken_file(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image at all")

    meta = read_metadata(broken)  # must not raise
    assert meta.taken_at is None


def test_null_island_coordinates_are_rejected():
    from photolib.exif import _valid_coords

    # (0, 0) is always a broken GPS write, never a real photo location.
    assert _valid_coords(0.0, 0.0) == (None, None)
    assert _valid_coords(51.5, -0.12) == (51.5, -0.12)
    assert _valid_coords(91.0, 0.0) == (None, None)


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

def test_thumbnails_are_sharded_and_regenerate_when_stale(tmp_path):
    import os
    import time

    from photolib.thumbnails import ThumbnailCache

    source = make_photo(tmp_path / "src.jpg", ["alice"], size=(1200, 900))
    cache = ThumbnailCache(tmp_path / "cache")

    path = cache.get_thumbnail(7, source, "grid")
    assert path.exists()
    # Sharded by id so no directory ever holds 200k entries.
    assert path.parent.name == f"{7 % 256:02x}"

    first_mtime = path.stat().st_mtime
    cache.get_thumbnail(7, source, "grid")
    assert path.stat().st_mtime == first_mtime, "cached thumbnail was rebuilt"

    make_photo(tmp_path / "src.jpg", ["bob"], size=(1200, 900))
    os.utime(source, (time.time() + 10, time.time() + 10))
    assert cache.get_thumbnail(7, source, "grid").stat().st_mtime != first_mtime


def test_thumbnail_respects_the_long_edge(tmp_path):
    from PIL import Image

    from photolib.thumbnails import SIZES, ThumbnailCache

    source = make_photo(tmp_path / "wide.jpg", size=(2000, 500))
    cache = ThumbnailCache(tmp_path / "cache")

    with Image.open(cache.get_thumbnail(1, source, "grid")) as img:
        assert max(img.size) == SIZES["grid"]


def test_face_crop_includes_padding_and_clamps_to_bounds(tmp_path):
    from PIL import Image

    from photolib.thumbnails import ThumbnailCache

    source = make_photo(tmp_path / "portrait.jpg", ["alice"], size=(400, 300))
    cache = ThumbnailCache(tmp_path / "cache")

    # A box flush against the top-left corner must not produce a negative crop.
    path = cache.get_face_crop(3, source, (0, 0, 40, 40))
    with Image.open(path) as img:
        assert img.width > 0 and img.height > 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_cors_origins_accept_a_comma_separated_string(monkeypatch):
    from photolib.config import get_settings, reset_settings_cache

    monkeypatch.setenv("PHOTO_CORS_ORIGINS",
                       "http://localhost:3000, http://192.168.1.20:3000")
    reset_settings_cache()
    try:
        assert get_settings().cors_origins == [
            "http://localhost:3000", "http://192.168.1.20:3000"]
    finally:
        reset_settings_cache()


def test_ivf_pq_parameters_scale_with_the_library():
    from photolib.db import ivf_pq_params

    small_partitions, small_sub = ivf_pq_params(10_000, 768)
    big_partitions, big_sub = ivf_pq_params(200_000, 768)

    assert small_partitions == 100
    assert big_partitions > small_partitions
    # Sub-vector count must divide the embedding dimension exactly.
    assert 768 % small_sub == 0
    assert 768 % big_sub == 0


def test_ivf_pq_handles_awkward_dimensions():
    from photolib.db import ivf_pq_params

    for dim in (128, 512, 640, 1152, 1000):
        _, sub = ivf_pq_params(50_000, dim)
        assert dim % sub == 0


def test_supported_extensions_cover_phone_and_camera_formats():
    from photolib.imageio import SUPPORTED_EXTS

    for ext in (".jpg", ".heic", ".png", ".webp", ".cr2", ".nef", ".dng", ".arw"):
        assert ext in SUPPORTED_EXTS


def test_recursive_scan_skips_hidden_and_sidecar_files(tmp_path):
    from photolib.imageio import iter_image_files

    root = tmp_path / "lib"
    make_photo(root / "good.jpg")
    make_photo(root / "sub" / "also-good.jpg")
    make_photo(root / ".hidden" / "ignored.jpg")
    (root / "._sidecar.jpg").write_bytes(b"junk")
    (root / "notes.txt").write_text("hello")

    found = {p.name for p in iter_image_files(root)}
    assert found == {"good.jpg", "also-good.jpg"}
