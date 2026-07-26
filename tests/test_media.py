"""Videos and AVIF join the library through the existing image pipeline.

A video is represented by its poster frame, so search, thumbnails, people,
albums, and trash all work on it unchanged. Test videos are synthesized with
the ffmpeg bundled in imageio-ffmpeg — solid-colour frames, with the filename
carrying the searchable words (the stub-embedder convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg")

API = "/api/v1"


def make_video(path: Path, seconds: float = 2.0, fps: int = 10,
               size=(64, 48), color=(200, 60, 60),
               creation_time: str | None = None) -> Path:
    import imageio_ffmpeg

    path.parent.mkdir(parents=True, exist_ok=True)
    output_params = []
    if creation_time:
        output_params += ["-metadata", f"creation_time={creation_time}"]
    writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps,
                                         codec="libx264",
                                         output_params=output_params)
    writer.send(None)
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:, :] = color
    for _ in range(int(seconds * fps)):
        writer.send(frame.tobytes())
    writer.close()
    return path


@pytest.fixture
def media_dir(photo_dir: Path) -> Path:
    """The sample photo library plus two videos."""
    make_video(photo_dir / "puppy-fetch-video-20230530.mp4",
               creation_time="2023-05-30T08:00:00Z")
    make_video(photo_dir / "nested" / "concert-stage-crowd.mp4",
               color=(40, 30, 90))
    return photo_dir


@pytest.fixture
def media_client(client, indexer, indexed_service, media_dir):
    """The standard client, after an incremental rescan picks up the videos."""
    indexer.index_directory(media_dir)
    indexed_service.index.invalidate()
    return client


# -- indexing ---------------------------------------------------------------

def test_videos_index_with_type_and_duration(indexer, indexed_service, media_dir):
    from photolib.browse import Filters

    stats = indexer.index_directory(media_dir)
    assert stats.added == 2  # the 8 photos are already indexed and skipped
    indexed_service.index.invalidate()

    page = indexed_service.search(None, Filters(media="video"))
    assert page.total == 2
    puppy = next(r for r in page.results if r["filename"].startswith("puppy"))
    assert puppy["media_type"] == "video"
    assert 1500 <= puppy["duration_ms"] <= 2500


def test_video_ranks_for_its_words(indexer, indexed_service, media_dir):
    from photolib.browse import Filters

    indexer.index_directory(media_dir)
    indexed_service.index.invalidate()
    page = indexed_service.search("puppy fetch video", Filters(),
                                  sort="relevance")
    assert page.results[0]["filename"] == "puppy-fetch-video-20230530.mp4"


def test_capture_time_comes_from_the_container(tmp_path):
    from photolib.exif import read_metadata

    tagged = make_video(tmp_path / "clip-tagged.mp4",
                        creation_time="2023-05-30T08:00:00Z")
    meta = read_metadata(tagged)
    # Stored UTC, converted to local wall-clock — the date can shift by one
    # day in extreme timezones, but never out of May 2023.
    assert meta.taken_at is not None
    assert (meta.taken_at.year, meta.taken_at.month) == (2023, 5)

    untagged = make_video(tmp_path / "clip-plain.mp4")
    assert read_metadata(untagged).taken_at is None

    named = make_video(tmp_path / "ski-trip-20220211.mp4")
    assert read_metadata(named).taken_at.year == 2022  # filename fallback


def test_probe_reports_size_and_duration(tmp_path):
    from photolib.imageio import probe_video, read_size

    clip = make_video(tmp_path / "clip.mp4", seconds=1.5, size=(64, 48))
    info = probe_video(clip)
    assert (info.width, info.height) == (64, 48)
    assert 1000 <= info.duration_ms <= 2000
    assert read_size(clip) == (64, 48)


# -- API --------------------------------------------------------------------

def _video_id(client) -> int:
    body = client.post(f"{API}/search",
                       json={"media": "video", "per_page": 5}).json()
    assert body["results"], "no videos in the test library"
    return body["results"][0]["image_id"]


def test_file_endpoint_supports_range_requests(media_client):
    """<video> seeking depends on 206 partial responses."""
    video_id = _video_id(media_client)
    r = media_client.get(f"{API}/images/{video_id}",
                         headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 0-99/")
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) == 100

    full = media_client.get(f"{API}/images/{video_id}")
    assert full.status_code == 200


def test_video_thumbnail_is_its_poster_frame(media_client):
    video_id = _video_id(media_client)
    r = media_client.get(f"{API}/images/{video_id}/thumbnail?size=grid")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"


def test_details_carry_media_fields(media_client):
    video_id = _video_id(media_client)
    details = media_client.get(f"{API}/images/{video_id}/details").json()
    assert details["media_type"] == "video"
    assert details["duration_ms"] > 0


def test_media_filter_separates_photos_from_videos(media_client):
    images = media_client.post(f"{API}/search",
                               json={"media": "image", "per_page": 50}).json()
    assert images["total"] == 8
    assert all(r["media_type"] == "image" for r in images["results"])

    everything = media_client.post(f"{API}/search",
                                   json={"per_page": 50}).json()
    assert everything["total"] == 10


def test_stats_count_videos(media_client):
    stats = media_client.get(f"{API}/stats").json()
    assert stats["total_videos"] == 2
    assert stats["total_images"] == 10


# -- migration ---------------------------------------------------------------

def test_old_library_gains_media_columns_in_place(indexed_service, library):
    """A pre-video library is upgraded by add_columns, not a re-index."""
    from photolib.browse import Filters
    from photolib.service import PhotoService

    images_before = library.images.count_rows(None)
    library.images.drop_columns(["media_type", "duration_ms"])
    assert "media_type" not in library.images.schema.names

    # Constructing a service is the normal migration trigger.
    fresh = PhotoService(settings=indexed_service.settings, library=library)
    assert "media_type" in library.images.schema.names
    assert library.images.count_rows(None) == images_before

    page = fresh.search(None, Filters(media="image"))
    assert page.total == images_before  # every legacy row is an image


def test_browse_tolerates_an_unmigrated_library(indexed_service, library):
    """Read-only browsing must not require the migration write."""
    from photolib.browse import Filters, LibraryIndex

    library.images.drop_columns(["media_type", "duration_ms"])
    index = LibraryIndex(library)
    index.ensure_fresh()
    assert index.select(Filters(media="video")).size == 0
    assert index.select(Filters()).size == 8


# -- AVIF ---------------------------------------------------------------------

def test_avif_files_are_indexed(indexer, indexed_service, photo_dir):
    from photolib.browse import Filters
    from photolib.imageio import avif_supported

    if not avif_supported():
        pytest.skip("this Pillow build lacks AVIF support")

    from PIL import Image

    array = np.zeros((60, 80, 3), dtype=np.uint8)
    array[:, :] = (90, 140, 210)
    Image.fromarray(array).save(photo_dir / "harbor-boats-morning.avif")

    indexer.index_directory(photo_dir)
    indexed_service.index.invalidate()
    page = indexed_service.search("harbor boats morning", Filters(),
                                  sort="relevance")
    assert page.results[0]["filename"] == "harbor-boats-morning.avif"
    assert page.results[0]["media_type"] == "image"
