"""In-memory browse index — the reason this stays fast at 200k photos.

Every "show me a page of photos" question (filter by date, by person, by
whether there's a GPS fix; sort by capture time; jump to page 400) is
answered from a handful of NumPy arrays instead of the database. Only the
~60 rows actually being rendered are ever read from LanceDB.

The cost is small and fixed: for 200k photos this is roughly 6 MB of arrays,
built in well under a second, and rebuilt only when the table version
changes. The alternative — what the previous version did — was to ask
LanceDB for ``offset + per_page`` rows and throw away the first ``offset`` of
them, which gets linearly slower the deeper you page and cannot sort at all.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .db import Library, OCR

logger = logging.getLogger(__name__)

SORT_OPTIONS = ("date_desc", "date_asc", "added_desc", "relevance", "random")


@dataclass
class Filters:
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    people_ids: Sequence[int] = ()
    # "any": photos containing at least one of them (default).
    # "all": photos containing every one of them — "me and both kids".
    people_mode: str = "any"
    has_location: Optional[bool] = None
    has_faces: Optional[bool] = None
    folder: Optional[str] = None
    camera: Optional[str] = None
    untagged_only: bool = False
    # "Photos taken near here": centre + radius in km. All three or nothing.
    near_lat: Optional[float] = None
    near_lon: Optional[float] = None
    near_km: float = 1.0
    # "image" or "video"; None = both.
    media: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return (self.start_date is None and self.end_date is None
                and not self.people_ids and self.has_location is None
                and self.has_faces is None and not self.folder
                and not self.camera and not self.untagged_only
                and self.near_lat is None and self.media is None)


class LibraryIndex:
    """Columnar snapshot of the images table, refreshed on version change."""

    def __init__(self, library: Library):
        self._library = library
        self._lock = threading.RLock()
        self._version: Optional[int] = None
        self._ocr_handle = None
        self._ocr_version: Optional[int] = None
        self._built = False

        self.image_ids = np.zeros(0, dtype=np.int64)
        self.taken_ts = np.zeros(0, dtype=np.float64)     # NaN = unknown date
        self.added_ts = np.zeros(0, dtype=np.float64)
        self.lat = np.zeros(0, dtype=np.float64)
        self.lon = np.zeros(0, dtype=np.float64)
        self.face_count = np.zeros(0, dtype=np.int32)
        self.is_video = np.zeros(0, dtype=bool)
        self.folders: List[str] = []
        self.cameras: List[str] = []
        self.ocr_text: List[str] = []          # lowercase; "" = none/unscanned
        self.ocr_scanned = np.zeros(0, dtype=bool)
        self._row_of_id: Dict[int, int] = {}
        self._rows_by_person: Dict[int, np.ndarray] = {}
        self._person_counts: Dict[int, int] = {}
        self._order_date_desc = np.zeros(0, dtype=np.int64)

    # -- lifecycle -------------------------------------------------------
    def ensure_fresh(self) -> None:
        with self._lock:
            try:
                version = self._library.images.version
            except Exception:
                version = None
            if self._built and version == self._version and self._ocr_fresh():
                return
            self._rebuild(version)

    def _ocr_fresh(self) -> bool:
        """Is the cached OCR text still current?

        Uses the table handle captured at rebuild time — ensure_fresh runs on
        every row lookup, so this must never touch the filesystem the way
        ``table_names()``/``open_table()`` do. A brand-new OCR table is
        picked up via ``invalidate()``, which every writer already calls.
        """
        if self._ocr_handle is None:
            return True
        try:
            return self._ocr_handle.version == self._ocr_version
        except Exception:
            return False

    def invalidate(self) -> None:
        with self._lock:
            self._built = False
            self._version = None

    def _rebuild(self, version: Optional[int]) -> None:
        images = self._library.images
        columns = ["image_id", "taken_at", "added_at", "lat", "lon",
                   "face_count", "people_ids", "folder", "camera"]
        # A library indexed before video support lacks this column until its
        # next indexing run migrates it; browsing must not require a write.
        has_media = "media_type" in images.schema.names
        if has_media:
            columns.append("media_type")
        table = images.to_lance().to_table(columns=columns)
        n = table.num_rows
        logger.debug("Rebuilding browse index over %d images", n)

        self.image_ids = np.asarray(table["image_id"].to_numpy(zero_copy_only=False),
                                    dtype=np.int64)
        self.taken_ts = _timestamps(table["taken_at"])
        self.added_ts = _timestamps(table["added_at"])
        self.lat = _floats(table["lat"])
        self.lon = _floats(table["lon"])
        self.face_count = np.nan_to_num(
            _floats(table["face_count"]), nan=0.0).astype(np.int32)
        self.folders = [f or "" for f in table["folder"].to_pylist()]
        self.cameras = [c or "" for c in table["camera"].to_pylist()]
        if has_media:
            self.is_video = np.fromiter(
                (m == "video" for m in table["media_type"].to_pylist()),
                dtype=bool, count=n)
        else:
            self.is_video = np.zeros(n, dtype=bool)

        self._row_of_id = {int(v): i for i, v in enumerate(self.image_ids)}

        by_person: Dict[int, List[int]] = {}
        for row, ids in enumerate(table["people_ids"].to_pylist()):
            for pid in ids or []:
                by_person.setdefault(int(pid), []).append(row)
        self._rows_by_person = {k: np.asarray(v, dtype=np.int64)
                                for k, v in by_person.items()}
        self._person_counts = {k: len(v) for k, v in by_person.items()}

        self._load_ocr(n)

        # Undated photos sort last rather than to 1970, which is where a
        # missing EXIF timestamp used to put them.
        keys = np.where(np.isnan(self.taken_ts), -np.inf, self.taken_ts)
        self._order_date_desc = np.lexsort((self.image_ids, keys))[::-1].astype(np.int64)

        self._version = version
        self._built = True

    def _load_ocr(self, n: int) -> None:
        """Attach extracted text to image rows. Absent table = no text."""
        self.ocr_text = [""] * n
        self.ocr_scanned = np.zeros(n, dtype=bool)
        self._ocr_handle = None
        self._ocr_version = None
        try:
            if not self._library.has_ocr():
                return
            self._ocr_handle = self._library.ocr
            self._ocr_version = self._ocr_handle.version
            table = self._ocr_handle.to_lance().to_table(
                columns=["image_id", "text"])
        except Exception as exc:
            logger.debug("OCR text not loaded: %s", exc)
            return
        for image_id, text in zip(table["image_id"].to_pylist(),
                                  table["text"].to_pylist()):
            row = self._row_of_id.get(int(image_id))
            if row is None:
                continue
            self.ocr_scanned[row] = True
            if text:
                self.ocr_text[row] = str(text).lower()

    def text_match(self, tokens: Sequence[str],
                   allowed: Optional[np.ndarray] = None) -> np.ndarray:
        """Rows whose extracted text contains every token (case-insensitive)."""
        self.ensure_fresh()
        if not tokens:
            return np.zeros(0, dtype=np.int64)
        wanted = [t.lower() for t in tokens if t]
        rows = (range(len(self.ocr_text)) if allowed is None
                else (int(r) for r in allowed))
        hits = [row for row in rows
                if self.ocr_text[row]
                and all(t in self.ocr_text[row] for t in wanted)]
        return np.asarray(hits, dtype=np.int64)

    # -- accessors -------------------------------------------------------
    @property
    def count(self) -> int:
        self.ensure_fresh()
        return int(self.image_ids.shape[0])

    def person_counts(self) -> Dict[int, int]:
        self.ensure_fresh()
        return dict(self._person_counts)

    def row_of(self, image_id: int) -> Optional[int]:
        self.ensure_fresh()
        return self._row_of_id.get(int(image_id))

    def row_map(self) -> Dict[int, int]:
        """image_id -> row, freshness checked once — for per-hit loops.

        Calling ``row_of`` inside a 1000-hit loop pays the freshness check
        1000 times; snapshotting the mapping pays it once.
        """
        self.ensure_fresh()
        return self._row_of_id

    def ids_of(self, rows: np.ndarray) -> List[int]:
        return [int(v) for v in self.image_ids[rows]]

    # -- filtering -------------------------------------------------------
    def select(self, filters: Filters) -> np.ndarray:
        """Row indices matching ``filters``, unordered."""
        self.ensure_fresh()
        n = self.image_ids.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)

        mask = np.ones(n, dtype=bool)

        if filters.start_date is not None or filters.end_date is not None:
            dated = ~np.isnan(self.taken_ts)
            # A photo with no capture date cannot be shown to be inside a
            # date range, so it is excluded. The old query said
            # "date >= X OR date IS NULL", which let every undated photo
            # match every date filter.
            mask &= dated
            if filters.start_date is not None:
                mask &= self.taken_ts >= filters.start_date.timestamp()
            if filters.end_date is not None:
                mask &= self.taken_ts <= filters.end_date.timestamp()

        if filters.people_ids:
            people_mask = np.zeros(n, dtype=bool) if filters.people_mode != "all" \
                else np.ones(n, dtype=bool)
            for pid in filters.people_ids:
                rows = self._rows_by_person.get(int(pid))
                one = np.zeros(n, dtype=bool)
                if rows is not None:
                    one[rows] = True
                if filters.people_mode == "all":
                    people_mask &= one
                else:
                    people_mask |= one
            mask &= people_mask

        if filters.has_location is not None:
            located = ~np.isnan(self.lat)
            mask &= located if filters.has_location else ~located

        if filters.near_lat is not None and filters.near_lon is not None:
            # Equirectangular approximation — exact enough at photo-radius
            # scales, and it vectorises to two multiplies per row.
            km_per_deg = 111.32
            dlat = (self.lat - filters.near_lat) * km_per_deg
            dlon = ((self.lon - filters.near_lon) * km_per_deg
                    * np.cos(np.radians(filters.near_lat)))
            with np.errstate(invalid="ignore"):
                within = (dlat * dlat + dlon * dlon) <= filters.near_km ** 2
            mask &= np.nan_to_num(within, nan=False).astype(bool)

        if filters.has_faces is not None:
            has = self.face_count > 0
            mask &= has if filters.has_faces else ~has

        if filters.media:
            mask &= self.is_video if filters.media == "video" else ~self.is_video

        if filters.untagged_only:
            tagged = np.zeros(n, dtype=bool)
            for rows in self._rows_by_person.values():
                tagged[rows] = True
            mask &= self.face_count > 0
            mask &= ~tagged

        if filters.camera:
            wanted = filters.camera.strip().lower()
            camera_mask = np.fromiter(
                (camera.strip().lower() == wanted for camera in self.cameras),
                dtype=bool, count=n)
            mask &= camera_mask

        if filters.folder:
            # Store native absolute paths, but compare with a canonical slash
            # so a Windows backslash does not turn subtree filtering into an
            # exact-folder-only match.
            prefix = filters.folder.replace("\\", "/").rstrip("/")
            folder_mask = np.fromiter(
                (normalised == prefix or normalised.startswith(prefix + "/")
                 for normalised in
                 (folder.replace("\\", "/").rstrip("/")
                  for folder in self.folders)),
                dtype=bool, count=n)
            mask &= folder_mask

        return np.flatnonzero(mask).astype(np.int64)

    # -- ordering --------------------------------------------------------
    def order(self, rows: np.ndarray, sort: str = "date_desc",
              seed: int = 0) -> np.ndarray:
        self.ensure_fresh()
        if rows.size == 0:
            return rows

        if sort == "date_desc":
            # Reuse the precomputed global order; an O(n) mask lookup beats
            # re-sorting the subset every request.
            keep = np.zeros(self.image_ids.shape[0], dtype=bool)
            keep[rows] = True
            return self._order_date_desc[keep[self._order_date_desc]]
        if sort == "date_asc":
            keys = np.where(np.isnan(self.taken_ts[rows]), np.inf, self.taken_ts[rows])
            return rows[np.lexsort((self.image_ids[rows], keys))]
        if sort == "added_desc":
            keys = np.nan_to_num(self.added_ts[rows], nan=0.0)
            return rows[np.lexsort((self.image_ids[rows], keys))[::-1]]
        if sort == "random":
            rng = np.random.default_rng(seed)
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            return shuffled
        return rows

    # -- timeline --------------------------------------------------------
    def month_histogram(self, rows: Optional[np.ndarray] = None
                        ) -> List[Tuple[str, int]]:
        """(YYYY-MM, count) pairs — powers the timeline scrubber."""
        self.ensure_fresh()
        ts = self.taken_ts if rows is None else self.taken_ts[rows]
        ts = ts[~np.isnan(ts)]
        if ts.size == 0:
            return []
        dates = np.asarray(ts, dtype="datetime64[s]").astype("datetime64[M]")
        values, counts = np.unique(dates, return_counts=True)
        return [(str(v), int(c)) for v, c in zip(values, counts)]


def _timestamps(column) -> np.ndarray:
    """Arrow timestamp column -> float64 POSIX seconds, NaN where null."""
    values = column.to_numpy(zero_copy_only=False)
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) == 0:
        return out
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        as_seconds = arr.astype("datetime64[ms]").astype("float64") / 1000.0
        valid = ~np.isnat(arr)
        out[valid] = as_seconds[valid]
        return out
    for i, v in enumerate(values):
        if v is None:
            continue
        try:
            out[i] = v.timestamp()
        except Exception:
            continue
    return out


def _floats(column) -> np.ndarray:
    values = column.to_numpy(zero_copy_only=False)
    return np.asarray(
        [np.nan if v is None else float(v) for v in values], dtype=np.float64
    ) if values.dtype == object else values.astype(np.float64)
