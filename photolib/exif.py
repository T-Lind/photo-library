"""EXIF extraction: capture time, GPS coordinates, and camera.

Two changes from the old implementation matter:

* HEIC is read through pillow-heif rather than ``pyheif``. pyheif is
  unmaintained, needs libheif headers at build time, and was the main reason
  the project would not install cleanly — and pillow-heif was already a
  dependency, exposing EXIF through the normal Pillow API.
* Coordinates are returned as real floats and stored in numeric columns.
  They used to be stringified Python tuples, which made every geographic
  question (near here, has location, on a map) impossible to answer without
  parsing text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from .imageio import HEIF_EXTS, RAW_EXTS, VIDEO_EXTS, register_heif

logger = logging.getLogger(__name__)

EXIF_IFD = 0x8769
GPS_IFD = 0x8825
DATETIME_ORIGINAL = 36867
DATETIME_DIGITIZED = 36868
DATETIME = 306
MAKE = 271
MODEL = 272
OFFSET_TIME_ORIGINAL = 36881

GPS_LATITUDE_REF, GPS_LATITUDE = 1, 2
GPS_LONGITUDE_REF, GPS_LONGITUDE = 3, 4

_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f")
# Filenames like IMG_20180704_153000 or 2018-07-04 15.30.00 — the fallback
# when a file has been stripped of EXIF by a messaging app.
_FILENAME_DATE = re.compile(
    r"(?P<y>19\d{2}|20\d{2})[-_.]?(?P<m>0[1-9]|1[0-2])[-_.]?(?P<d>0[1-9]|[12]\d|3[01])"
    r"(?:[-_ tT]?(?P<H>[01]\d|2[0-3])[-_.:]?(?P<M>[0-5]\d)(?:[-_.:]?(?P<S>[0-5]\d))?)?"
)


@dataclass
class PhotoMetadata:
    taken_at: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    camera: str = ""

    @property
    def has_location(self) -> bool:
        return self.lat is not None and self.lon is not None


def _parse_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip().strip("\x00")
    if not text or text.startswith("0000"):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:  # exifread Ratio
            return float(value.num) / float(value.den)
        except Exception:
            return None


def _dms_to_degrees(dms, ref: str) -> Optional[float]:
    try:
        parts = [_to_float(v) for v in dms]
    except TypeError:
        return None
    if len(parts) < 3 or any(p is None for p in parts):
        return None
    degrees = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if str(ref).upper().startswith(("S", "W")):
        degrees = -degrees
    if not (-180.0 <= degrees <= 180.0):
        return None
    return round(degrees, 6)


def _valid_coords(lat: Optional[float], lon: Optional[float]
                  ) -> Tuple[Optional[float], Optional[float]]:
    if lat is None or lon is None:
        return None, None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None
    # Exactly (0, 0) is Null Island — always a broken GPS write, never a photo.
    if abs(lat) < 1e-7 and abs(lon) < 1e-7:
        return None, None
    return lat, lon


def date_from_filename(path: Path) -> Optional[datetime]:
    m = _FILENAME_DATE.search(path.stem)
    if not m:
        return None
    g = m.groupdict()
    try:
        return datetime(
            int(g["y"]), int(g["m"]), int(g["d"]),
            int(g["H"] or 0), int(g["M"] or 0), int(g["S"] or 0))
    except ValueError:
        return None


def _from_pillow(path: Path) -> PhotoMetadata:
    from PIL import Image

    register_heif()
    with Image.open(path) as image:
        exif = image.getexif()
        if not exif:
            return PhotoMetadata()
        sub = exif.get_ifd(EXIF_IFD) or {}
        gps = exif.get_ifd(GPS_IFD) or {}

    taken = (_parse_datetime(sub.get(DATETIME_ORIGINAL))
             or _parse_datetime(sub.get(DATETIME_DIGITIZED))
             or _parse_datetime(exif.get(DATETIME)))

    lat = _dms_to_degrees(gps.get(GPS_LATITUDE), gps.get(GPS_LATITUDE_REF, "N")) \
        if gps.get(GPS_LATITUDE) else None
    lon = _dms_to_degrees(gps.get(GPS_LONGITUDE), gps.get(GPS_LONGITUDE_REF, "E")) \
        if gps.get(GPS_LONGITUDE) else None
    lat, lon = _valid_coords(lat, lon)

    make = str(exif.get(MAKE, "") or "").strip()
    model = str(exif.get(MODEL, "") or "").strip()
    camera = (f"{make} {model}".strip() if make and model.lower().find(make.lower()) < 0
              else model or make)

    return PhotoMetadata(taken_at=taken, lat=lat, lon=lon, camera=camera.strip())


def _from_exifread(path: Path) -> PhotoMetadata:
    import exifread

    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=False)

    taken = (_parse_datetime(tags.get("EXIF DateTimeOriginal"))
             or _parse_datetime(tags.get("Image DateTime")))

    lat = lon = None
    lat_tag, lon_tag = tags.get("GPS GPSLatitude"), tags.get("GPS GPSLongitude")
    if lat_tag and lon_tag:
        lat = _dms_to_degrees(lat_tag.values, str(tags.get("GPS GPSLatitudeRef", "N")))
        lon = _dms_to_degrees(lon_tag.values, str(tags.get("GPS GPSLongitudeRef", "E")))
    lat, lon = _valid_coords(lat, lon)

    make = str(tags.get("Image Make", "") or "").strip()
    model = str(tags.get("Image Model", "") or "").strip()
    camera = f"{make} {model}".strip() if make and make.lower() not in model.lower() else model

    return PhotoMetadata(taken_at=taken, lat=lat, lon=lon, camera=camera.strip())


def read_metadata(path) -> PhotoMetadata:
    """Best-effort metadata for any supported file. Never raises."""
    path = Path(path)
    suffix = path.suffix.lower()

    meta = PhotoMetadata()
    try:
        if suffix in VIDEO_EXTS:
            # Container metadata, not EXIF: capture time comes from the
            # (UTC) creation_time tag, converted to local wall-clock time.
            from .imageio import probe_video

            meta = PhotoMetadata(taken_at=probe_video(path).taken_at)
        elif suffix in RAW_EXTS:
            meta = _from_exifread(path)
        else:
            meta = _from_pillow(path)
            if meta.taken_at is None and suffix in HEIF_EXTS:
                # Some HEIC writers put EXIF where Pillow doesn't look.
                meta = _merge(meta, _from_exifread(path))
    except Exception as exc:
        logger.debug("EXIF read failed for %s: %s", path, exc)

    if meta.taken_at is None:
        meta.taken_at = date_from_filename(path)
    if meta.taken_at is not None and not (1900 <= meta.taken_at.year <= 2100):
        # Epoch-zero and other garbage dates used to sort to the very start of
        # every timeline; treat them as unknown instead.
        meta.taken_at = None
    return meta


def _merge(primary: PhotoMetadata, fallback: PhotoMetadata) -> PhotoMetadata:
    return PhotoMetadata(
        taken_at=primary.taken_at or fallback.taken_at,
        lat=primary.lat if primary.lat is not None else fallback.lat,
        lon=primary.lon if primary.lon is not None else fallback.lon,
        camera=primary.camera or fallback.camera,
    )
