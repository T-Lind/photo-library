"""Ingest pipeline: scan a folder, embed, detect faces, write to LanceDB.

Built around three constraints that come from actually having 200k photos:

* **Incremental by default and resumable.** Work is committed in batches, so
  an interrupted run loses at most one batch and re-running picks up where it
  stopped. Nothing depends on a JSON sidecar surviving.
* **Decode once.** Each photo is decoded a single time and the same pixel
  buffer feeds the embedder, the face detector, the perceptual hash, and the
  thumbnail writer.
* **Bounded memory.** Files are processed in fixed-size chunks and buffers
  are dropped as soon as the batch is written, so peak RSS does not depend
  on library size.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from .config import Settings, get_settings
from .db import (FACES, IMAGES, Library, LibraryMeta, SCHEMA_VERSION,
                 UNASSIGNED, faces_schema, images_schema, now_ms)
from .embeddings import Embedder, ImageInput, build_embedder
from .exif import read_metadata
from .faces import FaceBackend, build_face_backend
from .faces.cluster import FaceAssigner, FaceObservation
from .hashing import content_hash, phash
from .imageio import iter_image_files, load_rgb_array, open_image
from .thumbnails import ThumbnailCache

logger = logging.getLogger(__name__)

# Long edge the pixel buffer is decoded to. Big enough that RetinaFace still
# finds faces in a group shot, small enough that 8 concurrent buffers fit
# comfortably in memory.
WORK_MAX_SIDE = 1600


@dataclass
class IngestStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    faces_detected: int = 0
    people_created: int = 0
    elapsed: float = 0.0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["errors"] = self.errors[:50]
        d["elapsed"] = round(self.elapsed, 2)
        return d


@dataclass
class _Prepared:
    """Everything derived from one file before it hits the database."""

    path: str
    filename: str
    folder: str
    width: int
    height: int
    file_size: int
    mtime: float
    content_hash: str
    phash: int
    taken_at: Optional[datetime]
    lat: Optional[float]
    lon: Optional[float]
    camera: str
    array: Optional[np.ndarray] = None
    error: Optional[str] = None


ProgressFn = Callable[[str, int, int, dict], None]


class Indexer:
    def __init__(self, library: Library, settings: Optional[Settings] = None,
                 embedder: Optional[Embedder] = None,
                 face_backend: Optional[FaceBackend] = None,
                 thumbnails: Optional[ThumbnailCache] = None,
                 progress: Optional[ProgressFn] = None):
        self.settings = settings or get_settings()
        self.library = library
        self.embedder = embedder or build_embedder(self.settings)
        self.faces = face_backend or build_face_backend(self.settings)
        self.thumbs = thumbnails or ThumbnailCache(self.settings.thumbnail_cache_dir)
        self.progress = progress or (lambda *a, **k: None)

    # -- public API ------------------------------------------------------
    def meta(self) -> LibraryMeta:
        return LibraryMeta(
            schema_version=SCHEMA_VERSION,
            image_dim=self.embedder.dim,
            face_dim=self.faces.dim,
            embed_backend=self.embedder.backend,
            embed_model=self.embedder.model_name,
            face_backend=self.faces.backend,
            face_model=self.faces.model_name,
        )

    def index_directory(self, root: os.PathLike | str, rebuild: bool = False,
                        prune_missing: bool = False,
                        limit: Optional[int] = None) -> IngestStats:
        """Index every supported image under ``root``.

        Incremental unless ``rebuild``: files already present with an
        unchanged size and mtime are skipped, changed files are re-indexed in
        place, and only genuinely new files cost model time.
        """
        started = time.time()
        stats = IngestStats()
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")

        meta = self.meta()
        if rebuild or not self.library.initialised():
            self.library.create(meta, drop_existing=rebuild)
        else:
            self.library.verify_compatible(meta)

        self.progress("scanning", 0, 0, {"root": str(root)})
        files = list(iter_image_files(root, self.settings.follow_symlinks))
        files.sort()
        if limit:
            files = files[:limit]
        stats.scanned = len(files)

        known = self._existing_files()
        todo: List[Path] = []
        stale_ids: List[int] = []
        for p in files:
            key = str(p)
            prior = known.get(key)
            if prior is None:
                todo.append(p)
                continue
            image_id, size, mtime = prior
            try:
                st = p.stat()
            except OSError:
                stats.failed += 1
                continue
            if int(st.st_size) != size or abs(st.st_mtime - mtime) > 1.0:
                todo.append(p)
                stale_ids.append(image_id)   # re-index: drop the old row first
            else:
                stats.skipped += 1

        if prune_missing:
            present = {str(p) for p in files}
            missing = [image_id for path, (image_id, _, _) in known.items()
                       if path not in present and _under(path, root)]
            if missing:
                self.remove_images(missing)
                stats.removed = len(missing)

        if stale_ids:
            self.remove_images(stale_ids)
            stats.updated = len(stale_ids)

        if not todo:
            logger.info("Nothing to index: %d files already up to date", stats.skipped)
            stats.elapsed = time.time() - started
            self.progress("done", stats.scanned, stats.scanned, stats.as_dict())
            return stats

        logger.info("Indexing %d new/changed files (%d unchanged)",
                    len(todo), stats.skipped)
        self._ingest(todo, stats)

        self.progress("indexing_vectors", len(todo), len(todo), {})
        report = self.library.build_indexes(self.settings.ann_min_rows)
        logger.info("Index build: %s", report)

        stats.elapsed = time.time() - started
        self.progress("done", len(todo), len(todo), stats.as_dict())
        return stats

    def remove_images(self, image_ids: Sequence[int]) -> None:
        """Delete images, their faces, and their cached thumbnails."""
        if not image_ids:
            return
        for start in range(0, len(image_ids), 4096):
            chunk = image_ids[start:start + 4096]
            ids = ", ".join(str(int(i)) for i in chunk)
            self.library.images.delete(f"image_id IN ({ids})")
            self.library.faces.delete(f"image_id IN ({ids})")
        for image_id in image_ids:
            try:
                self.thumbs.purge_image(int(image_id))
            except OSError:
                pass

    # -- internals -------------------------------------------------------
    def _existing_files(self) -> Dict[str, Tuple[int, int, float]]:
        """path -> (image_id, file_size, mtime) for everything already indexed."""
        try:
            tbl = self.library.images.to_lance().to_table(
                columns=["image_id", "path", "file_size", "mtime"])
        except Exception:
            return {}
        return {
            row["path"]: (int(row["image_id"]), int(row["file_size"] or 0),
                          float(row["mtime"] or 0.0))
            for row in tbl.to_pylist()
        }

    def _ingest(self, files: Sequence[Path], stats: IngestStats) -> None:
        img_schema = images_schema(self.embedder.dim)
        face_schema = faces_schema(self.faces.dim)

        next_image_id = self.library.next_id(IMAGES, "image_id")
        next_face_id = self.library.next_id(FACES, "face_id")
        assigner = FaceAssigner(
            self.library, self.faces.dim,
            match_threshold=self.settings.face_match_threshold,
            strong_threshold=self.settings.face_strong_match_threshold,
        )
        people_before = len(assigner.people)

        chunk_size = max(self.settings.embed_batch_size, 8)
        workers = max(1, min(self.settings.worker_count, 16))
        image_rows: List[dict] = []
        face_rows: List[dict] = []
        done = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(files), chunk_size):
                chunk = files[start:start + chunk_size]
                prepared = list(pool.map(self._prepare, chunk))

                good = [p for p in prepared if p.error is None]
                for bad in (p for p in prepared if p.error is not None):
                    stats.failed += 1
                    if len(stats.errors) < 200:
                        stats.errors.append(f"{bad.path}: {bad.error}")

                if good:
                    vectors = self._embed(good, stats)
                    detections = self._detect_faces(good, stats)

                    for prep, vector, faces in zip(good, vectors, detections):
                        if vector is None:
                            continue
                        image_id = next_image_id
                        next_image_id += 1

                        # Detection ran on a downscaled buffer; boxes are
                        # stored in original-image coordinates so that face
                        # crops and the UI's overlay both line up with the
                        # full-resolution photo.
                        scale = self._detection_scale(prep)
                        observations = []
                        for face in faces:
                            observations.append(FaceObservation(
                                image_id=image_id,
                                embedding=face.embedding,
                                bbox=_scale_bbox(face.bbox, scale),
                                det_score=face.det_score,
                                quality=face.quality,
                                face_id=next_face_id,
                            ))
                            next_face_id += 1
                        assigner.assign(observations)
                        stats.faces_detected += len(observations)

                        people_ids = sorted({o.person_id for o in observations
                                             if o.person_id != UNASSIGNED})
                        image_rows.append(self._image_row(
                            image_id, prep, vector, people_ids, len(observations)))
                        face_rows.extend(self._face_row(o) for o in observations)

                    if self.settings.pregenerate_thumbnails:
                        pool.map(self._pregenerate,
                                 [(r["image_id"], r["path"]) for r in image_rows[-len(good):]])

                # Release the decoded buffers before the next chunk is read.
                for prep in prepared:
                    prep.array = None

                done += len(chunk)
                self.progress("ingesting", done, len(files),
                              {"added": stats.added + len(image_rows),
                               "faces": stats.faces_detected})

                if len(image_rows) >= self.settings.write_batch_size:
                    self._write(image_rows, face_rows, img_schema, face_schema)
                    stats.added += len(image_rows)
                    image_rows, face_rows = [], []
                    assigner.flush()

        if image_rows:
            self._write(image_rows, face_rows, img_schema, face_schema)
            stats.added += len(image_rows)
        assigner.flush()
        stats.people_created = max(0, len(assigner.people) - people_before)

    def _prepare(self, path: Path) -> _Prepared:
        try:
            st = path.stat()
            meta = read_metadata(path)
            array = load_rgb_array(path, max_side=WORK_MAX_SIDE)
            with open_image(path, target=(64, 64)) as small:
                perceptual = phash(small)
            width, height = self._true_size(path, array)
            return _Prepared(
                path=str(path),
                filename=path.name,
                folder=str(path.parent),
                width=width,
                height=height,
                file_size=int(st.st_size),
                mtime=float(st.st_mtime),
                content_hash=content_hash(path),
                phash=perceptual,
                taken_at=meta.taken_at,
                lat=meta.lat,
                lon=meta.lon,
                camera=meta.camera,
                array=array,
            )
        except Exception as exc:
            return _Prepared(path=str(path), filename=path.name,
                             folder=str(path.parent), width=0, height=0,
                             file_size=0, mtime=0.0, content_hash="", phash=0,
                             taken_at=None, lat=None, lon=None, camera="",
                             error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _detection_scale(prep: _Prepared) -> float:
        """Original width divided by the working buffer's width."""
        if prep.array is None or prep.array.shape[1] == 0 or prep.width <= 0:
            return 1.0
        return prep.width / float(prep.array.shape[1])

    @staticmethod
    def _true_size(path: Path, array: np.ndarray) -> Tuple[int, int]:
        """Original dimensions, not the downscaled working buffer's."""
        from .imageio import read_size

        try:
            return read_size(path)
        except Exception:
            return int(array.shape[1]), int(array.shape[0])

    def _embed(self, prepared: Sequence[_Prepared],
               stats: IngestStats) -> List[Optional[np.ndarray]]:
        inputs = [ImageInput(path=p.path, array=p.array) for p in prepared]
        try:
            vectors = self.embedder.embed_images(inputs)
            return [v for v in vectors]
        except Exception as exc:
            logger.warning("Batch embedding failed (%s); retrying one by one", exc)

        out: List[Optional[np.ndarray]] = []
        for item, prep in zip(inputs, prepared):
            try:
                out.append(self.embedder.embed_images([item])[0])
            except Exception as exc:
                stats.failed += 1
                if len(stats.errors) < 200:
                    stats.errors.append(f"{prep.path}: embedding failed: {exc}")
                out.append(None)
        return out

    def _detect_faces(self, prepared: Sequence[_Prepared],
                      stats: IngestStats) -> List[list]:
        arrays = [p.array for p in prepared]
        try:
            return self.faces.detect_batch(arrays)
        except Exception as exc:
            logger.warning("Batch face detection failed (%s); retrying one by one", exc)

        out = []
        for arr, prep in zip(arrays, prepared):
            try:
                out.append(self.faces.detect(arr))
            except Exception as exc:
                if len(stats.errors) < 200:
                    stats.errors.append(f"{prep.path}: face detection failed: {exc}")
                out.append([])
        return out

    def _image_row(self, image_id: int, prep: _Prepared, vector: np.ndarray,
                   people_ids: List[int], face_count: int) -> dict:
        return {
            "image_id": image_id,
            "vector": np.asarray(vector, dtype=np.float32).tolist(),
            "path": prep.path,
            "filename": prep.filename,
            "folder": prep.folder,
            "taken_at": prep.taken_at,
            "added_at": now_ms(),
            "width": prep.width,
            "height": prep.height,
            "file_size": prep.file_size,
            "mtime": prep.mtime,
            "content_hash": prep.content_hash,
            "phash": prep.phash,
            "lat": prep.lat,
            "lon": prep.lon,
            "place": "",
            "camera": prep.camera,
            "people_ids": [int(p) for p in people_ids],
            "face_count": face_count,
        }

    @staticmethod
    def _face_row(obs: FaceObservation) -> dict:
        x, y, w, h = obs.bbox
        return {
            "face_id": obs.face_id,
            "vector": np.asarray(obs.embedding, dtype=np.float32).tolist(),
            "image_id": obs.image_id,
            "person_id": int(obs.person_id),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "det_score": float(obs.det_score),
            "quality": float(obs.quality),
            "confirmed": False,
            "crop_path": "",
        }

    def _pregenerate(self, args: Tuple[int, str]) -> None:
        image_id, path = args
        self.thumbs.pregenerate(image_id, path)

    def _write(self, image_rows: List[dict], face_rows: List[dict],
               img_schema: pa.Schema, face_schema: pa.Schema) -> None:
        if image_rows:
            self.library.images.add(pa.Table.from_pylist(image_rows, schema=img_schema))
        if face_rows:
            self.library.faces.add(pa.Table.from_pylist(face_rows, schema=face_schema))


def _scale_bbox(bbox: Tuple[int, int, int, int], scale: float
                ) -> Tuple[int, int, int, int]:
    if abs(scale - 1.0) < 1e-6:
        return bbox
    x, y, w, h = bbox
    return (int(round(x * scale)), int(round(y * scale)),
            int(round(w * scale)), int(round(h * scale)))


def _under(path: str, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False
