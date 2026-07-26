"""Application service layer — everything the API does, minus HTTP.

Keeping this separate from the FastAPI routers means the same operations are
available from the CLI and from tests without spinning up a server.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .browse import Filters, LibraryIndex
from .config import Settings, get_settings
from .db import (FACES, IMAGE_LIST_COLUMNS, Library, UNASSIGNED, now_ms)
from .embeddings import Embedder, build_embedder
from .faces import FaceBackend, build_face_backend
from .faces.cluster import recluster, suggest_for_person
from .hashing import group_near_duplicates
from .jobs import JobManager
from .thumbnails import ThumbnailCache

logger = logging.getLogger(__name__)


class NotFound(LookupError):
    pass


@dataclass
class SearchPage:
    total: int
    page: int
    per_page: int
    results: List[dict]
    took_ms: float = 0.0
    # Present when the query was semantic; lets the UI show a relevance bar.
    scored: bool = False


class PhotoService:
    def __init__(self, settings: Optional[Settings] = None,
                 library: Optional[Library] = None,
                 embedder: Optional[Embedder] = None,
                 face_backend: Optional[FaceBackend] = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.library = library or Library(self.settings.db_uri)
        self.index = LibraryIndex(self.library)
        self.thumbs = ThumbnailCache(self.settings.thumbnail_cache_dir)
        self.jobs = JobManager(self.settings.state_dir)

        self._embedder = embedder
        self._faces = face_backend
        self._model_lock = threading.Lock()
        self._people_cache: Optional[Tuple[int, Dict[int, dict]]] = None

    # -- lazily-loaded models -------------------------------------------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            with self._model_lock:
                if self._embedder is None:
                    self._embedder = build_embedder(self.settings)
        return self._embedder

    @property
    def face_backend(self) -> FaceBackend:
        if self._faces is None:
            with self._model_lock:
                if self._faces is None:
                    self._faces = build_face_backend(self.settings)
        return self._faces

    @property
    def ready(self) -> bool:
        return self.library.initialised()

    def require_ready(self) -> None:
        if not self.ready:
            raise NotFound(
                "Library is empty. Index a folder first "
                "(POST /api/v1/admin/index, or `python -m photolib.cli index <folder>`).")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: Optional[str], filters: Filters, sort: str = "date_desc",
               page: int = 1, per_page: Optional[int] = None,
               min_score: Optional[float] = None) -> SearchPage:
        import time

        started = time.perf_counter()
        self.require_ready()
        per_page = min(per_page or self.settings.default_page_size,
                       self.settings.max_page_size)

        allowed = self.index.select(filters)
        has_query = bool(query and query.strip())

        if has_query:
            rows, scores = self._semantic_rows(query.strip(), allowed, min_score)
            if sort == "relevance":
                ordered = rows
            else:
                # An explicit date/random sort re-orders the matches; relevance
                # scores no longer describe the ordering, so drop them.
                ordered, scores = self.index.order(rows, sort), None
        else:
            # "relevance" has no meaning without a query.
            ordered = self.index.order(
                allowed, "date_desc" if sort == "relevance" else sort)
            scores = None

        total = int(ordered.shape[0])
        start = (page - 1) * per_page
        window = ordered[start:start + per_page]
        image_ids = self.index.ids_of(window)

        results = self.hydrate(image_ids)
        if scores is not None:
            score_map = dict(zip(image_ids, scores[start:start + per_page].tolist()))
            for item in results:
                item["score"] = round(float(score_map.get(item["image_id"], 0.0)), 4)

        return SearchPage(
            total=total, page=page, per_page=per_page, results=results,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
            scored=has_query)

    def _semantic_rows(self, query: str, allowed: np.ndarray,
                       min_score: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        vector = self.embedder.embed_texts([query])[0]
        return self._vector_rows(vector, allowed, min_score)

    def _vector_rows(self, vector: np.ndarray, allowed: np.ndarray,
                     min_score: Optional[float] = None,
                     exclude_image_id: Optional[int] = None,
                     limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Rank ``allowed`` rows against a query vector using the ANN index.

        Filtering happens on the in-memory mask rather than in a SQL
        prefilter: LanceDB's prefilter would force a scan of the metadata for
        every probe, and the mask is a single NumPy lookup.
        """
        self.index.ensure_fresh()
        limit = limit or self.settings.max_candidates
        # Over-fetch when a filter is active so the page can still be filled
        # after the mask removes non-matching hits.
        library_size = max(self.index.count, 1)
        selectivity = max(allowed.shape[0] / library_size, 1e-3)
        fetch = int(min(library_size, max(limit, limit / selectivity)))

        table = (
            self.library.images.search(np.asarray(vector, dtype=np.float32).tolist())
            .metric("cosine")
            .nprobes(self.settings.nprobes)
            .refine_factor(self.settings.refine_factor)
            .select(["image_id"])
            .limit(fetch)
            .to_arrow()
        )
        if table.num_rows == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        allowed_set = np.zeros(self.index.count, dtype=bool)
        allowed_set[allowed] = True

        hits = table["image_id"].to_pylist()
        distances = (table["_distance"].to_pylist()
                     if "_distance" in table.column_names
                     else [0.0] * table.num_rows)

        rows: List[int] = []
        scores: List[float] = []
        for image_id, distance in zip(hits, distances):
            image_id = int(image_id)
            if exclude_image_id is not None and image_id == exclude_image_id:
                continue
            row = self.index.row_of(image_id)
            if row is None or not allowed_set[row]:
                continue
            similarity = 1.0 - float(distance)
            if min_score is not None and similarity < min_score:
                continue
            rows.append(row)
            scores.append(similarity)
            if len(rows) >= limit:
                break

        return np.asarray(rows, dtype=np.int64), np.asarray(scores, dtype=np.float32)

    def search_by_image(self, image_path: os.PathLike | str, filters: Filters,
                        page: int = 1, per_page: Optional[int] = None) -> SearchPage:
        """Reverse image search from an uploaded file."""
        import time

        started = time.perf_counter()
        self.require_ready()
        per_page = min(per_page or self.settings.default_page_size,
                       self.settings.max_page_size)

        vector = self.embedder.embed_images([str(image_path)])[0]
        allowed = self.index.select(filters)
        rows, scores = self._vector_rows(vector, allowed)

        start = (page - 1) * per_page
        window = rows[start:start + per_page]
        results = self.hydrate(self.index.ids_of(window))
        for item, score in zip(results, scores[start:start + per_page]):
            item["score"] = round(float(score), 4)

        return SearchPage(total=int(rows.shape[0]), page=page, per_page=per_page,
                          results=results,
                          took_ms=round((time.perf_counter() - started) * 1000, 2),
                          scored=True)

    def similar_images(self, image_id: int, limit: int = 24) -> List[dict]:
        self.require_ready()
        row = self._image_row(image_id, ["vector"])
        vector = np.asarray(row["vector"], dtype=np.float32)
        allowed = self.index.select(Filters())
        rows, scores = self._vector_rows(
            vector, allowed, exclude_image_id=image_id, limit=limit)
        results = self.hydrate(self.index.ids_of(rows[:limit]))
        for item, score in zip(results, scores[:limit]):
            item["score"] = round(float(score), 4)
        return results

    # ------------------------------------------------------------------
    # Image reads
    # ------------------------------------------------------------------
    def hydrate(self, image_ids: Sequence[int]) -> List[dict]:
        """Fetch display rows for exactly these ids, preserving order."""
        ids = [int(i) for i in image_ids]
        if not ids:
            return []
        id_list = ", ".join(str(i) for i in ids)
        rows = (
            self.library.images.search()
            .where(f"image_id IN ({id_list})")
            .select(IMAGE_LIST_COLUMNS)
            .limit(len(ids))
            .to_arrow()
            .to_pylist()
        )
        by_id = {int(r["image_id"]): _image_dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def _image_row(self, image_id: int, columns: List[str]) -> dict:
        rows = (
            self.library.images.search()
            .where(f"image_id = {int(image_id)}")
            .select(columns)
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            raise NotFound(f"Image {image_id} not found")
        return rows[0]

    def image_path(self, image_id: int) -> str:
        return str(self._image_row(image_id, ["path"])["path"])

    def image_details(self, image_id: int) -> dict:
        self.require_ready()
        row = self._image_row(
            image_id,
            IMAGE_LIST_COLUMNS + ["folder", "camera", "file_size", "content_hash"])
        details = _image_dict(row)
        details.update({
            "folder": row.get("folder") or "",
            "camera": row.get("camera") or "",
            "file_size": int(row.get("file_size") or 0),
        })

        names = self.people_by_id()
        faces = self.faces_in_image(image_id)
        seen: Dict[int, dict] = {}
        for face in faces:
            pid = face["person_id"]
            if pid == UNASSIGNED or pid in seen:
                continue
            seen[pid] = {
                "person_id": pid,
                "name": names.get(pid, {}).get("name", ""),
                "face_id": face["face_id"],
            }
        details["people"] = list(seen.values())
        details["faces"] = faces
        return details

    def faces_in_image(self, image_id: int) -> List[dict]:
        rows = (
            self.library.faces.search()
            .where(f"image_id = {int(image_id)}")
            .select(["face_id", "person_id", "x", "y", "w", "h",
                     "det_score", "quality", "confirmed"])
            .limit(256)
            .to_arrow()
            .to_pylist()
        )
        return [{
            "face_id": int(r["face_id"]),
            "person_id": int(r["person_id"]),
            "bbox": [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])],
            "det_score": round(float(r["det_score"]), 3),
            "quality": round(float(r["quality"]), 3),
            "confirmed": bool(r["confirmed"]),
        } for r in rows]

    # ------------------------------------------------------------------
    # People
    # ------------------------------------------------------------------
    def people_by_id(self) -> Dict[int, dict]:
        try:
            version = self.library.people.version
        except Exception:
            version = -1
        if self._people_cache and self._people_cache[0] == version:
            return self._people_cache[1]

        try:
            rows = self.library.people.to_lance().to_table(
                columns=["person_id", "name", "face_count", "cover_face_id",
                         "hidden"]).to_pylist()
        except Exception:
            rows = []
        mapping = {
            int(r["person_id"]): {
                "person_id": int(r["person_id"]),
                "name": r["name"] or "",
                "face_count": int(r["face_count"] or 0),
                # `or -1` would turn a legitimate face_id of 0 into "none".
                "cover_face_id": int(r["cover_face_id"])
                if r["cover_face_id"] is not None else -1,
                "hidden": bool(r["hidden"]),
            } for r in rows
        }
        self._people_cache = (version, mapping)
        return mapping

    def list_people(self, include_hidden: bool = False, named_only: bool = False,
                    min_photos: int = 1) -> List[dict]:
        self.require_ready()
        counts = self.index.person_counts()
        out = []
        for person in self.people_by_id().values():
            if person["hidden"] and not include_hidden:
                continue
            if named_only and not person["name"]:
                continue
            photo_count = counts.get(person["person_id"], 0)
            if photo_count < min_photos:
                continue
            out.append({**person, "photo_count": photo_count})
        # Named people first (they're the ones you actually browse by), then
        # by how much of the library they appear in.
        out.sort(key=lambda p: (not p["name"], -p["photo_count"], p["person_id"]))
        return out

    def get_person(self, person_id: int) -> dict:
        person = self.people_by_id().get(int(person_id))
        if person is None:
            raise NotFound(f"Person {person_id} not found")
        return {**person, "photo_count": self.index.person_counts().get(int(person_id), 0)}

    def rename_person(self, person_id: int, name: str) -> dict:
        self.get_person(person_id)
        self.library.people.update(where=f"person_id = {int(person_id)}",
                                   values={"name": name})
        self._people_cache = None
        return self.get_person(person_id)

    def set_person_hidden(self, person_id: int, hidden: bool) -> dict:
        self.get_person(person_id)
        self.library.people.update(where=f"person_id = {int(person_id)}",
                                   values={"hidden": bool(hidden)})
        self._people_cache = None
        return self.get_person(person_id)

    def merge_people(self, source_id: int, target_id: int) -> dict:
        if source_id == target_id:
            raise ValueError("Cannot merge a person into themselves")
        self.get_person(source_id)
        target = self.get_person(target_id)

        self.library.faces.update(where=f"person_id = {int(source_id)}",
                                  values={"person_id": int(target_id)})
        self.library.people.delete(f"person_id = {int(source_id)}")
        self._recompute_person(target_id)
        self._rewrite_image_people([int(source_id), int(target_id)])
        self._people_cache = None
        return self.get_person(target_id)

    def delete_person(self, person_id: int) -> dict:
        """Forget an identity; its faces go back to the unassigned pool."""
        self.get_person(person_id)
        affected = self.index.person_counts().get(int(person_id), 0)
        self.library.faces.update(where=f"person_id = {int(person_id)}",
                                  values={"person_id": UNASSIGNED})
        self.library.people.delete(f"person_id = {int(person_id)}")
        self._rewrite_image_people([int(person_id)])
        self._people_cache = None
        return {"deleted": int(person_id), "affected_images": affected}

    # ------------------------------------------------------------------
    # Faces
    # ------------------------------------------------------------------
    def get_face(self, face_id: int) -> dict:
        rows = (
            self.library.faces.search()
            .where(f"face_id = {int(face_id)}")
            .select(["face_id", "image_id", "person_id", "x", "y", "w", "h",
                     "quality", "det_score", "confirmed"])
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            raise NotFound(f"Face {face_id} not found")
        r = rows[0]
        return {
            "face_id": int(r["face_id"]),
            "image_id": int(r["image_id"]),
            "person_id": int(r["person_id"]),
            "bbox": [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])],
            "quality": round(float(r["quality"]), 3),
            "det_score": round(float(r["det_score"]), 3),
            "confirmed": bool(r["confirmed"]),
        }

    def assign_faces(self, face_ids: Sequence[int], person_id: Optional[int],
                     name: Optional[str] = None) -> dict:
        """Move faces to a person (or to a brand-new one when ``person_id`` is None).

        Manual assignments are marked ``confirmed`` so that later automatic
        reclustering treats them as ground truth instead of overwriting them.
        """
        ids = [int(f) for f in face_ids]
        if not ids:
            return {"updated": 0}

        touched = {self.get_face(f)["person_id"] for f in ids}

        if person_id is None:
            person_id = self._create_person(name or "")
        else:
            self.get_person(person_id)

        id_list = ", ".join(str(i) for i in ids)
        self.library.faces.update(
            where=f"face_id IN ({id_list})",
            values={"person_id": int(person_id), "confirmed": True})

        self._recompute_person(int(person_id))
        for old in touched:
            if old != UNASSIGNED and old != person_id:
                self._recompute_person(int(old))
        self._rewrite_image_people(list(touched | {int(person_id)}))
        self._people_cache = None
        return {"updated": len(ids), "person_id": int(person_id)}

    def detach_faces(self, face_ids: Sequence[int]) -> dict:
        """Mark faces as not-this-person, returning them to the unassigned pool."""
        ids = [int(f) for f in face_ids]
        if not ids:
            return {"updated": 0}
        touched = {self.get_face(f)["person_id"] for f in ids}
        id_list = ", ".join(str(i) for i in ids)
        self.library.faces.update(
            where=f"face_id IN ({id_list})",
            values={"person_id": UNASSIGNED, "confirmed": True})
        for old in touched:
            if old != UNASSIGNED:
                self._recompute_person(int(old))
        self._rewrite_image_people([o for o in touched if o != UNASSIGNED])
        self._people_cache = None
        return {"updated": len(ids)}

    def search_faces(self, vector: np.ndarray, limit: int = 60,
                     min_similarity: float = 0.3,
                     unassigned_only: bool = False) -> List[dict]:
        """Nearest faces to a query embedding."""
        query = self.library.faces.search(
            np.asarray(vector, dtype=np.float32).tolist()).metric("cosine")
        if unassigned_only:
            query = query.where(f"person_id = {UNASSIGNED}", prefilter=True)
        rows = (
            query.nprobes(self.settings.nprobes)
            .refine_factor(self.settings.refine_factor)
            .select(["face_id", "image_id", "person_id", "x", "y", "w", "h", "quality"])
            .limit(limit * 2)
            .to_arrow()
            .to_pylist()
        )
        out = []
        for r in rows:
            similarity = 1.0 - float(r.get("_distance", 1.0))
            if similarity < min_similarity:
                continue
            out.append({
                "face_id": int(r["face_id"]),
                "image_id": int(r["image_id"]),
                "person_id": int(r["person_id"]),
                "bbox": [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])],
                "quality": round(float(r["quality"]), 3),
                "similarity": round(similarity, 4),
            })
            if len(out) >= limit:
                break
        return out

    def search_faces_by_face(self, face_id: int, limit: int = 60,
                             min_similarity: float = 0.3) -> List[dict]:
        rows = (
            self.library.faces.search()
            .where(f"face_id = {int(face_id)}")
            .select(["vector"])
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            raise NotFound(f"Face {face_id} not found")
        vector = np.asarray(rows[0]["vector"], dtype=np.float32)
        return [f for f in self.search_faces(vector, limit + 1, min_similarity)
                if f["face_id"] != int(face_id)][:limit]

    def search_faces_in_upload(self, image_path: os.PathLike | str,
                               limit: int = 60,
                               min_similarity: float = 0.3) -> List[dict]:
        """"Who is this?" — detect faces in an uploaded photo and match them."""
        from .imageio import load_rgb_array

        array = load_rgb_array(image_path, max_side=1600)
        detected = self.face_backend.detect(array)
        if not detected:
            return []
        detected.sort(key=lambda f: f.area, reverse=True)
        return self.search_faces(detected[0].embedding, limit, min_similarity)

    def unassigned_faces(self, limit: int = 120, min_quality: float = 0.3) -> List[dict]:
        rows = (
            self.library.faces.search()
            .where(f"person_id = {UNASSIGNED}")
            .select(["face_id", "image_id", "x", "y", "w", "h", "quality"])
            .limit(limit * 4)
            .to_arrow()
            .to_pylist()
        )
        good = [r for r in rows if float(r["quality"]) >= min_quality]
        good.sort(key=lambda r: float(r["quality"]), reverse=True)
        return [{
            "face_id": int(r["face_id"]),
            "image_id": int(r["image_id"]),
            "bbox": [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])],
            "quality": round(float(r["quality"]), 3),
        } for r in good[:limit]]

    def person_suggestions(self, person_id: int, limit: int = 60) -> List[dict]:
        self.get_person(person_id)
        return suggest_for_person(
            self.library, int(person_id), self.face_backend.dim,
            self.settings.face_match_threshold, limit)

    def person_faces(self, person_id: int, limit: int = 200) -> List[dict]:
        """A person's faces, best quality first — the person review view."""
        self.get_person(person_id)
        rows = (
            self.library.faces.search()
            .where(f"person_id = {int(person_id)}")
            .select(["face_id", "image_id", "x", "y", "w", "h",
                     "quality", "confirmed"])
            .limit(limit)
            .to_arrow()
            .to_pylist()
        )
        rows.sort(key=lambda r: float(r["quality"] or 0.0), reverse=True)
        return [{
            "face_id": int(r["face_id"]),
            "image_id": int(r["image_id"]),
            "bbox": [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])],
            "quality": round(float(r["quality"]), 3),
            "confirmed": bool(r["confirmed"]),
        } for r in rows]

    def merge_suggestions(self, limit: int = 20,
                          min_similarity: Optional[float] = None) -> List[dict]:
        """Pairs of people who are probably the same person.

        Clustering errs on the side of splitting (merging two strangers is
        worse than showing one person twice), so a real library accumulates
        split identities. Comparing the stored centroids finds them; a pair
        is vetoed when the two people appear together in a photo, because
        two faces in the same frame are almost never the same person.
        """
        self.require_ready()
        if min_similarity is None:
            min_similarity = self.settings.face_cluster_threshold

        try:
            rows = self.library.people.to_lance().to_table(
                columns=["person_id", "name", "centroid", "hidden"]).to_pylist()
        except Exception:
            return []

        counts = self.index.person_counts()
        people = []
        for r in rows:
            if r["hidden"]:
                continue
            centroid = np.asarray(r["centroid"] or [], dtype=np.float32)
            if centroid.size == 0 or float(np.linalg.norm(centroid)) < 1e-6:
                continue
            people.append({
                "person_id": int(r["person_id"]),
                "name": r["name"] or "",
                "centroid": centroid,
                "photo_count": counts.get(int(r["person_id"]), 0),
            })
        if len(people) < 2:
            return []

        matrix = np.stack([p["centroid"] for p in people])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        sims = matrix @ matrix.T

        covers = self.people_by_id()
        pairs = []
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                similarity = float(sims[i, j])
                if similarity < min_similarity:
                    continue
                if self._people_cooccur(people[i]["person_id"],
                                        people[j]["person_id"]):
                    continue
                pairs.append((similarity, people[i], people[j]))

        pairs.sort(key=lambda t: t[0], reverse=True)

        def public(p: dict) -> dict:
            info = covers.get(p["person_id"], {})
            return {
                "person_id": p["person_id"],
                "name": p["name"],
                "photo_count": p["photo_count"],
                "face_count": info.get("face_count", 0),
                "cover_face_id": info.get("cover_face_id", -1),
            }

        out = []
        for similarity, a, b in pairs[:limit]:
            # Keep the named identity; failing that, the better-established one.
            a_keeps = (bool(a["name"]), a["photo_count"], -a["person_id"])
            b_keeps = (bool(b["name"]), b["photo_count"], -b["person_id"])
            target, source = (a, b) if a_keeps >= b_keeps else (b, a)
            out.append({
                "source": public(source),
                "target": public(target),
                "similarity": round(similarity, 4),
            })
        return out

    def _people_cooccur(self, person_a: int, person_b: int) -> bool:
        rows_a = self.index._rows_by_person.get(int(person_a))
        rows_b = self.index._rows_by_person.get(int(person_b))
        if rows_a is None or rows_b is None:
            return False
        return bool(np.intersect1d(rows_a, rows_b, assume_unique=False).size)

    # -- person bookkeeping ---------------------------------------------
    def _create_person(self, name: str = "") -> int:
        import pyarrow as pa

        from .db import people_schema

        person_id = self.library.next_id("people", "person_id")
        schema = people_schema(self.face_backend.dim)
        self.library.people.add(pa.Table.from_pylist([{
            "person_id": int(person_id),
            "name": name,
            "centroid": [0.0] * self.face_backend.dim,
            "face_count": 0,
            "cover_face_id": -1,
            "created_at": now_ms(),
            "hidden": False,
        }], schema=schema))
        self._people_cache = None
        return int(person_id)

    def _recompute_person(self, person_id: int) -> None:
        """Refresh a person's centroid, cover face, and count from their faces."""
        rows = (
            self.library.faces.search()
            .where(f"person_id = {int(person_id)}")
            .select(["face_id", "vector", "quality"])
            .limit(10_000)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            self.library.people.update(
                where=f"person_id = {int(person_id)}",
                values={"face_count": 0, "cover_face_id": -1})
            return

        vectors = np.stack([np.asarray(r["vector"], dtype=np.float32) for r in rows])
        weights = np.maximum(
            np.asarray([r["quality"] for r in rows], dtype=np.float32), 1e-3)
        centroid = (vectors * weights[:, None]).sum(axis=0) / weights.sum()
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        cover = int(rows[int(np.argmax(weights))]["face_id"])

        self.library.people.update(
            where=f"person_id = {int(person_id)}",
            values={"centroid": centroid.astype(np.float32).tolist(),
                    "face_count": len(rows),
                    "cover_face_id": cover})

    def _rewrite_image_people(self, person_ids: Sequence[int]) -> None:
        """Resync ``images.people_ids`` for every photo touching these people.

        ``images.people_ids`` is a denormalised copy of the face table that
        exists so browsing by person never has to join. Any edit to face
        ownership has to refresh it or the two disagree.
        """
        person_ids = [int(p) for p in person_ids if int(p) != UNASSIGNED]
        image_ids: set[int] = set()

        counts = self.index.person_counts()
        for pid in person_ids:
            if counts.get(pid):
                rows = self.index._rows_by_person.get(pid)
                if rows is not None:
                    image_ids.update(int(i) for i in self.index.image_ids[rows])
            hits = (
                self.library.faces.search()
                .where(f"person_id = {pid}")
                .select(["image_id"])
                .limit(100_000)
                .to_arrow()
            )
            image_ids.update(int(i) for i in hits["image_id"].to_pylist())

        if not image_ids:
            self.index.invalidate()
            return

        ids = sorted(image_ids)
        id_list = ", ".join(str(i) for i in ids)
        faces = (
            self.library.faces.search()
            .where(f"image_id IN ({id_list})")
            .select(["image_id", "person_id"])
            .limit(500_000)
            .to_arrow()
        )
        grouped: Dict[int, set] = {i: set() for i in ids}
        for image_id, person_id in zip(faces["image_id"].to_pylist(),
                                       faces["person_id"].to_pylist()):
            if int(person_id) != UNASSIGNED:
                grouped[int(image_id)].add(int(person_id))

        # Group images by the exact people set so identical updates share one
        # statement instead of issuing one per photo.
        by_signature: Dict[Tuple[int, ...], List[int]] = {}
        for image_id, people in grouped.items():
            by_signature.setdefault(tuple(sorted(people)), []).append(image_id)

        for signature, images in by_signature.items():
            for start in range(0, len(images), 4096):
                chunk = images[start:start + 4096]
                where = f"image_id IN ({', '.join(str(i) for i in chunk)})"
                if signature:
                    self.library.images.update(
                        where=where, values={"people_ids": list(signature)})
                else:
                    # An empty Python list has no inferable element type;
                    # make_array() writes a correctly-typed empty list.
                    self.library.images.update(
                        where=where, values_sql={"people_ids": "make_array()"})

        self.index.invalidate()

    # ------------------------------------------------------------------
    # Library-wide
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        if not self.ready:
            return {"ready": False, "total_images": 0, "total_people": 0,
                    "total_faces": 0}

        self.index.ensure_fresh()
        dated = self.index.taken_ts[~np.isnan(self.index.taken_ts)]
        people = self.list_people()
        meta = self.library.read_meta()

        return {
            "ready": True,
            "total_images": self.index.count,
            "total_people": len(people),
            "named_people": sum(1 for p in people if p["name"]),
            "total_faces": self.library.faces.count_rows(None),
            "unassigned_faces": self.library.faces.count_rows(
                f"person_id = {UNASSIGNED}"),
            "images_with_location": int((~np.isnan(self.index.lat)).sum()),
            "images_with_faces": int((self.index.face_count > 0).sum()),
            "images_without_date": int(np.isnan(self.index.taken_ts).sum()),
            "earliest_date": _iso(dated.min()) if dated.size else None,
            "latest_date": _iso(dated.max()) if dated.size else None,
            "embed_model": f"{meta.embed_backend}:{meta.embed_model}" if meta else None,
            "embed_dim": meta.image_dim if meta else None,
            "face_model": f"{meta.face_backend}:{meta.face_model}" if meta else None,
        }

    def timeline(self, filters: Optional[Filters] = None) -> List[dict]:
        self.require_ready()
        rows = self.index.select(filters) if filters else None
        return [{"month": month, "count": count}
                for month, count in self.index.month_histogram(rows)]

    def duplicates(self, max_distance: int = 6, limit: int = 200) -> List[dict]:
        """Groups of identical or near-identical photos."""
        self.require_ready()
        table = self.library.images.to_lance().to_table(
            columns=["image_id", "phash", "content_hash", "file_size"])
        pairs = [(int(i), int(h)) for i, h in
                 zip(table["image_id"].to_pylist(), table["phash"].to_pylist())
                 if h]
        exact: Dict[str, List[int]] = {}
        for image_id, chash in zip(table["image_id"].to_pylist(),
                                   table["content_hash"].to_pylist()):
            if chash:
                exact.setdefault(chash, []).append(int(image_id))

        groups: List[dict] = []
        seen: set = set()
        for chash, ids in exact.items():
            if len(ids) > 1:
                groups.append({"kind": "identical", "image_ids": sorted(ids)})
                seen.update(ids)

        for group in group_near_duplicates(pairs, max_distance=max_distance):
            remaining = [i for i in group if i not in seen]
            if len(remaining) > 1:
                groups.append({"kind": "similar", "image_ids": remaining})
        return groups[:limit]

    def folders(self) -> List[dict]:
        """Folder tree with counts, for browsing by where files live."""
        self.require_ready()
        self.index.ensure_fresh()
        counts: Dict[str, int] = {}
        for folder in self.index.folders:
            counts[folder] = counts.get(folder, 0) + 1
        return sorted(({"folder": f, "count": c} for f, c in counts.items()),
                      key=lambda d: d["folder"])

    def cameras(self) -> List[dict]:
        """Distinct camera models with counts — drives the camera filter."""
        self.require_ready()
        self.index.ensure_fresh()
        counts: Dict[str, int] = {}
        for camera in self.index.cameras:
            if camera:
                counts[camera] = counts.get(camera, 0) + 1
        return sorted(({"camera": c, "count": n} for c, n in counts.items()),
                      key=lambda d: (-d["count"], d["camera"]))

    # ------------------------------------------------------------------
    # Source folders (library roots)
    # ------------------------------------------------------------------
    # The library itself only knows folders that contain photos. The roots
    # file remembers what the *user* chose to index — a Pictures folder, an
    # external drive, a scattered project directory — so the UI can offer
    # "rescan everything" without asking them to retype paths.

    def _roots_file(self) -> Path:
        return Path(self.settings.state_dir) / "roots.json"

    def _read_roots(self) -> List[str]:
        import json

        try:
            data = json.loads(self._roots_file().read_text(encoding="utf-8"))
            roots = data.get("roots", [])
            return [str(r) for r in roots if isinstance(r, str)]
        except (OSError, ValueError):
            return []

    def _write_roots(self, roots: List[str]) -> None:
        import json

        path = self._roots_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"roots": roots}, indent=2), encoding="utf-8")

    def list_roots(self) -> List[dict]:
        roots = self._read_roots()
        counts: Dict[str, int] = {}
        if self.ready:
            self.index.ensure_fresh()
            for root in roots:
                prefix = root.replace("\\", "/").rstrip("/")
                counts[root] = sum(
                    1 for folder in self.index.folders
                    if (n := folder.replace("\\", "/").rstrip("/")) == prefix
                    or n.startswith(prefix + "/"))
        return [{
            "path": root,
            "exists": Path(root).is_dir(),
            "photo_count": counts.get(root, 0),
        } for root in roots]

    def add_root(self, folder: str) -> List[dict]:
        root = Path(folder).expanduser()
        if not root.is_dir():
            raise ValueError(f"{folder} is not a directory")
        resolved = str(root.resolve())
        roots = self._read_roots()
        # Windows paths are case-insensitive; don't list one drive twice.
        if resolved.lower() not in [r.lower() for r in roots]:
            roots.append(resolved)
            self._write_roots(roots)
        return self.list_roots()

    def remove_root(self, folder: str) -> List[dict]:
        """Forget a source folder. Its photos stay in the library."""
        target = str(Path(folder).expanduser()).lower()
        resolved = str(Path(folder).expanduser().resolve()).lower()
        roots = [r for r in self._read_roots()
                 if r.lower() not in (target, resolved)]
        self._write_roots(roots)
        return self.list_roots()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def start_index_job(self, folder: str, rebuild: bool = False,
                        prune_missing: bool = False):
        from .indexer import Indexer

        root = Path(folder).expanduser()
        if not root.is_dir():
            raise ValueError(f"{folder} is not a directory")
        try:
            self.add_root(str(root))
        except Exception:  # remembering the root must never block indexing
            logger.warning("Could not record %s as a library root", root)

        def run(progress) -> dict:
            indexer = Indexer(self.library, self.settings, self.embedder,
                              self.face_backend, self.thumbs, progress)
            stats = indexer.index_directory(
                root, rebuild=rebuild, prune_missing=prune_missing)
            self.index.invalidate()
            self._people_cache = None
            return stats.as_dict()

        return self.jobs.submit("index", run)

    def start_recluster_job(self, threshold: Optional[float] = None,
                            knn: Optional[int] = None):
        def run(progress) -> dict:
            progress("reclustering", 0, 1, {})
            result = recluster(
                self.library, self.face_backend.dim,
                threshold=threshold if threshold is not None
                else self.settings.face_cluster_threshold,
                knn=knn or self.settings.face_cluster_knn,
                min_cluster_size=self.settings.face_min_cluster_size)
            progress("resyncing", 1, 2, result)
            self._resync_all_image_people()
            self.index.invalidate()
            self._people_cache = None
            progress("done", 2, 2, result)
            return result

        return self.jobs.submit("recluster", run)

    def model_status(self) -> dict:
        from .models import status

        return status(self.settings)

    def start_fetch_models_job(self):
        """Download any missing model weights, with progress.

        Only the face model is ever fetched — the image/text model ships with
        the application. The desktop app calls this on first launch so the
        download is a visible, cancellable step rather than a silent stall
        the first time someone searches.
        """
        from .models import ensure_face_model

        def run(progress) -> dict:
            def on_bytes(name: str, done: int, total: int) -> None:
                progress("downloading", done, total, {"model": name})

            progress("downloading", 0, 1, {})
            path = ensure_face_model(self.settings, on_bytes)
            # Drop any cached backend so the next call picks up the weights.
            self._faces = None
            return {"installed": str(path), **status(self.settings)}

        from .models import status

        return self.jobs.submit("fetch_models", run)

    def start_compact_job(self):
        def run(progress) -> dict:
            progress("compacting", 0, 1, {})
            self.library.compact()
            report = self.library.build_indexes(self.settings.ann_min_rows)
            self.index.invalidate()
            progress("done", 1, 1, {})
            return {"indexes": report}

        return self.jobs.submit("compact", run)

    def _resync_all_image_people(self) -> None:
        """Rebuild every image's people_ids from the face table in one pass."""
        faces = self.library.faces.to_lance().to_table(
            columns=["image_id", "person_id"])
        grouped: Dict[int, set] = {}
        for image_id, person_id in zip(faces["image_id"].to_pylist(),
                                       faces["person_id"].to_pylist()):
            if int(person_id) == UNASSIGNED:
                continue
            grouped.setdefault(int(image_id), set()).add(int(person_id))

        images = self.library.images.to_lance().to_table(columns=["image_id"])
        all_ids = [int(i) for i in images["image_id"].to_pylist()]

        by_signature: Dict[Tuple[int, ...], List[int]] = {}
        for image_id in all_ids:
            signature = tuple(sorted(grouped.get(image_id, ())))
            by_signature.setdefault(signature, []).append(image_id)

        for signature, ids in by_signature.items():
            for start in range(0, len(ids), 4096):
                chunk = ids[start:start + 4096]
                where = f"image_id IN ({', '.join(str(i) for i in chunk)})"
                if signature:
                    self.library.images.update(
                        where=where, values={"people_ids": list(signature)})
                else:
                    self.library.images.update(
                        where=where, values_sql={"people_ids": "make_array()"})


def _image_dict(row) -> dict:
    people = row.get("people_ids")
    if people is None:
        people = []
    taken = row.get("taken_at")
    return {
        "image_id": int(row["image_id"]),
        "filename": row.get("filename") or Path(str(row.get("path", ""))).name,
        "taken_at": _iso_ts(taken),
        "lat": _opt_float(row.get("lat")),
        "lon": _opt_float(row.get("lon")),
        "place": row.get("place") or "",
        "people_ids": [int(p) for p in people],
        "face_count": int(row.get("face_count") or 0),
        "width": int(row.get("width") or 0),
        "height": int(row.get("height") or 0),
    }


def _iso(seconds: float) -> str:
    return datetime.utcfromtimestamp(float(seconds)).isoformat()


def _iso_ts(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return None


def _opt_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 6)
