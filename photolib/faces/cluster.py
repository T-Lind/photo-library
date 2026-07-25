"""Face identity assignment and clustering.

The old pipeline ran one DBSCAN over every face encoding in the library.
That has three problems that show up immediately on a real collection:

1. It is O(n^2) in the worst case and has to hold every encoding in memory,
   so it simply does not run on a 200k-photo library (which can easily hold
   half a million faces).
2. It is all-or-nothing. Adding one photo meant re-running everything, so
   incremental indexing had to fall back to a cached JSON blob.
3. It clusters on *centroid distance to a mean encoding*, which drifts badly
   when a cluster picks up one wrong face, and it treats a blurry 30-pixel
   background face as equal evidence to a sharp portrait.

This module replaces it with an incremental assigner that does bounded work
per face, plus an offline reclustering pass for when you want to redo
everything from scratch.

Key ideas
---------
* **Quality-weighted centroids.** A face contributes to its person's centroid
  in proportion to how good the crop is. Junk faces never move the model.
* **Multi-exemplar matching.** A person is represented by their centroid
  *and* their best few faces. People change (age, beards, glasses, lighting)
  and a single mean vector cannot represent that; the nearest-exemplar
  similarity can.
* **Ambiguity handling.** If a face is nearly equidistant between two people,
  it is still shown under the best match, but it is not allowed to update
  either person's model. Guessing *and* learning from the guess is how
  clusters merge into one useless blob.
* **One-person-per-photo.** Two faces in the same photograph are almost
  never the same person, so a duplicate assignment within an image is
  resolved in favour of the more confident face.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..db import FACES, PEOPLE, UNASSIGNED, Library, now_ms, people_schema

logger = logging.getLogger(__name__)

# A face this poor never creates a new identity and never updates one.
MIN_QUALITY_FOR_IDENTITY = 0.25
# Minimum lead the best match needs over the runner-up to be trusted.
AMBIGUITY_MARGIN = 0.04
# Exemplars kept in memory per person during a run.
EXEMPLARS_PER_PERSON = 6
# People whose exemplars are cached at once (bounds memory on huge libraries).
EXEMPLAR_CACHE_PEOPLE = 4096


@dataclass
class FaceObservation:
    """A detected face awaiting an identity."""

    image_id: int
    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]
    det_score: float
    quality: float
    face_id: int = -1
    person_id: int = UNASSIGNED
    confirmed: bool = False


@dataclass
class PersonState:
    person_id: int
    name: str = ""
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    weight: float = 0.0        # sum of quality weights folded into the centroid
    face_count: int = 0
    cover_face_id: int = -1
    cover_quality: float = -1.0
    dirty: bool = False
    is_new: bool = False

    def absorb(self, embedding: np.ndarray, quality: float) -> None:
        """Fold a face into the running quality-weighted mean."""
        w = max(float(quality), 1e-3)
        if self.weight <= 0 or self.centroid.size == 0:
            self.centroid = embedding.astype(np.float32).copy()
            self.weight = w
        else:
            total = self.weight + w
            self.centroid = (self.centroid * self.weight + embedding * w) / total
            self.weight = total
        # Keep the centroid on the unit sphere so cosine stays a dot product.
        norm = float(np.linalg.norm(self.centroid))
        if norm > 1e-12:
            self.centroid = (self.centroid / norm).astype(np.float32)
        self.dirty = True


class ExemplarCache:
    """LRU cache of each person's best face embeddings."""

    def __init__(self, library: Library, dim: int, per_person: int = EXEMPLARS_PER_PERSON,
                 capacity: int = EXEMPLAR_CACHE_PEOPLE):
        self._library = library
        self._dim = dim
        self._per_person = per_person
        self._capacity = capacity
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._pending: Dict[int, List[Tuple[float, np.ndarray]]] = defaultdict(list)

    def get(self, person_id: int) -> np.ndarray:
        pending = self._pending.get(person_id)
        cached = self._cache.get(person_id)
        if cached is None:
            cached = self._load(person_id)
            self._cache[person_id] = cached
            self._cache.move_to_end(person_id)
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
        if pending:
            extra = np.stack([e for _, e in pending])
            return np.concatenate([cached, extra]) if cached.size else extra
        return cached

    def add(self, person_id: int, embedding: np.ndarray, quality: float) -> None:
        """Register a freshly-assigned face so later faces in the same run see it."""
        bucket = self._pending[person_id]
        bucket.append((quality, embedding.astype(np.float32)))
        if len(bucket) > self._per_person:
            bucket.sort(key=lambda t: t[0], reverse=True)
            del bucket[self._per_person:]

    def _load(self, person_id: int) -> np.ndarray:
        try:
            rows = (
                self._library.faces.search()
                .where(f"person_id = {person_id}")
                .select(["vector", "quality"])
                .limit(self._per_person * 4)
                .to_arrow()
                .to_pylist()
            )
        except Exception as exc:  # pragma: no cover - empty/missing table
            logger.debug("Exemplar load failed for person %d: %s", person_id, exc)
            return np.zeros((0, self._dim), dtype=np.float32)

        if not rows:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows.sort(key=lambda r: float(r["quality"] or 0.0), reverse=True)
        return np.stack([np.asarray(r["vector"], dtype=np.float32)
                         for r in rows[:self._per_person]])


class FaceAssigner:
    """Assigns face observations to people, creating people as needed."""

    def __init__(self, library: Library, dim: int, match_threshold: float = 0.38,
                 strong_threshold: float = 0.55, min_quality: float = MIN_QUALITY_FOR_IDENTITY):
        self.library = library
        self.dim = dim
        self.match_threshold = match_threshold
        self.strong_threshold = strong_threshold
        self.min_quality = min_quality

        self.people: Dict[int, PersonState] = {}
        self._ids: List[int] = []
        self._matrix: Optional[np.ndarray] = None   # (P, dim) stacked centroids
        self._next_person_id = 0
        self.exemplars = ExemplarCache(library, dim)
        self._load_people()

    # -- loading / persistence ------------------------------------------
    def _load_people(self) -> None:
        try:
            tbl = self.library.people
            rows = tbl.to_lance().to_table(
                columns=["person_id", "name", "centroid", "face_count",
                         "cover_face_id"]).to_pylist()
        except Exception:  # pragma: no cover - table may not exist yet
            rows = []

        for r in rows:
            centroid = np.asarray(r["centroid"] or [], dtype=np.float32)
            count = int(r["face_count"] or 0)
            self.people[int(r["person_id"])] = PersonState(
                person_id=int(r["person_id"]),
                name=r["name"] or "",
                centroid=centroid,
                # Reconstruct an approximate weight so old faces still
                # outweigh a single new one.
                weight=float(max(count, 1)) * 0.6,
                face_count=count,
                cover_face_id=(int(r["cover_face_id"])
                               if r["cover_face_id"] is not None else -1),
            )
        self._next_person_id = (max(self.people) + 1) if self.people else 0
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        usable = [p for p in self.people.values() if p.centroid.size == self.dim]
        self._ids = [p.person_id for p in usable]
        self._matrix = (np.stack([p.centroid for p in usable])
                        if usable else np.zeros((0, self.dim), dtype=np.float32))

    # -- matching --------------------------------------------------------
    def _candidates(self, embedding: np.ndarray, shortlist: int = 5
                    ) -> List[Tuple[int, float]]:
        """Score people against one face: centroid first, exemplars to refine."""
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        sims = self._matrix @ embedding
        # Shortlist on the cheap centroid score, then pay for exemplars only
        # on the handful of people that could plausibly win.
        k = min(shortlist, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k] if sims.shape[0] > k else np.arange(sims.shape[0])

        scored: List[Tuple[int, float]] = []
        floor = self.match_threshold - 0.15
        for idx in top:
            person_id = self._ids[int(idx)]
            centroid_sim = float(sims[int(idx)])
            score = centroid_sim
            if centroid_sim >= floor:
                ex = self.exemplars.get(person_id)
                if ex.size:
                    score = max(score, float(np.max(ex @ embedding)))
            scored.append((person_id, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def _create_person(self, obs: FaceObservation) -> PersonState:
        person = PersonState(person_id=self._next_person_id, is_new=True, dirty=True)
        self._next_person_id += 1
        self.people[person.person_id] = person
        return person

    def assign(self, observations: Sequence[FaceObservation]) -> None:
        """Assign each observation in place, grouped per image.

        Observations must already carry their ``face_id``; the caller owns id
        allocation so that face rows and person links stay consistent.
        """
        by_image: Dict[int, List[FaceObservation]] = defaultdict(list)
        for obs in observations:
            by_image[obs.image_id].append(obs)

        for image_id, faces in by_image.items():
            # Best faces first: a sharp portrait should claim the identity
            # before a blurry one in the same frame gets a chance to.
            faces.sort(key=lambda f: (f.quality, f.det_score), reverse=True)
            claimed: Dict[int, float] = {}

            for obs in faces:
                usable = obs.quality >= self.min_quality
                scored = self._candidates(obs.embedding)
                # Drop people already claimed by a better face in this photo.
                scored = [(pid, s) for pid, s in scored if pid not in claimed]

                best_id, best_score = (scored[0] if scored else (None, -1.0))
                runner_up = scored[1][1] if len(scored) > 1 else -1.0
                margin = best_score - runner_up

                if best_id is not None and best_score >= self.strong_threshold:
                    self._attach(obs, best_id, learn=usable)
                elif (best_id is not None and best_score >= self.match_threshold
                      and margin >= AMBIGUITY_MARGIN):
                    self._attach(obs, best_id, learn=usable)
                elif best_id is not None and best_score >= self.match_threshold:
                    # Ambiguous: show it, but do not let it teach either person.
                    self._attach(obs, best_id, learn=False)
                elif usable:
                    person = self._create_person(obs)
                    self._attach(obs, person.person_id, learn=True)
                else:
                    # Too poor to identify and too poor to seed a new person.
                    obs.person_id = UNASSIGNED
                    continue

                claimed[obs.person_id] = obs.quality

            self._rebuild_matrix()

    def _attach(self, obs: FaceObservation, person_id: int, learn: bool) -> None:
        person = self.people[person_id]
        obs.person_id = person_id
        person.face_count += 1
        person.dirty = True
        if learn:
            person.absorb(obs.embedding, obs.quality)
            self.exemplars.add(person_id, obs.embedding, obs.quality)
        if obs.quality > person.cover_quality:
            person.cover_quality = obs.quality
            person.cover_face_id = obs.face_id

    # -- writing back ----------------------------------------------------
    def flush(self) -> Tuple[int, int]:
        """Persist new and modified people. Returns (created, updated)."""
        new_rows = []
        updated = 0
        schema = people_schema(self.dim)
        created_at = now_ms()

        for person in self.people.values():
            if not person.dirty:
                continue
            if person.is_new:
                new_rows.append({
                    "person_id": person.person_id,
                    "name": person.name,
                    "centroid": person.centroid.tolist(),
                    "face_count": person.face_count,
                    "cover_face_id": person.cover_face_id,
                    "created_at": created_at,
                    "hidden": False,
                })
                person.is_new = False
            else:
                self.library.people.update(
                    where=f"person_id = {person.person_id}",
                    values={
                        "centroid": person.centroid.tolist(),
                        "face_count": person.face_count,
                        "cover_face_id": person.cover_face_id,
                    },
                )
                updated += 1
            person.dirty = False

        if new_rows:
            import pyarrow as pa

            self.library.people.add(pa.Table.from_pylist(new_rows, schema=schema))
        return len(new_rows), updated


# ---------------------------------------------------------------------------
# Offline reclustering
# ---------------------------------------------------------------------------

def mutual_knn_components(embeddings: np.ndarray, k: int, threshold: float,
                          block: int = 2048) -> np.ndarray:
    """Cluster unit vectors via connected components of a mutual-kNN graph.

    Two faces are linked only when *each* is in the other's top-k and their
    cosine similarity clears ``threshold``. Requiring mutuality is what stops
    the chaining failure that makes plain single-linkage (and DBSCAN with a
    loose eps) merge separate people through a few ambiguous faces.

    Similarities are computed in blocks, so peak memory is
    ``block x n x 4`` bytes rather than ``n^2 x 4``.

    Returns an array of cluster labels, one per row.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)

    k = min(k, n - 1)
    neighbours = np.empty((n, k), dtype=np.int32)
    sims = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = embeddings[start:stop] @ embeddings.T           # (b, n)
        rows = np.arange(start, stop)
        chunk[np.arange(stop - start), rows] = -np.inf          # drop self-match
        idx = np.argpartition(-chunk, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(chunk, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        neighbours[start:stop] = np.take_along_axis(idx, order, axis=1)
        sims[start:stop] = np.take_along_axis(vals, order, axis=1)

    neighbour_sets = [set(neighbours[i][sims[i] >= threshold].tolist()) for i in range(n)]

    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for i in range(n):
        for j in neighbour_sets[i]:
            if i in neighbour_sets[j]:       # mutual
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[rj] = ri

    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32)


def recluster(library: Library, dim: int, threshold: float, knn: int,
              min_cluster_size: int = 2, respect_confirmed: bool = True,
              max_faces: int = 400_000) -> Dict[str, int]:
    """Rebuild every person from scratch from the stored face embeddings.

    Confirmed faces act as anchors: any cluster containing confirmed faces
    inherits that person's id and name, so a manual correction survives a
    recluster instead of being silently undone.
    """
    faces_tbl = library.faces
    total = faces_tbl.count_rows(None)
    if total == 0:
        return {"faces": 0, "people": 0}
    if total > max_faces:
        raise ValueError(
            f"Refusing to recluster {total} faces in one pass (limit {max_faces}). "
            "Raise max_faces if the machine has the memory, or keep using "
            "incremental assignment, which stays accurate without a full pass."
        )

    tbl = faces_tbl.to_lance().to_table(
        columns=["face_id", "vector", "person_id", "quality", "confirmed", "image_id"])
    face_ids = np.asarray(tbl["face_id"].to_pylist(), dtype=np.int64)
    qualities = np.asarray(tbl["quality"].to_pylist(), dtype=np.float32)
    confirmed = np.asarray([bool(c) for c in tbl["confirmed"].to_pylist()], dtype=bool)
    prior_person = np.asarray(tbl["person_id"].to_pylist(), dtype=np.int32)
    embeddings = np.asarray(
        [np.asarray(v, dtype=np.float32) for v in tbl["vector"].to_pylist()],
        dtype=np.float32)

    labels = mutual_knn_components(embeddings, k=knn, threshold=threshold)

    # Faces too poor to identify are never used to define a cluster.
    usable = qualities >= MIN_QUALITY_FOR_IDENTITY

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)

    assignments = np.full(len(face_ids), UNASSIGNED, dtype=np.int32)
    people_rows = []
    next_id = 0
    reserved = set()

    if respect_confirmed:
        reserved = {int(p) for p, c in zip(prior_person, confirmed) if c and p != UNASSIGNED}
        next_id = (max(reserved) + 1) if reserved else 0

    created_at = now_ms()
    name_by_id = _existing_names(library)

    for members in clusters.values():
        good = [i for i in members if usable[i]]
        if len(good) < min_cluster_size and not any(confirmed[i] for i in members):
            continue

        anchor_ids = {int(prior_person[i]) for i in members
                      if respect_confirmed and confirmed[i] and prior_person[i] != UNASSIGNED}
        if len(anchor_ids) == 1:
            person_id = anchor_ids.pop()
        else:
            # No anchor, or a cluster that merged two confirmed identities —
            # in the ambiguous case a fresh id is safer than picking one.
            while next_id in reserved:
                next_id += 1
            person_id = next_id
            next_id += 1

        idx = np.array(good if good else members)
        weights = np.maximum(qualities[idx], 1e-3)
        centroid = (embeddings[idx] * weights[:, None]).sum(axis=0) / weights.sum()
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        cover = int(face_ids[idx[int(np.argmax(qualities[idx]))]])

        assignments[np.array(members)] = person_id
        people_rows.append({
            "person_id": int(person_id),
            "name": name_by_id.get(int(person_id), ""),
            "centroid": centroid.astype(np.float32).tolist(),
            "face_count": len(members),
            "cover_face_id": cover,
            "created_at": created_at,
            "hidden": False,
        })

    _rewrite_people(library, dim, people_rows)
    _rewrite_face_assignments(library, face_ids, assignments)

    return {"faces": int(len(face_ids)), "people": len(people_rows),
            "unassigned": int((assignments == UNASSIGNED).sum())}


def _existing_names(library: Library) -> Dict[int, str]:
    try:
        rows = library.people.to_lance().to_table(
            columns=["person_id", "name"]).to_pylist()
    except Exception:  # pragma: no cover
        return {}
    return {int(r["person_id"]): (r["name"] or "") for r in rows}


def _rewrite_people(library: Library, dim: int, rows: List[dict]) -> None:
    import pyarrow as pa

    schema = people_schema(dim)
    library.db.drop_table(PEOPLE)
    tbl = library.db.create_table(PEOPLE, schema=schema)
    if rows:
        tbl.add(pa.Table.from_pylist(rows, schema=schema))


def _rewrite_face_assignments(library: Library, face_ids: np.ndarray,
                              assignments: np.ndarray) -> None:
    """Apply new person ids with one UPDATE per distinct person.

    Updating row-by-row would issue hundreds of thousands of statements;
    grouping by target id keeps it to one statement per person.
    """
    faces_tbl = library.faces
    by_person: Dict[int, List[int]] = defaultdict(list)
    for fid, pid in zip(face_ids.tolist(), assignments.tolist()):
        by_person[int(pid)].append(int(fid))

    for person_id, ids in by_person.items():
        for start in range(0, len(ids), 4096):
            chunk = ids[start:start + 4096]
            id_list = ", ".join(str(i) for i in chunk)
            faces_tbl.update(where=f"face_id IN ({id_list})",
                             values={"person_id": person_id})


def suggest_for_person(library: Library, person_id: int, dim: int,
                       threshold: float, limit: int = 60) -> List[dict]:
    """Unassigned faces that plausibly belong to ``person_id``.

    Powers the "is this also Alice?" review queue — the cheapest way to get a
    library from 80% correct to ~99% correct is to let a human confirm a page
    of near-misses instead of hunting for them.
    """
    people = (
        library.people.search()
        .where(f"person_id = {person_id}")
        .select(["centroid"])
        .limit(1)
        .to_arrow()
        .to_pylist()
    )
    if not people:
        return []
    centroid = np.asarray(people[0]["centroid"] or [], dtype=np.float32)
    if centroid.size != dim:
        return []

    rows = (
        library.faces.search(centroid.tolist())
        .metric("cosine")
        .where(f"person_id = {UNASSIGNED}", prefilter=True)
        .select(["face_id", "image_id", "quality", "x", "y", "w", "h"])
        .limit(limit * 3)
        .to_arrow()
        .to_pylist()
    )
    out = []
    for row in rows:
        # LanceDB returns cosine *distance*; similarity is 1 - distance.
        similarity = 1.0 - float(row.get("_distance", 1.0))
        if similarity < threshold:
            continue
        out.append({
            "face_id": int(row["face_id"]),
            "image_id": int(row["image_id"]),
            "similarity": round(similarity, 4),
            "quality": float(row["quality"]),
            "bbox": [int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])],
        })
        if len(out) >= limit:
            break
    return out
