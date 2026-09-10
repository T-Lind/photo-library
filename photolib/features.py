"""Curation, quality ranking, saved searches, and verified source relocation."""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa

from . import catalog, quality
from .db import UNASSIGNED
from .hashing import content_hash, hamming


class LibraryFeatures:
    def annotations(self, image_ids=None):
        if image_ids is None:
            rows = catalog.records(self.library, "annotation:")
        else:
            values = catalog.selected(self.library, (f"annotation:{int(i)}" for i in image_ids))
            rows = {key.removeprefix("annotation:"): value for key, value in values.items()}
        return {int(i): value for i, value in rows.items()}

    @catalog.serialized
    def annotate(self, image_id, favorite=None, rating=None):
        self._image_row(image_id, ["image_id"])
        value = catalog.get(self.library, f"annotation:{image_id}",
                            {"favorite": False, "rating": 0})
        if favorite is not None:
            value["favorite"] = bool(favorite)
        if rating is not None:
            if not 0 <= rating <= 5:
                raise ValueError("Rating must be between 0 and 5")
            value["rating"] = int(rating)
        catalog.put(self.library, f"annotation:{image_id}", value)
        return {"image_id": image_id, **value}

    def saved_searches(self):
        return sorted(catalog.records(self.library, "search:").values(), key=lambda v: v["name"].lower())

    @catalog.serialized
    def save_search(self, name, request):
        from .api.schemas import SearchRequest
        name = name.strip()
        if not name:
            raise ValueError("Give this search a name")
        payload = SearchRequest.model_validate(request).model_dump(mode="json")
        payload["page"] = 1
        existing = next((s for s in self.saved_searches() if s["name"].casefold() == name.casefold()), None)
        key = existing["id"] if existing else uuid.uuid4().hex
        value = {"id": key, "name": name, "request": payload}
        catalog.put(self.library, f"search:{key}", value)
        return value

    def quality_scores(self, image_ids, people_ids=()):
        ids = list(map(int, image_ids))
        measurements = {}
        face_quality = {}
        # Bound predicates and Arrow vector-free reads even for a full browse.
        for start in range(0, len(ids), 4096):
            chunk = ids[start:start+4096]
            measurements.update(quality.get(self.library, chunk))
            if people_ids:
                where = (f"image_id IN ({','.join(map(str,chunk))}) AND "
                         f"person_id IN ({','.join(str(int(p)) for p in people_ids)})")
                faces = self.library.faces.to_lance().to_table(
                    columns=["image_id", "person_id", "quality"], filter=where)
                for face in faces.to_pylist():
                    key = (face["image_id"], face["person_id"])
                    face_quality[key] = max(face_quality.get(key, 0), face["quality"] or 0)
        out = {}
        for image_id in ids:
            m = measurements.get(image_id)
            if m is None:
                out[image_id] = {"quality_score": None, "quality_reasons": ["Quality scan pending"]}
                continue
            score = float(m["score"])
            reasons = [f"Sharpness {m['sharpness']:.0%}", f"Exposure {m['exposure']:.0%}",
                       f"Usable resolution {m['resolution']:.0%}"]
            if people_ids:
                weakest = min(face_quality.get((image_id, p), 0) for p in people_ids)
                score = .5 * score + .5 * weakest
                reasons.append(f"Weakest selected face {weakest:.0%} (sharpness, size, confidence)")
            out[image_id] = {"quality_score": round(score, 4), "quality_reasons": reasons}
        return out

    def rank_quality(self, rows, people_ids, relevance=None):
        details = self.quality_scores(self.index.ids_of(rows), people_ids)
        def key(row):
            image_id = int(self.index.image_ids[row])
            score = details[image_id]["quality_score"]
            if score is None:
                return (1, 0, image_id)
            # When a query is present, relevance stays the dominant signal.
            combined = score if relevance is None else .75 * relevance.get(int(row), 0) + .25 * score
            return (0, -combined, image_id)
        return np.asarray(sorted(rows.tolist(), key=key), dtype=np.int64), details

    def start_quality_job(self):
        def run(progress):
            from .imageio import load_rgb_array
            known = quality.get(self.library)
            dataset = self.library.images.to_lance()
            total = dataset.count_rows()
            scanned = failed = 0
            for batch in dataset.to_batches(columns=["image_id", "path", "width", "height", "media_type"], batch_size=64):
                rows = []
                for row in batch.to_pylist():
                    progress("quality", scanned, total, {"failed": failed})
                    scanned += 1
                    if row["image_id"] in known or row["media_type"] == "video":
                        continue
                    try:
                        measured = quality.measure(load_rgb_array(row["path"], max_side=1600), row["width"], row["height"])
                        rows.append({"image_id": row["image_id"], **measured})
                    except Exception:
                        failed += 1
                with catalog.atomic(self.library):
                    quality.save(self.library, rows)
            return {"scanned": scanned, "failed": failed}
        return self.jobs.submit("quality", run)

    def failures(self):
        return list(catalog.records(self.library, "failure:").values())

    def start_retry_job(self):
        def run(progress):
            from .indexer import Indexer
            worker = Indexer(self.library, self.settings, self.embedder,
                             self.face_backend, self.thumbs, progress)
            try:
                return worker.retry_failed()
            finally:
                self.index.invalidate()
                self._people_cache = None
        return self.jobs.submit("retry", run)

    @catalog.serialized
    def relocate_root(self, old_folder, new_folder, verification=None):
        old = Path(old_folder).expanduser().absolute()
        new = Path(new_folder).expanduser().resolve()
        roots = self._read_roots()
        source = next((r for r in roots if catalog.path_key(r) == catalog.path_key(old)), None)
        if source is None or not new.is_dir():
            raise ValueError("Choose a known source and an accessible destination folder")
        if catalog.path_key(old) == catalog.path_key(new):
            raise ValueError("Choose a different destination")
        if new.is_relative_to(old) or old.is_relative_to(new):
            raise ValueError("Source and destination must not contain each other")
        matches = []
        all_paths = set()
        for batch in self.library.images.to_lance().to_batches(
                columns=["image_id", "path", "content_hash"], batch_size=4096):
            for row in batch.to_pylist():
                all_paths.add(catalog.path_key(row["path"]))
                try:
                    relative = Path(row["path"]).relative_to(old)
                except ValueError:
                    continue
                target = new / relative
                if not target.is_file():
                    raise ValueError(f"Destination is missing {relative}; no paths changed")
                if catalog.path_key(target) in all_paths:
                    raise ValueError("Destination already contains indexed photos")
                matches.append((row, target))
        if not matches:
            raise ValueError("No indexed photos belong to this source")
        if any(catalog.path_key(target) in all_paths for _, target in matches):
            raise ValueError("Destination already contains indexed photos")
        # Verify every original before changing any path. This runs in the
        # API worker; originals are read but never moved or overwritten.
        for row, target in matches:
            if content_hash(target) != row["content_hash"]:
                raise ValueError(f"Content differs at {target.name}; no paths changed")
        summary = {"old_folder": str(old), "new_folder": str(new), "photos": len(matches),
                   "version": self.library.images.version}
        if verification is None:
            token = uuid.uuid4().hex
            catalog.put(self.library, f"relocation:{token}", summary)
            return {**summary, "verification": token, "verified": True}
        pending = catalog.records(self.library, "relocation:").get(verification)
        if pending != summary:
            raise ValueError("Library changed since verification; verify the destination again")
        with catalog.atomic(self.library):
            for start in range(0, len(matches), 256):
                chunk = matches[start:start+256]
                ids = ",".join(str(r["image_id"]) for r, _ in chunk)
                table = self.library.images.to_lance().to_table(filter=f"image_id IN ({ids})")
                target_of = {r["image_id"]: target for r, target in chunk}
                rows = table.to_pylist()
                for row in rows:
                    target = target_of[row["image_id"]]
                    row.update(path=str(target), folder=str(target.parent), filename=target.name,
                               mtime=target.stat().st_mtime, file_size=target.stat().st_size)
                self.library.images.merge_insert("image_id").when_matched_update_all().execute(
                    pa.Table.from_pylist(rows, schema=table.schema))
            self._write_roots([str(new) if r == source else r for r in roots])
            source_ids = catalog.records(self.library, "source:")
            for key, value in source_ids.items():
                if catalog.path_key(value["path"]) == catalog.path_key(source):
                    catalog.put(self.library, f"source:{key}", {**value, "path": str(new)})
            for saved in self.saved_searches():
                folder = saved["request"].get("folder")
                if folder:
                    try:
                        saved["request"]["folder"] = str(new / Path(folder).relative_to(old))
                        catalog.put(self.library, f"search:{saved['id']}", saved)
                    except ValueError:
                        pass
            for key, failure in catalog.records(self.library, "failure:").items():
                try:
                    destination = new / Path(failure["path"]).relative_to(old)
                    catalog.put(self.library, "failure:" + catalog.path_key(destination),
                                {**failure, "path": str(destination)})
                    catalog.delete(self.library, "failure:" + key)
                except ValueError:
                    pass
            catalog.delete(self.library, f"relocation:{verification}")
        self.index.invalidate()
        return {"relocated": len(matches), "roots": self.list_roots()}

    def burst_candidates(self, image_id):
        anchor = self._image_row(image_id, ["taken_at", "phash", "camera", "folder", "media_type"])
        if anchor["taken_at"] is None or anchor["media_type"] == "video":
            return {"images": self.hydrate([image_id]), "suggested_keeper": image_id}
        from datetime import timezone
        self.index.ensure_fresh()
        # Arrow's timezone-free capture dates use UTC-like numeric storage.
        # datetime.timestamp() on a naive value would apply this PC's timezone.
        timestamp = anchor["taken_at"].replace(tzinfo=timezone.utc).timestamp()
        rows = np.flatnonzero((np.abs(self.index.taken_ts - timestamp) <= 10) & ~self.index.is_video)
        candidates = []
        for start in range(0, len(rows), 4096):
            ids = self.index.ids_of(rows[start:start+4096])
            if not ids:
                continue
            table = self.library.images.to_lance().to_table(columns=["image_id", "phash", "camera", "folder"],
                filter=f"image_id IN ({','.join(map(str,ids))})")
            for r in table.to_pylist():
                if r["folder"] == anchor["folder"] and r["camera"] == anchor["camera"] and r["phash"] is not None and anchor["phash"] is not None and hamming(r["phash"], anchor["phash"]) <= 12:
                    candidates.append(r["image_id"])
        if image_id not in candidates:
            candidates.append(image_id)
        scores = self.quality_scores(candidates)
        candidates.sort(key=lambda i: (scores[i]["quality_score"] is None, -(scores[i]["quality_score"] or 0), i))
        return {"images": [{**r, **scores[r["image_id"]]} for r in self.hydrate(candidates[:60])],
                "total": len(candidates), "suggested_keeper": candidates[0]}
