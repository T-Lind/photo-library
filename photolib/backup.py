"""Checksummed curation backups with exact, content-addressed face corrections."""
from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter, defaultdict

from . import catalog
from .db import UNASSIGNED


def digest(data):
    return hashlib.sha256(json.dumps({k: v for k, v in data.items() if k != "checksum"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def library_id(service):
    value = catalog.records(service.library, "identity:").get("library")
    if value is None:
        value = uuid.uuid4().hex
        catalog.put(service.library, "identity:library", value)
    return value


def extend(service, data):
    images = service.library.images.to_lance().to_table(columns=["image_id", "content_hash", "path"]).to_pylist()
    image_of = {r["image_id"]: r for r in images}
    corrections = []
    for batch in service.library.faces.to_lance().to_batches(
            columns=["face_id", "image_id", "person_id", "x", "y", "w", "h", "confirmed"],
            filter="confirmed = true", batch_size=4096):
        for face in batch.to_pylist():
            image = image_of.get(face["image_id"])
            if image:
                corrections.append({"image_id": image["image_id"], "hash": image["content_hash"],
                    "path": image["path"], "bbox": [face[k] for k in ("x", "y", "w", "h")],
                    "person_id": face["person_id"]})
    data.update(version=2, library_id=library_id(service), corrections=corrections,
        identities=[{"person_id": p["person_id"], "name": p["name"], "hidden": p["hidden"]}
                    for p in service.people_by_id().values()],
        annotations=[{"hash": image_of[i]["content_hash"], "path": image_of[i]["path"],
                      "image_id": i, **v} for i, v in service.annotations().items() if i in image_of],
        saved_searches=service.saved_searches())
    data["checksum"] = digest(data)
    return data


def plan(service, data):
    if data.get("format") != "photolib-curation" or data.get("version") != 2:
        raise ValueError("Exact restore requires a version 2 photolib backup")
    if data.get("checksum") != digest(data):
        raise ValueError("Backup checksum does not match; nothing was restored")
    identities = {int(p["person_id"]): p for p in data["identities"]}
    if len(identities) != len(data["identities"]):
        raise ValueError("Backup contains duplicate identities")
    images = service.library.images.to_lance().to_table(columns=["image_id", "content_hash", "path"]).to_pylist()
    by_hash = defaultdict(list)
    for image in images:
        by_hash[image["content_hash"]].append(image)
    same_library = library_id(service) == data["library_id"]

    def locate(record):
        matches = by_hash.get(record["hash"], [])
        exact = [i for i in matches if (same_library and i["image_id"] == record["image_id"])
                 or catalog.path_key(i["path"]) == catalog.path_key(record["path"])]
        if len(exact) == 1:
            return exact[0]["image_id"]
        return matches[0]["image_id"] if len(matches) == 1 else None

    faces = defaultdict(list)
    for batch in service.library.faces.to_lance().to_batches(
            columns=["face_id", "image_id", "person_id", "x", "y", "w", "h"], batch_size=4096):
        for f in batch.to_pylist():
            faces[(f["image_id"], tuple(f[k] for k in ("x", "y", "w", "h")))].append(f)
    matched, unmatched = [], 0
    seen_faces = set()
    for correction in data["corrections"]:
        pid = int(correction["person_id"])
        if pid != UNASSIGNED and pid not in identities:
            raise ValueError("Backup correction references an unknown identity")
        hits = faces.get((locate(correction), tuple(correction["bbox"])), [])
        if len(hits) != 1:
            unmatched += 1
        else:
            if hits[0]["face_id"] in seen_faces:
                raise ValueError("Backup contains duplicate face corrections")
            seen_faces.add(hits[0]["face_id"])
            matched.append((correction, hits[0]))
    current = service.people_by_id()
    targets, claimed, conflicts = {}, set(), set()
    imported = catalog.records(service.library, "restored-person:")
    for pid, person in identities.items():
        key = f"{data['library_id']}:{pid}"
        old = pid if same_library else imported.get(key)
        if old not in current:
            candidates = Counter(f["person_id"] for c, f in matched if c["person_id"] == pid and f["person_id"] != UNASSIGNED)
            old = next((p for p, _ in candidates.most_common() if p not in claimed), None)
        if old in claimed:
            old = None  # preserve identity splits even if auto-clustering merged them
        if old in current and current[old]["name"] and current[old]["name"] != person["name"]:
            conflicts.add(pid)
            continue
        targets[pid] = old
        if old is not None:
            claimed.add(old)
    annotations = []
    for annotation in data.get("annotations", []):
        image_id = locate(annotation)
        rating = annotation["rating"]
        if not isinstance(rating, int) or not 0 <= rating <= 5 or not isinstance(annotation["favorite"], bool):
            raise ValueError("Backup contains an invalid rating or favorite")
        if image_id is not None:
            annotations.append((image_id, annotation))
    from .api.schemas import SearchRequest
    for saved in data.get("saved_searches", []):
        SearchRequest.model_validate(saved["request"])
    return {"verified": True, "faces_matched": sum(c["person_id"] not in conflicts for c, _ in matched),
            "faces_unmatched": unmatched, "identity_conflicts": len(conflicts),
            "annotations_matched": len(annotations), "_matched": matched,
            "_targets": targets, "_conflicts": conflicts, "_annotations": annotations}


def preview(service, data):
    return {k: v for k, v in plan(service, data).items() if not k.startswith("_")}


def restore(service, data):
    report = plan(service, data)
    targets = report["_targets"]
    # Keep the v1 import for backward-compatible name/album matching, then
    # restore the exact corrections, including confirmed unassigned faces.
    with catalog.atomic(service.library):
        legacy = service._import_curation_legacy(data)
        touched = set()
        for person in data["identities"]:
            pid = person["person_id"]
            if pid in report["_conflicts"]:
                continue
            target = targets[pid]
            if target is None:
                target = service._create_person(person["name"])
                targets[pid] = target
            service.rename_person(target, person["name"])
            service.set_person_hidden(target, person["hidden"])
            catalog.put(service.library, f"restored-person:{data['library_id']}:{pid}", target)
        for correction, face in report["_matched"]:
            pid = correction["person_id"]
            if pid in report["_conflicts"]:
                continue
            target = UNASSIGNED if pid == UNASSIGNED else targets[pid]
            service.library.faces.update(where=f"face_id = {face['face_id']}",
                values={"person_id": target, "confirmed": True})
            touched.update((face["person_id"], target))
        for pid in touched - {UNASSIGNED}:
            service._recompute_person(pid)
        service._resync_all_image_people()
        for image_id, annotation in report["_annotations"]:
            service.annotate(image_id, annotation["favorite"], annotation["rating"])
        for saved in data.get("saved_searches", []):
            payload = dict(saved["request"])
            payload["people_ids"] = [targets[p] for p in payload.get("people_ids", []) if p in targets]
            # Do not broaden a saved identity filter when its person conflicted.
            if len(payload["people_ids"]) == len(saved["request"].get("people_ids", [])):
                service.save_search(saved["name"], payload)
    service.index.invalidate()
    service._people_cache = None
    return {**legacy, **{k: v for k, v in report.items() if not k.startswith("_")}}
