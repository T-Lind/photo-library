"""LanceDB schema, connection management, and index maintenance.

Schema v2 stores one row per image *and* one row per detected face. Keeping
faces as first-class rows (rather than a JSON sidecar) is what makes
"find every photo with this face" a single ANN query instead of a full scan,
and it lets a person's identity be corrected face-by-face.

All embedding vectors are stored L2-normalised and every vector index uses
the cosine metric, which is what both SigLIP/CLIP and ArcFace are trained
for. The previous version stored raw CLIP outputs and searched them with L2,
which ranks partly by vector magnitude — an artefact, not a similarity.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

IMAGES = "images"
FACES = "faces"
PEOPLE = "people"
META = "meta"
# Extracted text, one row per scanned image. A separate table rather than an
# images column so that adding OCR to an existing library is an append, not
# a schema migration — no re-embedding, no version bump.
OCR = "ocr"
# User-curated collections. Also created lazily, for the same reason.
ALBUMS = "albums"
ALBUM_ITEMS = "album_items"

UNASSIGNED = -1  # person_id for a face that belongs to no person yet

# Columns safe to hand to clients / load into the browse index. Never
# includes `vector`: pulling 200k embeddings through pandas to render a grid
# is the single easiest way to make this system feel slow.
IMAGE_LIST_COLUMNS = [
    "image_id", "path", "filename", "taken_at", "lat", "lon", "place",
    "people_ids", "face_count", "width", "height",
]

BROWSE_COLUMNS = ["image_id", "taken_at", "people_ids", "lat", "lon", "face_count"]


def images_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("image_id", pa.int64(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
        pa.field("path", pa.string(), nullable=False),
        pa.field("filename", pa.string()),
        pa.field("folder", pa.string()),
        pa.field("taken_at", pa.timestamp("ms")),
        pa.field("added_at", pa.timestamp("ms")),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("file_size", pa.int64()),
        pa.field("mtime", pa.float64()),
        # sha256 prefix of the file bytes — exact-duplicate detection
        pa.field("content_hash", pa.string()),
        # 64-bit perceptual hash — near-duplicate / burst detection
        pa.field("phash", pa.int64()),
        pa.field("lat", pa.float64()),
        pa.field("lon", pa.float64()),
        pa.field("place", pa.string()),
        pa.field("camera", pa.string()),
        pa.field("people_ids", pa.list_(pa.int32())),
        pa.field("face_count", pa.int32()),
    ])


def faces_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("face_id", pa.int64(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
        pa.field("image_id", pa.int64(), nullable=False),
        pa.field("person_id", pa.int32(), nullable=False),
        pa.field("x", pa.int32()),
        pa.field("y", pa.int32()),
        pa.field("w", pa.int32()),
        pa.field("h", pa.int32()),
        pa.field("det_score", pa.float32()),
        pa.field("quality", pa.float32()),
        # True once a human has confirmed/assigned this face; automatic
        # reclustering must never move a confirmed face.
        pa.field("confirmed", pa.bool_()),
        pa.field("crop_path", pa.string()),
    ])


def people_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("person_id", pa.int32(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("centroid", pa.list_(pa.float32(), list_size=dim)),
        pa.field("face_count", pa.int32()),
        pa.field("cover_face_id", pa.int64()),
        pa.field("created_at", pa.timestamp("ms")),
        pa.field("hidden", pa.bool_()),
    ])


META_SCHEMA = pa.schema([
    pa.field("key", pa.string(), nullable=False),
    pa.field("value", pa.string()),
])

# A row exists for every image that has been *scanned*, even when no text
# was found — that is what lets a backfill know what is left to do.
OCR_SCHEMA = pa.schema([
    pa.field("image_id", pa.int64(), nullable=False),
    pa.field("text", pa.string()),
    pa.field("engine", pa.string()),
    pa.field("updated_at", pa.timestamp("ms")),
])

ALBUMS_SCHEMA = pa.schema([
    pa.field("album_id", pa.int32(), nullable=False),
    pa.field("name", pa.string()),
    pa.field("created_at", pa.timestamp("ms")),
    pa.field("cover_image_id", pa.int64()),
])

ALBUM_ITEMS_SCHEMA = pa.schema([
    pa.field("album_id", pa.int32(), nullable=False),
    pa.field("image_id", pa.int64(), nullable=False),
    pa.field("added_at", pa.timestamp("ms")),
])


@dataclass(frozen=True)
class LibraryMeta:
    schema_version: int
    image_dim: int
    face_dim: int
    embed_backend: str
    embed_model: str
    face_backend: str
    face_model: str

    def as_rows(self) -> List[Dict[str, str]]:
        return [{"key": k, "value": str(v)} for k, v in {
            "schema_version": self.schema_version,
            "image_dim": self.image_dim,
            "face_dim": self.face_dim,
            "embed_backend": self.embed_backend,
            "embed_model": self.embed_model,
            "face_backend": self.face_backend,
            "face_model": self.face_model,
        }.items()]


class SchemaMismatch(RuntimeError):
    """Raised when the configured models disagree with what's on disk."""


class Library:
    """Thin, thread-safe handle around a LanceDB photo library."""

    def __init__(self, uri: str):
        self.uri = uri
        self._db = lancedb.connect(uri)
        self._lock = threading.RLock()

    # -- connection ------------------------------------------------------
    @property
    def db(self):
        return self._db

    def table_names(self) -> List[str]:
        return list(self._db.table_names())

    def table(self, name: str):
        return self._db.open_table(name)

    @property
    def images(self):
        return self._db.open_table(IMAGES)

    @property
    def faces(self):
        return self._db.open_table(FACES)

    @property
    def people(self):
        return self._db.open_table(PEOPLE)

    @property
    def ocr(self):
        return self._db.open_table(OCR)

    def has_ocr(self) -> bool:
        return OCR in self.table_names()

    def ensure_ocr(self):
        """Create the OCR table on first use — old libraries gain it in place."""
        with self._lock:
            if OCR not in self.table_names():
                self._db.create_table(OCR, schema=OCR_SCHEMA)
        return self._db.open_table(OCR)

    @property
    def albums(self):
        return self._db.open_table(ALBUMS)

    @property
    def album_items(self):
        return self._db.open_table(ALBUM_ITEMS)

    def has_albums(self) -> bool:
        return ALBUMS in self.table_names()

    def ensure_albums(self) -> None:
        with self._lock:
            names = set(self.table_names())
            if ALBUMS not in names:
                self._db.create_table(ALBUMS, schema=ALBUMS_SCHEMA)
            if ALBUM_ITEMS not in names:
                self._db.create_table(ALBUM_ITEMS, schema=ALBUM_ITEMS_SCHEMA)

    def initialised(self) -> bool:
        names = set(self.table_names())
        return {IMAGES, FACES, PEOPLE, META}.issubset(names)

    # -- metadata --------------------------------------------------------
    def read_meta(self) -> Optional[LibraryMeta]:
        if META not in self.table_names():
            return None
        rows = self._db.open_table(META).to_arrow().to_pylist()
        kv = {r["key"]: r["value"] for r in rows}
        if "image_dim" not in kv:
            return None
        return LibraryMeta(
            schema_version=int(kv.get("schema_version", 1)),
            image_dim=int(kv["image_dim"]),
            face_dim=int(kv.get("face_dim", 512)),
            embed_backend=kv.get("embed_backend", ""),
            embed_model=kv.get("embed_model", ""),
            face_backend=kv.get("face_backend", ""),
            face_model=kv.get("face_model", ""),
        )

    def write_meta(self, meta: LibraryMeta) -> None:
        with self._lock:
            if META in self.table_names():
                self._db.drop_table(META)
            tbl = self._db.create_table(META, schema=META_SCHEMA)
            tbl.add(pa.Table.from_pylist(meta.as_rows(), schema=META_SCHEMA))

    # -- creation --------------------------------------------------------
    def create(self, meta: LibraryMeta, drop_existing: bool = False) -> None:
        """Create the tables. With drop_existing, wipe anything already there."""
        with self._lock:
            if drop_existing:
                for name in (IMAGES, FACES, PEOPLE, META):
                    if name in self.table_names():
                        self._db.drop_table(name)

            existing = set(self.table_names())
            if IMAGES not in existing:
                self._db.create_table(IMAGES, schema=images_schema(meta.image_dim))
            if FACES not in existing:
                self._db.create_table(FACES, schema=faces_schema(meta.face_dim))
            if PEOPLE not in existing:
                self._db.create_table(PEOPLE, schema=people_schema(meta.face_dim))
            self.write_meta(meta)

    def verify_compatible(self, meta: LibraryMeta) -> LibraryMeta:
        """Check on-disk models against the configured ones.

        Dimension changes are fatal — the stored vectors are meaningless
        under a different model. A same-dimension model swap is also fatal
        for the same reason, so we compare model identity too.
        """
        stored = self.read_meta()
        if stored is None:
            raise SchemaMismatch("Library has no metadata; run a full index first.")
        if stored.schema_version != SCHEMA_VERSION:
            raise SchemaMismatch(
                f"Library uses schema v{stored.schema_version}, this build needs "
                f"v{SCHEMA_VERSION}. Re-index with --rebuild."
            )
        if stored.image_dim != meta.image_dim or stored.embed_model != meta.embed_model:
            raise SchemaMismatch(
                f"Library was indexed with {stored.embed_backend}:{stored.embed_model} "
                f"({stored.image_dim}d) but the configured model is "
                f"{meta.embed_backend}:{meta.embed_model} ({meta.image_dim}d). "
                "Re-index with --rebuild or set PHOTO_EMBED_MODEL back."
            )
        if stored.face_dim != meta.face_dim or stored.face_model != meta.face_model:
            raise SchemaMismatch(
                f"Library face embeddings came from {stored.face_backend}:"
                f"{stored.face_model} ({stored.face_dim}d) but the configured face "
                f"model is {meta.face_backend}:{meta.face_model} ({meta.face_dim}d). "
                "Re-index with --rebuild."
            )
        return stored

    # -- ids -------------------------------------------------------------
    def next_id(self, table_name: str, column: str) -> int:
        """Smallest unused id. Cheap: reads one small column."""
        import pyarrow.compute as pc

        tbl = self._db.open_table(table_name)
        if tbl.count_rows(None) == 0:
            return 0
        arr = tbl.to_lance().to_table(columns=[column])[column]
        return int(pc.max(arr).as_py()) + 1

    # -- indexes ---------------------------------------------------------
    def build_indexes(self, min_rows: int, force: bool = False) -> Dict[str, str]:
        """Create ANN + scalar indexes, sized to the data.

        Small libraries skip the ANN index entirely: an exhaustive scan over
        a few thousand vectors is faster than an IVF probe, and IVF_PQ
        training fails outright when there are fewer rows than centroids.
        """
        report: Dict[str, str] = {}
        with self._lock:
            report[IMAGES] = self._build_vector_index(
                IMAGES, "vector", min_rows, force)
            report[FACES] = self._build_vector_index(
                FACES, "vector", min_rows, force)

            # Scalar indexes make the hot filters (person membership,
            # per-image face lookup, path dedupe) index scans, not table scans.
            for table_name, column, kind in (
                (FACES, "person_id", "BTREE"),
                (FACES, "image_id", "BTREE"),
                (IMAGES, "image_id", "BTREE"),
                (IMAGES, "path", "BTREE"),
            ):
                try:
                    self._db.open_table(table_name).create_scalar_index(
                        column, replace=True, index_type=kind)
                except Exception as exc:  # pragma: no cover - version dependent
                    logger.debug("Scalar index %s.%s skipped: %s", table_name, column, exc)
        return report

    def _build_vector_index(self, table_name: str, column: str,
                            min_rows: int, force: bool) -> str:
        tbl = self._db.open_table(table_name)
        rows = tbl.count_rows(None)
        if rows < min_rows and not force:
            return f"skipped ({rows} rows < {min_rows}; brute force is faster)"

        partitions, sub_vectors = ivf_pq_params(rows, self._vector_dim(tbl, column))
        try:
            tbl.create_index(
                metric="cosine",
                num_partitions=partitions,
                num_sub_vectors=sub_vectors,
                vector_column_name=column,
                replace=True,
            )
            return f"IVF_PQ(partitions={partitions}, sub_vectors={sub_vectors})"
        except Exception as exc:
            logger.warning("Vector index on %s failed, using brute force: %s",
                           table_name, exc)
            return f"failed: {exc}"

    @staticmethod
    def _vector_dim(tbl, column: str) -> int:
        field = tbl.schema.field(column)
        return field.type.list_size

    # -- maintenance -----------------------------------------------------
    def compact(self) -> None:
        """Merge small fragments and prune old versions.

        Incremental indexing appends many small fragments; without this a
        library that has been updated a few hundred times reads far more
        files than it needs to.
        """
        for name in (IMAGES, FACES, PEOPLE, OCR):
            if name == OCR and not self.has_ocr():
                continue
            try:
                ds = self._db.open_table(name).to_lance()
                ds.optimize.compact_files()
                ds.cleanup_old_versions()
            except Exception as exc:  # pragma: no cover
                logger.warning("Compaction of %s failed: %s", name, exc)


def ivf_pq_params(rows: int, dim: int) -> tuple[int, int]:
    """Pick IVF_PQ parameters that suit the library size.

    Rule of thumb from the ANN literature: ~sqrt(n) partitions, so each probe
    touches ~sqrt(n) vectors. Sub-vectors must divide the dimension evenly.
    """
    import math

    partitions = max(1, min(4096, int(math.sqrt(max(rows, 1)))))
    # Target 8 dimensions per sub-vector, then snap to a divisor of `dim`.
    target = max(1, dim // 8)
    sub_vectors = next(
        (c for c in range(target, 0, -1) if dim % c == 0),
        1,
    )
    return partitions, sub_vectors


def now_ms() -> datetime:
    """Timezone-naive UTC timestamp, matching the Arrow timestamp('ms') columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def rows_to_arrow(rows: Iterable[Dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build an Arrow table under an explicit schema.

    Going through the schema rather than letting pyarrow infer types is what
    keeps nullable lat/lon and empty ``people_ids`` lists from blowing up
    type inference on the first batch that happens to be all-null.
    """
    return pa.Table.from_pylist(list(rows), schema=schema)
