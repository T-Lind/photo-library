"""Versioned technical quality signals, computed locally at a fixed scale."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
from PIL import Image

from .faces.base import laplacian_variance

VERSION = 1
TABLE = "image_quality"
SCHEMA = pa.schema([
    ("image_id", pa.int64()), ("version", pa.int32()),
    ("sharpness", pa.float32()), ("exposure", pa.float32()),
    ("resolution", pa.float32()), ("score", pa.float32()),
])


def measure(array, width, height):
    image = Image.fromarray(array)
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    gray = np.asarray(image, dtype=np.float32).mean(axis=2)
    sharpness = float(np.clip(np.log1p(laplacian_variance(gray)) / np.log1p(500), 0, 1))
    # A soft ranking signal, never an automatic rejection of silhouettes.
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    exposure = float(np.clip(1 - clipped, 0, 1))
    resolution = float(np.clip(np.sqrt(max(0, width * height) / 12_000_000), 0, 1))
    return dict(version=VERSION, sharpness=sharpness, exposure=exposure,
                resolution=resolution,
                score=0.4 * sharpness + 0.3 * exposure + 0.3 * resolution)


def save(library, rows):
    if not rows:
        return
    if TABLE not in library.table_names():
        library.db.create_table(TABLE, schema=SCHEMA)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    library.table(TABLE).merge_insert("image_id").when_matched_update_all().when_not_matched_insert_all().execute(table)


def get(library, ids=None):
    if TABLE not in library.table_names():
        return {}
    dataset = library.table(TABLE).to_lance()
    if ids is None:
        table = dataset.to_table(filter=f"version = {VERSION}")
    else:
        ids = list(ids)
        if not ids:
            return {}
        table = dataset.to_table(filter=(
            f"version = {VERSION} AND image_id IN ({','.join(str(int(i)) for i in ids)})"))
    return {int(r["image_id"]): r for r in table.to_pylist()}
