"""Content and perceptual hashing for duplicate detection.

A family library that has been copied between phones, laptops, and backup
drives is full of duplicates. Two cheap hashes catch almost all of them:

* ``content_hash`` — SHA-256 of the file bytes. Catches byte-identical copies
  even when the filename and timestamps differ.
* ``phash`` — a 64-bit DCT perceptual hash. Catches re-encodes, resizes, and
  "shared via WhatsApp" versions of the same photo, and groups burst shots.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from .imageio import open_image

_CHUNK = 1 << 20  # 1 MiB
_MASK64 = (1 << 64) - 1


def _unsigned(value: int) -> int:
    """Reinterpret a stored (possibly negative) int64 hash as unsigned."""
    return value & _MASK64


def _to_int64(value: int) -> int:
    """Fold an unsigned 64-bit hash into the signed range Arrow stores."""
    value &= _MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def content_hash(path: os.PathLike | str) -> str:
    """SHA-256 of the file, streamed so a 200MB RAW never lands in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def _dct2(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II via matrix multiplication.

    Implemented here rather than pulled from scipy: it is a 32x32 matmul on a
    tiny array, and it keeps scipy out of the dependency tree.
    """
    n = block.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis[0, :] = basis[0, :] * (1.0 / np.sqrt(2))
    return basis @ block @ basis.T


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit perceptual hash (DCT of a 32x32 greyscale reduction)."""
    size = hash_size * highfreq_factor
    small = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    coeffs = _dct2(pixels)[:hash_size, :hash_size]
    # Skip the DC term when computing the threshold; it encodes overall
    # brightness, which we explicitly do not want to be sensitive to.
    median = np.median(coeffs.flatten()[1:])
    bits = (coeffs > median).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    # Stored as int64, so fold the top bit into the sign rather than overflow.
    return _to_int64(value)


def phash_file(path: os.PathLike | str) -> int:
    with open_image(path, target=(64, 64)) as img:
        return phash(img)


def hamming(a: int, b: int) -> int:
    return bin(_unsigned(a) ^ _unsigned(b)).count("1")


def group_near_duplicates(items: Iterable[Tuple[int, int]],
                          max_distance: int = 6) -> List[List[int]]:
    """Group ``(image_id, phash)`` pairs into near-duplicate sets.

    Uses multi-index bucketing (split the 64-bit hash into 4 x 16-bit bands):
    two hashes within `max_distance <= 6` bits must agree exactly on at least
    one band by the pigeonhole principle, so only same-band pairs need to be
    compared. That turns an O(n^2) sweep over 200k photos into a couple of
    dictionary passes.
    """
    items = list(items)
    if not items:
        return []

    bands: List[dict] = [dict() for _ in range(4)]
    for image_id, h in items:
        uh = _unsigned(h)
        for b in range(4):
            key = (uh >> (16 * b)) & 0xFFFF
            bands[b].setdefault(key, []).append(image_id)

    parent = {image_id: image_id for image_id, _ in items}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = dict(items)
    for band in bands:
        for bucket in band.values():
            if len(bucket) < 2 or len(bucket) > 512:
                # Huge buckets are degenerate (e.g. thousands of black
                # frames); comparing them all pairwise is not worth it.
                continue
            for i, a in enumerate(bucket):
                for b in bucket[i + 1:]:
                    if hamming(hashes[a], hashes[b]) <= max_distance:
                        union(a, b)

    groups: dict = {}
    for image_id, _ in items:
        groups.setdefault(find(image_id), []).append(image_id)
    return [sorted(g) for g in groups.values() if len(g) > 1]
