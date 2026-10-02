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
    return (_unsigned(a) ^ _unsigned(b)).bit_count()


def group_near_duplicates(items: Iterable[Tuple[int, int]],
                          max_distance: int = 6) -> List[List[int]]:
    """Group ``(image_id, phash)`` pairs into near-duplicate sets.

    For distances up to seven, probe four 16-bit bands within distance
    ``max_distance // 4``. At least one band must satisfy that bound, so
    candidates are complete even when every band differs. Larger radii use
    an exact Hamming BK-tree. Equal hashes are collapsed before matching;
    large buckets never silently lose matches.
    """
    items = list(items)
    if not 0 <= max_distance <= 64:
        raise ValueError("max_distance must be between 0 and 64")
    if not items:
        return []

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

    representatives: dict[int, int] = {}
    for image_id, value in items:
        value = _unsigned(value)
        if value in representatives:
            union(image_id, representatives[value])
        else:
            representatives[value] = image_id

    if max_distance == 64:
        first = items[0][0]
        for image_id, _ in items:
            union(first, image_id)
    elif max_distance <= 7:
        bands: List[dict] = [dict() for _ in range(4)]
        masks = [0] + ([1 << bit for bit in range(16)] if max_distance >= 4 else [])
        for value, image_id in representatives.items():
            candidates: set[int] = set()
            for band_index, band in enumerate(bands):
                key = (value >> (16 * band_index)) & 0xFFFF
                for mask in masks:
                    candidates.update(band.get(key ^ mask, ()))
            for other in candidates:
                if (value ^ other).bit_count() <= max_distance:
                    union(image_id, representatives[other])
            for band_index, band in enumerate(bands):
                key = (value >> (16 * band_index)) & 0xFFFF
                band.setdefault(key, []).append(value)
    elif representatives:
        # A node is (hash, children keyed by distance from that hash).
        root = None
        for value, image_id in representatives.items():
            if root is None:
                root = (value, {})
                continue
            pending = [root]
            while pending:
                other, children = pending.pop()
                distance = (value ^ other).bit_count()
                if distance <= max_distance:
                    union(image_id, representatives[other])
                pending.extend(child for edge, child in children.items()
                               if distance - max_distance <= edge <= distance + max_distance)
            node = root
            while True:
                distance = (value ^ node[0]).bit_count()
                child = node[1].get(distance)
                if child is None:
                    node[1][distance] = (value, {})
                    break
                node = child

    groups: dict = {}
    for image_id, _ in items:
        groups.setdefault(find(image_id), []).append(image_id)
    return [sorted(g) for g in groups.values() if len(g) > 1]
