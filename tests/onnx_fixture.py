"""Builds a tiny but real two-tower ONNX model directory.

The genuine SigLIP 2 export can only be produced on a machine with PyTorch
and network access to the model hub, so it is verified in CI. What *can* be
verified anywhere is the runtime path around it: session loading, NumPy
preprocessing, tokenisation, batching, normalisation, and the golden
self-check. This builds a model with the same interface and known weights so
all of that is exercised for real rather than mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

VOCAB = ["<pad>", "a", "photo", "of", "the", "beach", "sunset", "snow",
         "children", "playing", "birthday", "cake", "candles", "two", "at",
         "in", "with", "dog", "mountain", "[UNK]"]
DIM = 8
MAX_LENGTH = 12


def _tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {token: i for i, token in enumerate(VOCAB)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


def _vision_model(path: Path, height: int, width: int, weight: np.ndarray) -> None:
    """pixel_values (N,3,H,W) -> mean over H,W -> MatMul -> (N,DIM)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Opset 17's ReduceMean takes `axes` as an attribute; it only moved to
    # being an input in opset 18.
    nodes = [
        helper.make_node("ReduceMean", ["pixel_values"], ["pooled"],
                         axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["pooled", "vision_w"], ["embedding"]),
    ]
    graph = helper.make_graph(
        nodes, "vision",
        inputs=[helper.make_tensor_value_info(
            "pixel_values", TensorProto.FLOAT, ["batch", 3, height, width])],
        outputs=[helper.make_tensor_value_info(
            "embedding", TensorProto.FLOAT, ["batch", DIM])],
        initializer=[numpy_helper.from_array(weight, "vision_w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


def _text_model(path: Path, table: np.ndarray) -> None:
    """input_ids (N,L) -> embedding table lookup -> mean over L -> (N,DIM)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["looked_up"], axis=0),
        helper.make_node("ReduceMean", ["looked_up"], ["embedding"],
                         axes=[1], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "text",
        inputs=[helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch", MAX_LENGTH])],
        outputs=[helper.make_tensor_value_info(
            "embedding", TensorProto.FLOAT, ["batch", DIM])],
        initializer=[numpy_helper.from_array(table, "table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


def build_model_dir(root: Path, canonicalize: bool = False,
                    center_crop: bool = False, height: int = 16,
                    width: int = 16, with_golden: bool = True) -> Path:
    """Write a complete exported-model directory to ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    vision_w = rng.standard_normal((3, DIM)).astype(np.float32)
    table = rng.standard_normal((len(VOCAB), DIM)).astype(np.float32)

    _vision_model(root / "vision.onnx", height, width, vision_w)
    _text_model(root / "text.onnx", table)
    _tokenizer(root / "tokenizer.json")

    (root / "preprocess.json").write_text(json.dumps({
        "model": "synthetic/two-tower",
        "backend": "siglip",
        "dim": DIM,
        "opset": 17,
        "image": {
            "height": height,
            "width": width,
            "do_center_crop": center_crop,
            "crop_height": height,
            "crop_width": width,
            "resample": 3,
            "rescale_factor": 1 / 255,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
        "text": {
            "max_length": MAX_LENGTH,
            "pad_token_id": 0,
            "pad_token": "<pad>",
            "canonicalize": canonicalize,
        },
    }, indent=2))

    if with_golden:
        _write_golden(root, table, vision_w, height, width, canonicalize)
    return root


def _write_golden(root: Path, table: np.ndarray, vision_w: np.ndarray,
                  height: int, width: int, canonicalize: bool) -> None:
    """Compute reference outputs independently of the runtime under test."""
    from tokenizers import Tokenizer

    from photolib.embeddings.onnx_vision import canonicalize_text

    texts = ["A Photo, of the BEACH at sunset!", "two children playing in snow"]
    prepared = [canonicalize_text(t) for t in texts] if canonicalize else texts

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding(length=MAX_LENGTH, pad_id=0, pad_token="<pad>")
    ids = np.asarray([e.ids for e in tokenizer.encode_batch(prepared)],
                     dtype=np.int64)

    # Reference embeddings computed in plain NumPy, not via onnxruntime.
    text_embeddings = table[ids].mean(axis=1)

    rng = np.random.default_rng(11)
    pixels = rng.standard_normal((2, 3, height, width)).astype(np.float32)
    image_embeddings = pixels.mean(axis=(2, 3)) @ vision_w

    (root / "golden.json").write_text(json.dumps({
        "texts": texts,
        "token_ids": ids.tolist(),
        "text_embeddings": text_embeddings.tolist(),
        "pixels": pixels.tolist(),
        "image_embeddings": image_embeddings.tolist(),
    }))
