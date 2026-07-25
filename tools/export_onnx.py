"""Export an image/text embedding model to ONNX for the packaged app.

Run this once, on a machine that has PyTorch, to produce a self-contained
model directory the packaged application can load with nothing but
onnxruntime. That is the difference between a ~3 GB installer that fights
PyInstaller and a ~500 MB one that doesn't.

    python tools/export_onnx.py --model google/siglip2-base-patch16-224 \
                                --out models/siglip2-base

The output directory contains:

    vision.onnx        image tower, pixel_values -> embedding
    text.onnx          text tower, input_ids -> embedding
    tokenizer.json     HF `tokenizers` file, loadable without transformers
    preprocess.json    everything needed to reproduce preprocessing exactly

Preprocessing parameters are *read off the real processor* and written to
preprocess.json rather than hardcoded, so the runtime cannot drift from what
the model was trained with, and so the same exporter works for CLIP and
SigLIP alike.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("export_onnx")

# Opset 17 has native LayerNormalization, which keeps the exported graph much
# smaller and faster than the decomposed form older opsets produce.
OPSET = 17


def _image_config(processor) -> dict:
    ip = getattr(processor, "image_processor", processor)
    size = getattr(ip, "size", {}) or {}
    crop = getattr(ip, "crop_size", {}) or {}

    height = size.get("height") or size.get("shortest_edge") or 224
    width = size.get("width") or size.get("shortest_edge") or 224

    return {
        "height": int(height),
        "width": int(width),
        "do_center_crop": bool(getattr(ip, "do_center_crop", False)),
        "crop_height": int(crop.get("height", height)),
        "crop_width": int(crop.get("width", width)),
        "resample": int(getattr(ip, "resample", 3)),      # PIL filter id
        "rescale_factor": float(getattr(ip, "rescale_factor", 1 / 255)),
        "image_mean": [float(v) for v in getattr(ip, "image_mean", [0.5] * 3)],
        "image_std": [float(v) for v in getattr(ip, "image_std", [0.5] * 3)],
    }


PROBE_TEXTS = [
    "A Photo, of the BEACH at sunset!",
    "two children playing in the snow",
    "birthday cake with candles",
]


def _text_config(processor, backend: str) -> dict:
    tokenizer = getattr(processor, "tokenizer", processor)
    # SigLIP's text tower is trained at a fixed 64-token context with no
    # attention mask; CLIP uses 77 and does use one.
    max_length = 64 if backend == "siglip" else 77
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max < 1024:
        max_length = model_max

    return {
        "max_length": int(max_length),
        "pad_token_id": int(getattr(tokenizer, "pad_token_id", 0) or 0),
        "pad_token": str(getattr(tokenizer, "pad_token", "<pad>") or "<pad>"),
        "canonicalize": _detects_canonicalization(processor, tokenizer, max_length),
    }


def _detects_canonicalization(processor, tokenizer, max_length: int) -> bool:
    """Does the processor lowercase and strip punctuation before tokenising?

    SigLIP does; CLIP does not. Rather than encode that as a guess keyed off
    the model name, ask the real processor: tokenise a probe string with
    capitals and punctuation both ways and see which the processor agrees
    with. A wrong answer here degrades every text query subtly and silently.
    """
    from photolib.embeddings.onnx_vision import canonicalize_text

    probe = PROBE_TEXTS[0]
    try:
        reference = processor(text=[probe], padding="max_length",
                              max_length=max_length, truncation=True,
                              return_tensors="np")["input_ids"][0].tolist()
    except Exception as exc:
        logger.warning("Could not probe the processor (%s); assuming no "
                       "canonicalisation", exc)
        return False

    def ids_for(text: str):
        return tokenizer(text, padding="max_length", max_length=max_length,
                         truncation=True, return_tensors="np")["input_ids"][0].tolist()

    if ids_for(probe) == reference:
        return False
    if ids_for(canonicalize_text(probe)) == reference:
        logger.info("Processor canonicalises text (lowercase, no punctuation)")
        return True

    logger.warning("Neither raw nor canonicalised tokenisation reproduced the "
                   "processor's output — the golden self-check will catch this.")
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--out", default="models/siglip2-base")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--quantize", action="store_true",
                        help="Also emit int8-quantised graphs (about 4x smaller, "
                             "slight quality loss — verify before shipping)")
    parser.add_argument("--tolerance", type=float, default=2e-2,
                        help="Max allowed cosine drift from the PyTorch model")
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        logger.error("This tool needs torch and transformers installed: %s", exc)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    backend = "siglip" if "siglip" in args.model.lower() else "clip"

    image_cfg = _image_config(processor)
    text_cfg = _text_config(processor, backend)

    class VisionTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m.get_image_features(pixel_values=pixel_values)

    class TextTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            return self.m.get_text_features(input_ids=input_ids)

    height, width = image_cfg["crop_height"], image_cfg["crop_width"]
    dummy_pixels = torch.randn(2, 3, height, width)
    dummy_ids = torch.randint(0, 1000, (2, text_cfg["max_length"]), dtype=torch.int64)

    logger.info("Exporting vision tower (%dx%d)", height, width)
    torch.onnx.export(
        VisionTower(model), (dummy_pixels,), str(out / "vision.onnx"),
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True)

    logger.info("Exporting text tower (%d tokens)", text_cfg["max_length"])
    torch.onnx.export(
        TextTower(model), (dummy_ids,), str(out / "text.onnx"),
        input_names=["input_ids"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True)

    # The runtime tokenises with the `tokenizers` library alone — no
    # transformers, no sentencepiece, no Python-side vocab handling.
    tokenizer = getattr(processor, "tokenizer", processor)
    saved = Path(tokenizer.save_pretrained(str(out / "_tok"))[0]).parent
    tokenizer_json = saved / "tokenizer.json"
    if not tokenizer_json.exists():
        try:
            tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
        except Exception as exc:
            logger.error("Could not export a fast tokenizer.json: %s", exc)
            return 3
    else:
        shutil.copy(tokenizer_json, out / "tokenizer.json")
    shutil.rmtree(out / "_tok", ignore_errors=True)

    with torch.inference_mode():
        reference_image = model.get_image_features(pixel_values=dummy_pixels).numpy()
        reference_text = model.get_text_features(input_ids=dummy_ids).numpy()

    dim = int(reference_image.shape[-1])
    (out / "preprocess.json").write_text(json.dumps({
        "model": args.model,
        "backend": backend,
        "dim": dim,
        "opset": args.opset,
        "image": image_cfg,
        "text": text_cfg,
    }, indent=2))

    _write_golden(out, model, processor, text_cfg, height, width)

    if args.quantize:
        _quantize(out)

    ok = _verify(out, reference_image, reference_text, dummy_pixels.numpy(),
                 dummy_ids.numpy(), args.tolerance)
    ok = _verify_runtime(out) and ok

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    logger.info("Wrote %s (%.0f MB, %d-dim)", out, total / 1e6, dim)
    return 0 if ok else 1


def _write_golden(out: Path, model, processor, text_cfg: dict,
                  height: int, width: int) -> None:
    """Record reference tokenisations and embeddings from the real model.

    The runtime reimplements preprocessing in NumPy. These let it prove it
    still agrees with 🤗 — a silent preprocessing drift would otherwise show
    up months later as "search got worse".
    """
    import numpy as np
    import torch

    ids = processor(text=PROBE_TEXTS, padding="max_length",
                    max_length=text_cfg["max_length"], truncation=True,
                    return_tensors="np")["input_ids"].astype(np.int64)

    rng = np.random.default_rng(0)
    pixels = rng.standard_normal((2, 3, height, width)).astype(np.float32)

    with torch.inference_mode():
        text_embeddings = model.get_text_features(
            input_ids=torch.from_numpy(ids)).numpy()
        image_embeddings = model.get_image_features(
            pixel_values=torch.from_numpy(pixels)).numpy()

    (out / "golden.json").write_text(json.dumps({
        "texts": PROBE_TEXTS,
        "token_ids": ids.tolist(),
        "text_embeddings": text_embeddings.tolist(),
        "pixels": pixels.tolist(),
        "image_embeddings": image_embeddings.tolist(),
    }))
    logger.info("Wrote golden.json (%d probe texts)", len(PROBE_TEXTS))


def _verify_runtime(out: Path) -> bool:
    """Load the export through the actual runtime class and self-check it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from photolib.embeddings.onnx_vision import OnnxVisionEmbedder

    report = OnnxVisionEmbedder(out).self_check()
    logger.info("Runtime self-check: %s", json.dumps(report, indent=2))
    if not report.get("checked"):
        return False
    if not report.get("ok"):
        logger.error("The NumPy runtime does not reproduce the 🤗 pipeline.")
        return False
    return True


def _quantize(out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name in ("vision", "text"):
        source = out / f"{name}.onnx"
        target = out / f"{name}.int8.onnx"
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
        logger.info("Quantised %s: %.0f MB -> %.0f MB", name,
                    source.stat().st_size / 1e6, target.stat().st_size / 1e6)


def _verify(out: Path, reference_image, reference_text, pixels, ids,
            tolerance: float) -> bool:
    """Confirm the exported graphs match PyTorch, on normalised vectors.

    Cosine similarity is what actually matters — search ranks by it — so
    that, not raw tensor equality, is the thing to check.
    """
    import numpy as np
    import onnxruntime as ort

    def normalise(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

    ok = True
    for name, feed, reference in (
        ("vision", {"pixel_values": pixels}, reference_image),
        ("text", {"input_ids": ids}, reference_text),
    ):
        session = ort.InferenceSession(str(out / f"{name}.onnx"),
                                       providers=["CPUExecutionProvider"])
        actual = session.run(None, feed)[0]
        similarity = (normalise(actual) * normalise(reference)).sum(-1)
        worst = float(similarity.min())
        logger.info("%s parity: cosine %.6f", name, worst)
        if 1.0 - worst > tolerance:
            logger.error("%s drifted beyond tolerance (%.6f)", name, worst)
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(main())
