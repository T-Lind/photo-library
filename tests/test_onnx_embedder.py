"""The ONNX embedder used by the packaged desktop build.

The real SigLIP 2 export needs PyTorch and the model hub, so parity against
🤗 is checked in CI by ``tools/export_onnx.py``. Everything around it —
preprocessing arithmetic, tokenisation, batching, normalisation, session
handling — is checked here against a synthetic two-tower model with known
weights, so it is genuinely executed rather than mocked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from photolib.embeddings.onnx_vision import (ModelDirError, OnnxVisionEmbedder,
                                             canonicalize_text)
from tests.onnx_fixture import DIM, MAX_LENGTH, build_model_dir

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    return build_model_dir(tmp_path_factory.mktemp("onnx-model"))


@pytest.fixture(scope="module")
def embedder(model_dir) -> OnnxVisionEmbedder:
    return OnnxVisionEmbedder(model_dir)


def test_reports_its_dimension_without_loading_sessions(model_dir):
    # Constructing must not pay for onnxruntime session creation — the API
    # server builds an embedder at startup and may never run inference.
    fresh = OnnxVisionEmbedder(model_dir)
    assert fresh.dim == DIM
    assert fresh._vision is None and fresh._text is None


def test_missing_model_directory_gives_an_actionable_error(tmp_path):
    with pytest.raises(ModelDirError, match="export_onnx"):
        OnnxVisionEmbedder(tmp_path / "nope")


def test_text_embeddings_are_normalised_and_ordered(embedder):
    vectors = embedder.embed_texts(["a photo of the beach", "snow", "cake"])

    assert vectors.shape == (3, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    # Same input, same output — no hidden state between calls.
    assert np.allclose(vectors[1], embedder.embed_texts(["snow"])[0], atol=1e-6)


def test_different_text_gives_different_embeddings(embedder):
    beach, snow = embedder.embed_texts(["beach sunset", "snow mountain"])
    assert float(beach @ snow) < 0.999


def test_image_embeddings_are_normalised(embedder, tmp_path):
    from tests.conftest import make_photo

    paths = [make_photo(tmp_path / f"p{i}.jpg", size=(120, 90)) for i in range(3)]
    vectors = embedder.embed_images([str(p) for p in paths])

    assert vectors.shape == (3, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_input_returns_an_empty_array(embedder):
    assert embedder.embed_texts([]).shape == (0, DIM)
    assert embedder.embed_images([]).shape == (0, DIM)


def test_batching_matches_one_at_a_time(model_dir, tmp_path):
    """A batch of 5 through a batch size of 2 must not reorder or corrupt."""
    from tests.conftest import make_photo

    paths = [str(make_photo(tmp_path / f"b{i}.jpg", size=(100, 80), tint=(i * 40, 90, 200)))
             for i in range(5)]

    batched = OnnxVisionEmbedder(model_dir, batch_size=2).embed_images(paths)
    singly = np.concatenate(
        [OnnxVisionEmbedder(model_dir).embed_images([p]) for p in paths])

    assert np.allclose(batched, singly, atol=1e-5)


def test_arrays_and_paths_give_the_same_embedding(model_dir, tmp_path):
    """The indexer decodes once and passes arrays; both routes must agree."""
    from photolib.embeddings.base import ImageInput
    from photolib.imageio import load_rgb_array
    from tests.conftest import make_photo

    path = str(make_photo(tmp_path / "same.jpg", size=(140, 110)))
    embedder = OnnxVisionEmbedder(model_dir)

    from_path = embedder.embed_images([path])
    from_array = embedder.embed_images(
        [ImageInput(path=path, array=load_rgb_array(path))])

    assert np.allclose(from_path, from_array, atol=1e-5)


# -- preprocessing arithmetic ---------------------------------------------

def test_preprocessing_matches_an_independent_reference(embedder, tmp_path):
    """Reproduce the 🤗 normalisation by hand and compare."""
    from PIL import Image

    from photolib.imageio import open_image
    from tests.conftest import make_photo

    path = make_photo(tmp_path / "pre.jpg", size=(64, 48), tint=(200, 100, 50))
    cfg = embedder.image_cfg

    with open_image(path) as img:
        actual = embedder.preprocess_image(img)

        expected_img = img.resize((cfg["width"], cfg["height"]),
                                  Image.Resampling.BICUBIC)
        expected = np.asarray(expected_img, dtype=np.float32) / 255.0
        expected = (expected - 0.5) / 0.5
        expected = np.transpose(expected, (2, 0, 1))

    assert actual.shape == (3, cfg["height"], cfg["width"])
    assert np.allclose(actual, expected, atol=1e-5)


def test_center_crop_config_is_honoured(tmp_path):
    """CLIP-style exports scale the short edge then crop, not squash."""
    from photolib.imageio import open_image
    from tests.conftest import make_photo

    root = build_model_dir(tmp_path / "crop-model", center_crop=True,
                           height=16, width=16, with_golden=False)
    embedder = OnnxVisionEmbedder(root)

    path = make_photo(tmp_path / "wide.jpg", size=(200, 50))
    with open_image(path) as img:
        tensor = embedder.preprocess_image(img)

    assert tensor.shape == (3, 16, 16)


def test_greyscale_and_rgba_sources_are_converted(embedder, tmp_path):
    from PIL import Image

    for mode, name in (("L", "grey.png"), ("RGBA", "alpha.png")):
        path = tmp_path / name
        Image.new(mode, (40, 30), color=128).save(path)
        vectors = embedder.embed_images([str(path)])
        assert vectors.shape == (1, DIM)
        assert np.isfinite(vectors).all()


# -- tokenisation ----------------------------------------------------------

def test_tokenisation_pads_to_the_exported_length(embedder):
    ids = embedder.preprocess_texts(["beach", "a photo of the beach at sunset"])

    # The text tower is exported at a fixed sequence length; anything else
    # is a shape error at inference time.
    assert ids.shape == (2, MAX_LENGTH)
    assert ids.dtype == np.int64
    assert ids[0][-1] == 0  # padded with pad_token_id


def test_long_text_is_truncated_not_rejected(embedder):
    ids = embedder.preprocess_texts([" ".join(["beach"] * 200)])
    assert ids.shape == (1, MAX_LENGTH)


def test_canonicalisation_is_applied_only_when_configured(tmp_path):
    plain = OnnxVisionEmbedder(
        build_model_dir(tmp_path / "plain", canonicalize=False, with_golden=False))
    canon = OnnxVisionEmbedder(
        build_model_dir(tmp_path / "canon", canonicalize=True, with_golden=False))

    messy = "A Photo, of the BEACH!"
    assert not np.array_equal(plain.preprocess_texts([messy]),
                              canon.preprocess_texts([messy]))
    # With canonicalisation on, case and punctuation stop mattering.
    assert np.array_equal(canon.preprocess_texts([messy]),
                          canon.preprocess_texts(["a photo of the beach"]))


@pytest.mark.parametrize("raw,expected", [
    ("A Photo, of the BEACH!", "a photo of the beach"),
    ("two   children  playing", "two children playing"),
    ("Birthday-Cake (with candles)", "birthdaycake with candles"),
    ("", ""),
])
def test_canonicalize_text(raw, expected):
    assert canonicalize_text(raw) == expected


# -- the golden self-check -------------------------------------------------

def test_self_check_passes_against_the_exporter_reference(embedder):
    report = embedder.self_check()

    assert report["checked"] is True
    assert report["tokenizer_matches"] is True
    assert report["text_matches"] is True
    assert report["image_matches"] is True
    assert report["ok"] is True


def test_self_check_catches_a_preprocessing_regression(model_dir, monkeypatch):
    """The whole point of golden.json: silent drift becomes a hard failure."""
    embedder = OnnxVisionEmbedder(model_dir)

    # Simulate someone "fixing" tokenisation by turning canonicalisation on.
    monkeypatch.setitem(embedder.text_cfg, "canonicalize", True)

    report = embedder.self_check()
    assert report["tokenizer_matches"] is False
    assert report["ok"] is False


def test_self_check_reports_cleanly_when_there_is_no_golden_file(tmp_path):
    root = build_model_dir(tmp_path / "bare", with_golden=False)
    report = OnnxVisionEmbedder(root).self_check()

    assert report["checked"] is False
    assert "golden" in report["reason"]


def test_backend_is_selected_from_configuration(model_dir, monkeypatch):
    from photolib.config import get_settings, reset_settings_cache
    from photolib.embeddings import build_embedder

    monkeypatch.setenv("PHOTO_EMBED_BACKEND", "onnx")
    monkeypatch.setenv("PHOTO_ONNX_MODEL_DIR", str(model_dir))
    reset_settings_cache()
    try:
        embedder = build_embedder(get_settings())
        assert isinstance(embedder, OnnxVisionEmbedder)
        assert embedder.dim == DIM
    finally:
        reset_settings_cache()
