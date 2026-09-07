import io
import json

import pytest
from PIL import Image

from feedbard.ingestion.html_parser import Document, Visual
from feedbard.pipeline import visual_describer
from feedbard.pipeline.visual_describer import (
    _downscale_if_needed,
    describe_visual,
    describe_visuals,
    load_visual_shard,
    save_visual_shard,
)


def _visual(vid="aaaaaaaaaa", hero=False):
    return Visual(vid=vid, src=f"s3://{vid}", fetch_url=f"https://img.example/{vid}.png", hero=hero)


def _png_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color="red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _sandboxed_shards(monkeypatch, tmp_path):
    """Every test in this file gets its own shard directory: describe_visual
    always writes a shard, so leaving this unpatched would hit the real
    VISUAL_SHARDS_DIR on disk even in tests that never mention shards."""

    def shard_path(vid):
        return tmp_path / f"{vid}.json"

    monkeypatch.setattr(visual_describer, "VISUAL_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(visual_describer, "visual_shard_path", shard_path)
    return tmp_path


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------


def test_save_and_load_roundtrip():
    save_visual_shard("aaaaaaaaaa", "essential", "una curva", language="Italian")

    shard = load_visual_shard("aaaaaaaaaa", language="Italian")
    assert shard == {"klass": "essential", "description": "una curva", "language": "Italian"}


def test_shard_in_another_language_is_a_miss():
    save_visual_shard("aaaaaaaaaa", "essential", "a curve", language="English")

    assert load_visual_shard("aaaaaaaaaa", language="Italian") is None


def test_shard_without_a_language_field_is_taken_at_face_value(_sandboxed_shards):
    # Written before the language was recorded at all.
    path = _sandboxed_shards / "aaaaaaaaaa.json"
    path.write_text(json.dumps({"klass": "essential", "description": "una curva"}))

    assert load_visual_shard("aaaaaaaaaa", language="Italian") is not None


# --------------------------------------------------------------------------
# Downscaling
# --------------------------------------------------------------------------


def test_small_image_is_left_untouched():
    original = _png_bytes(100, 50)
    out_bytes, media_type = _downscale_if_needed(original, "image/png")
    assert out_bytes == original
    assert media_type == "image/png"


def test_oversized_image_is_downscaled_and_converted_to_jpeg():
    huge = _png_bytes(visual_describer.MAX_IMAGE_DIM + 2000, 1000)
    out_bytes, media_type = _downscale_if_needed(huge, "image/png")

    assert media_type == "image/jpeg"
    with Image.open(io.BytesIO(out_bytes)) as img:
        assert max(img.size) <= visual_describer.MAX_IMAGE_DIM


# --------------------------------------------------------------------------
# describe_visual
# --------------------------------------------------------------------------


def test_describe_visual_parses_a_bare_json_reply(monkeypatch):
    monkeypatch.setattr(visual_describer, "_fetch_image", lambda vid, url: (b"bytes", "image/png"))
    monkeypatch.setattr(
        visual_describer,
        "generate_vision_response",
        lambda prompt, image_bytes, media_type: (
            '{"klass": "essential", "description": "La curva sale."}',
            0.0,
        ),
    )
    v = _visual()
    describe_visual(v)
    assert v.klass == "essential"
    assert v.description == "La curva sale."


def test_describe_visual_extracts_json_from_surrounding_prose(monkeypatch):
    # Some models wrap the JSON in commentary despite the prompt asking for
    # raw JSON; the extractor has to find the object anyway.
    monkeypatch.setattr(visual_describer, "_fetch_image", lambda vid, url: (b"bytes", "image/png"))
    monkeypatch.setattr(
        visual_describer,
        "generate_vision_response",
        lambda prompt, image_bytes, media_type: (
            'Ecco il risultato: {"klass": "illustrative", "description": "Un ritratto."} Fine.',
            0.0,
        ),
    )
    v = _visual()
    describe_visual(v)
    assert v.klass == "illustrative"
    assert v.description == "Un ritratto."


def test_describe_visual_rejects_an_unknown_klass(monkeypatch):
    monkeypatch.setattr(visual_describer, "_fetch_image", lambda vid, url: (b"bytes", "image/png"))
    monkeypatch.setattr(
        visual_describer,
        "generate_vision_response",
        lambda prompt, image_bytes, media_type: ('{"klass": "boh", "description": "x"}', 0.0),
    )
    v = _visual()
    describe_visual(v)
    assert v.klass == "decorative"


def test_describe_visual_blank_description_becomes_none(monkeypatch):
    monkeypatch.setattr(visual_describer, "_fetch_image", lambda vid, url: (b"bytes", "image/png"))
    monkeypatch.setattr(
        visual_describer,
        "generate_vision_response",
        lambda prompt, image_bytes, media_type: ('{"klass": "decorative", "description": ""}', 0.0),
    )
    v = _visual()
    describe_visual(v)
    assert v.description is None


def test_describe_visual_renders_hint_alt_and_caption_into_the_prompt(monkeypatch):
    seen = {}

    def generate_vision_response(prompt, image_bytes, media_type):
        seen["prompt"] = prompt
        return '{"klass": "essential", "description": "d"}', 0.0

    monkeypatch.setattr(visual_describer, "_fetch_image", lambda vid, url: (b"bytes", "image/png"))
    monkeypatch.setattr(visual_describer, "generate_vision_response", generate_vision_response)

    v = _visual()
    v.hint = "formula"
    v.alt = "diagramma di flusso"
    v.caption = "Figura 3"
    describe_visual(v)

    assert "formula" in seen["prompt"]
    assert "diagramma di flusso" in seen["prompt"]
    assert "Figura 3" in seen["prompt"]


# --------------------------------------------------------------------------
# describe_visuals: batch orchestration
# --------------------------------------------------------------------------


def test_a_failing_visual_falls_back_to_decorative_instead_of_aborting(monkeypatch):
    def boom(vid, url):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(visual_describer, "_fetch_image", boom)

    doc = Document(blocks=[], visuals={"a": _visual("a")}, notes={})
    describe_visuals(doc)

    assert doc.visuals["a"].klass == "decorative"
    assert doc.visuals["a"].description is None


def test_hero_image_is_never_sent_to_the_model(monkeypatch):
    monkeypatch.setattr(visual_describer, "_fetch_image", _unreachable_fetch)

    doc = Document(blocks=[], visuals={"a": _visual("a", hero=True)}, notes={})
    describe_visuals(doc)

    assert doc.visuals["a"].klass == "decorative"


def _unreachable_fetch(vid, url):
    raise AssertionError("a hero image must never reach the vision model")


def test_a_second_run_resumes_every_visual_from_shards(monkeypatch):
    calls = []

    def generate_vision_response(prompt, image_bytes, media_type):
        calls.append(prompt)
        return '{"klass": "essential", "description": "d"}', 0.0

    monkeypatch.setattr(visual_describer, "_fetch_image", lambda vid, url: (b"x", "image/png"))
    monkeypatch.setattr(visual_describer, "generate_vision_response", generate_vision_response)

    doc1 = Document(blocks=[], visuals={"a": _visual("a")}, notes={})
    describe_visuals(doc1)
    doc2 = Document(blocks=[], visuals={"a": _visual("a")}, notes={})
    describe_visuals(doc2)

    assert len(calls) == 1
    assert doc2.visuals["a"].klass == "essential"


@pytest.mark.parametrize(
    "content_type,expected",
    [("image/jpeg", "image/jpeg"), ("image/png; charset=binary", "image/png")],
)
def test_fetch_image_downloads_when_not_cached(monkeypatch, tmp_path, content_type, expected):
    monkeypatch.setattr(visual_describer, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(visual_describer, "find_figure", lambda vid: None)
    monkeypatch.setattr(visual_describer, "figure_path", lambda vid, ext: tmp_path / f"{vid}{ext}")

    payload = _png_bytes(10, 10)

    class FakeResponse:
        content = payload
        headers = {"Content-Type": content_type}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(visual_describer.requests, "get", lambda url, timeout: FakeResponse())

    image_bytes, media_type = visual_describer._fetch_image(
        "aaaaaaaaaa", "https://img.example/a.png"
    )
    assert media_type == expected
    assert len(list(tmp_path.iterdir())) == 1  # cached to disk


def test_fetch_image_uses_the_cache_and_skips_the_network(monkeypatch, tmp_path):
    cached = tmp_path / "aaaaaaaaaa.jpg"
    cached.write_bytes(b"cached-bytes")
    monkeypatch.setattr(visual_describer, "find_figure", lambda vid: cached)
    monkeypatch.setattr(
        visual_describer.requests,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network hit despite cache")),
    )

    image_bytes, media_type = visual_describer._fetch_image(
        "aaaaaaaaaa", "https://img.example/a.png"
    )
    assert image_bytes == b"cached-bytes"
    assert media_type == "image/jpeg"


def test_a_shard_written_before_the_rename_still_classifies(monkeypatch, tmp_path):
    """Shards on disk record the old Italian class names. They are mapped on
    read, not invalidated: re-deriving them would re-bill the vision model for
    every figure already described."""
    shard = tmp_path / "a.json"
    shard.write_text(
        json.dumps({"klass": "decorativo", "description": None, "language": "Italian"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(visual_describer, "visual_shard_path", lambda vid: shard)
    monkeypatch.setattr(visual_describer, "_fetch_image", _unreachable)

    doc = Document(blocks=[], visuals={"a": _visual("a")}, notes={})
    describe_visuals(doc)

    assert doc.visuals["a"].klass == visual_describer.SKIPPED_KLASS


def _unreachable(*a, **kw):
    raise AssertionError("a shard was on disk: the model must not be called")
