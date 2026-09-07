"""Vision pass: fills in Visual.klass and Visual.description by actually
looking at the image.

Runs after extract() and before `narration.attach_descriptions`, which is
what puts the description back into the block stream so it reaches the
episode. The description is written directly in the narration language, so
nothing translates it afterwards.

A visual that fails to fetch or to classify is treated as decorative rather
than aborting the item: losing one figure's narration is better than losing
the article it belongs to.
"""

from __future__ import annotations

import io
import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from jinja2 import Template
from PIL import Image

from feedbard.ingestion.html_parser import Document, Visual
from feedbard.language import TARGET_LANGUAGE
from feedbard.llm_client import generate_vision_response
from feedbard.logger import logger
from feedbard.paths import (
    ASSETS_DIR,
    FIGURES_DIR,
    VISUAL_SHARDS_DIR,
    figure_path,
    find_figure,
    visual_shard_path,
)

VISUAL_PROMPT_PATH = ASSETS_DIR / "visual_prompt.txt"

# Internal taxonomy, not display text: nothing here is ever narrated, so these
# are code identifiers and stay in the language the code is written in. They
# used to be Italian, which read as a narration-language choice the pipeline
# was not entitled to make -- a Japanese episode was still asking the vision
# model to answer `decorativo`.
#
# Only `decorative` changes what happens to the visual: it is the one class
# that never reaches the episode.
KLASSES = ("decorative", "illustrative", "essential")
SKIPPED_KLASS = "decorative"

# Shards written before the rename. Mapped on read rather than invalidated:
# the class is the same judgement under a different name, and re-deriving it
# would re-bill the vision model for every figure already on disk.
LEGACY_KLASSES = {
    "decorativo": "decorative",
    "illustrativo": "illustrative",
    "essenziale": "essential",
}


def normalize_klass(klass: str | None) -> str:
    """Any recorded or returned class -> one of KLASSES, defaulting to skip."""
    klass = LEGACY_KLASSES.get(klass or "", klass or "")
    return klass if klass in KLASSES else SKIPPED_KLASS


_JSON_RE = re.compile(r"\{.*\}", re.S)

# vid is a content hash of the image's canonical source, stable across runs
# and articles, so both the figure and its shard are keyed on it rather than
# on title: interrupting mid-article and restarting skips every
# already-described visual instead of re-billing the vision model for it, and
# a figure reused across two articles is fetched once.


def load_visual_shard(vid: str, language: str = TARGET_LANGUAGE) -> dict | None:
    """A shard describing the image in another language is a miss: it would
    put a foreign-language paragraph in the middle of the episode. Shards
    written before the language was recorded are taken at face value."""
    path = visual_shard_path(vid)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("language", language) != language:
        return None
    return data


def save_visual_shard(
    vid: str, klass: str, description: str | None, language: str = TARGET_LANGUAGE
) -> None:
    VISUAL_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    visual_shard_path(vid).write_text(
        json.dumps({"klass": klass, "description": description, "language": language}),
        encoding="utf-8",
    )


# Bedrock rejects images with either dimension over 8000px. Substack strips
# (e.g. wide formula crops) routinely exceed that, so downscale defensively
# rather than losing the visual to a hard API error.
MAX_IMAGE_DIM = 8000


def _downscale_if_needed(image_bytes: bytes, media_type: str) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(image_bytes)) as img:
        w, h = img.size
        if w <= MAX_IMAGE_DIM and h <= MAX_IMAGE_DIM:
            return image_bytes, media_type

        scale = MAX_IMAGE_DIM / max(w, h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = img.convert("RGB").resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


def _fetch_image(vid: str, url: str) -> tuple[bytes, str]:
    """Fetch through the figures cache.

    What lands on disk is the downscaled image actually sent to the vision
    model, not the original: it is the version worth keeping, and re-deriving
    it costs another download plus a resize.
    """
    cached = find_figure(vid)
    if cached is not None:
        media_type = mimetypes.guess_type(cached.name)[0] or "image/jpeg"
        return cached.read_bytes(), media_type

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    media_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
    if not media_type.startswith("image/"):
        media_type = mimetypes.guess_type(url)[0] or "image/jpeg"
    # SVG is vector, not raster: PIL can't open it and it has no pixel
    # dimensions to downscale, so skip straight past that path.
    if media_type == "image/svg+xml":
        image_bytes = resp.content
    else:
        image_bytes, media_type = _downscale_if_needed(resp.content, media_type)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ext = mimetypes.guess_extension(media_type) or ".jpg"
    figure_path(vid, ext).write_bytes(image_bytes)
    return image_bytes, media_type


def describe_visual(v: Visual, language: str = TARGET_LANGUAGE) -> None:
    image_bytes, media_type = _fetch_image(v.vid, v.fetch_url)
    if media_type == "image/svg+xml":
        # Badges (shields.io, arXiv, build status) are the near-universal case
        # here, and no vision model can read vector content anyway.
        v.klass, v.description = "decorativo", None
        save_visual_shard(v.vid, v.klass, v.description, language)
        return

    prompt = Template(VISUAL_PROMPT_PATH.read_text(encoding="utf-8")).render(
        HINT=v.hint, ALT=v.alt, CAPTION=v.caption, TARGET_LANGUAGE=language
    )
    raw, _ = generate_vision_response(prompt, image_bytes, media_type)
    match = _JSON_RE.search(raw)
    parsed = json.loads(match.group(0) if match else raw)

    v.klass = normalize_klass(parsed.get("klass"))
    v.description = (parsed.get("description") or "").strip() or None
    save_visual_shard(v.vid, v.klass, v.description, language)


def _describe_or_fallback(v: Visual, language: str = TARGET_LANGUAGE) -> None:
    try:
        describe_visual(v, language)
    except Exception:
        logger.warning(
            "describe_visual failed for %s, falling back to %s",
            v.fetch_url,
            SKIPPED_KLASS,
            exc_info=True,
        )
        v.klass, v.description = SKIPPED_KLASS, None


def describe_visuals(doc: Document, language: str = TARGET_LANGUAGE, max_workers: int = 8) -> None:
    """Mutates doc.visuals in place. Call once per document, after extract()."""
    loaded = 0
    todo = []
    for v in doc.visuals.values():
        if v.hero:
            v.klass, v.description = SKIPPED_KLASS, None
            continue
        shard = load_visual_shard(v.vid, language)
        if shard is not None:
            v.klass, v.description = normalize_klass(shard["klass"]), shard["description"]
            loaded += 1
        else:
            todo.append(v)

    if loaded:
        logger.info("describe_visuals: resumed %d visual(s) from shards", loaded)
    if not todo:
        return

    with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
        list(pool.map(lambda v: _describe_or_fallback(v, language), todo))
