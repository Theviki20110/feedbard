"""LLM3: fills in Visual.klass and Visual.description by actually looking at
the image. Runs after extract() and before render_for_tts(), which is the
only place those two fields are read.

A visual that fails to fetch or to classify is treated as decorativo rather
than aborting the item: dropping one figure's narration is fine, silently
losing placeholder alignment (aggregator/tts's check_integrity) is not.
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

_JSON_RE = re.compile(r"\{.*\}", re.S)

# vid is a content hash of the image's canonical source, stable across runs
# and articles, so both the figure and its shard are keyed on it rather than
# on title: interrupting mid-article and restarting skips every
# already-described visual instead of re-billing the vision model for it, and
# a figure reused across two articles is fetched once.


def load_visual_shard(vid: str) -> dict | None:
    path = visual_shard_path(vid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_visual_shard(vid: str, klass: str, description: str | None) -> None:
    VISUAL_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    visual_shard_path(vid).write_text(
        json.dumps({"klass": klass, "description": description}), encoding="utf-8"
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
    image_bytes, media_type = _downscale_if_needed(resp.content, media_type)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ext = mimetypes.guess_extension(media_type) or ".jpg"
    figure_path(vid, ext).write_bytes(image_bytes)
    return image_bytes, media_type


def describe_visual(v: Visual) -> None:
    prompt = Template(open(VISUAL_PROMPT_PATH).read()).render(
        HINT=v.hint, ALT=v.alt, CAPTION=v.caption
    )
    image_bytes, media_type = _fetch_image(v.vid, v.fetch_url)
    raw, _ = generate_vision_response(prompt, image_bytes, media_type)
    match = _JSON_RE.search(raw)
    parsed = json.loads(match.group(0) if match else raw)

    klass = parsed.get("klass")
    v.klass = klass if klass in ("decorativo", "illustrativo", "essenziale") else "decorativo"
    v.description = (parsed.get("description") or "").strip() or None
    save_visual_shard(v.vid, v.klass, v.description)


def _describe_or_fallback(v: Visual) -> None:
    try:
        describe_visual(v)
    except Exception:
        logger.warning(
            "describe_visual failed for %s, falling back to decorativo",
            v.fetch_url,
            exc_info=True,
        )
        v.klass, v.description = "decorativo", None


def describe_visuals(doc: Document, max_workers: int = 8) -> None:
    """Mutates doc.visuals in place. Call once per document, after extract()."""
    loaded = 0
    todo = []
    for v in doc.visuals.values():
        if v.hero:
            v.klass, v.description = "decorativo", None
            continue
        shard = load_visual_shard(v.vid)
        if shard is not None:
            v.klass, v.description = shard["klass"], shard["description"]
            loaded += 1
        else:
            todo.append(v)

    if loaded:
        logger.info("describe_visuals: resumed %d visual(s) from shards", loaded)
    if not todo:
        return

    with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
        list(pool.map(_describe_or_fallback, todo))
