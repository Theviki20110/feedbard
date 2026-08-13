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
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from jinja2 import Template
from PIL import Image

from substack_feed.ingestion.html_parser import Document, Visual
from substack_feed.llm_client import generate_vision_response
from substack_feed.logger import logger
from substack_feed.paths import ASSETS_DIR

VISUAL_PROMPT_PATH = ASSETS_DIR / "visual_prompt.txt"

_JSON_RE = re.compile(r"\{.*\}", re.S)

# vid is a content hash of the image's canonical source, stable across runs
# and articles, so the shard is keyed on it directly rather than on title:
# interrupting mid-article and restarting skips every already-described
# visual instead of re-billing the vision model for it.
AUDIO_DIR = os.environ["AUDIO_DIR"]
VISUAL_SHARDS_DIR = os.path.join(AUDIO_DIR, "visual_shards")


def _shard_path(vid: str) -> str:
    return os.path.join(VISUAL_SHARDS_DIR, f"{vid}.json")


def load_visual_shard(vid: str) -> dict | None:
    path = _shard_path(vid)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_visual_shard(vid: str, klass: str, description: str | None) -> None:
    os.makedirs(VISUAL_SHARDS_DIR, exist_ok=True)
    with open(_shard_path(vid), "w", encoding="utf-8") as f:
        json.dump({"klass": klass, "description": description}, f)


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


def _fetch_image(url: str) -> tuple[bytes, str]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    media_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
    if not media_type.startswith("image/"):
        media_type = mimetypes.guess_type(url)[0] or "image/jpeg"
    return _downscale_if_needed(resp.content, media_type)


def describe_visual(v: Visual) -> None:
    prompt = Template(open(VISUAL_PROMPT_PATH).read()).render(
        HINT=v.hint, ALT=v.alt, CAPTION=v.caption
    )
    image_bytes, media_type = _fetch_image(v.fetch_url)
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
