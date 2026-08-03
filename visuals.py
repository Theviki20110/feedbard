"""LLM3: fills in Visual.klass and Visual.description by actually looking at
the image. Runs after extract() and before render_for_tts(), which is the
only place those two fields are read.

A visual that fails to fetch or to classify is treated as decorativo rather
than aborting the item: dropping one figure's narration is fine, silently
losing placeholder alignment (aggregator/tts's check_integrity) is not.
"""

from __future__ import annotations

import json
import mimetypes
import re

import requests
from jinja2 import Template

from llm_client import generate_vision_response
from cleaning import Document, Visual

VISUAL_PROMPT_PATH = __file__.rsplit("/", 1)[0] + "/assets/visual_prompt.txt"

_JSON_RE = re.compile(r"\{.*\}", re.S)


def _fetch_image(url: str) -> tuple[bytes, str]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    media_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
    if not media_type.startswith("image/"):
        media_type = mimetypes.guess_type(url)[0] or "image/jpeg"
    return resp.content, media_type


def describe_visual(v: Visual) -> None:
    prompt = Template(open(VISUAL_PROMPT_PATH).read()).render(
        HINT=v.hint, ALT=v.alt, CAPTION=v.caption)
    image_bytes, media_type = _fetch_image(v.fetch_url)
    raw, _ = generate_vision_response(prompt, image_bytes, media_type)
    match = _JSON_RE.search(raw)
    parsed = json.loads(match.group(0) if match else raw)

    klass = parsed.get("klass")
    v.klass = klass if klass in ("decorativo", "illustrativo", "essenziale") else "decorativo"
    v.description = (parsed.get("description") or "").strip() or None


def describe_visuals(doc: Document) -> None:
    """Mutates doc.visuals in place. Call once per document, after extract()."""
    for v in doc.visuals.values():
        if v.hero:
            v.klass, v.description = "decorativo", None
            continue
        try:
            describe_visual(v)
        except Exception:
            v.klass, v.description = "decorativo", None
