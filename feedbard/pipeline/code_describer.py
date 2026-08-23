"""Code block -> a paragraph a listener can follow.

`_emit_code`/`_emit_code_wrapper` record a CODE block's raw text and
language. That text is not always runnable code: a newsletter's "diagram"
is often the same `<pre>` styling wrapped around an ASCII illustration
(boxes, arrows, repeated characters used to show a shape rather than to
run). Read literally -- narrated like prose -- either kind is unlistenable,
so an LLM describes what the block does or shows instead, in the narration
language, and `narration.attach_descriptions` puts the result back where the
block stood.

Shards are keyed on the content hash of the code, exactly as tables and
visuals are: the same snippet quoted by two posts is described once, and an
interrupted run resumes without re-billing the model.

When the model is unusable, the block is dropped rather than read raw: unlike
a table's numbers, code read character by character carries no information a
listener can use, so silence is the safer failure than noise.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Template

from feedbard.ingestion.html_parser import Document, Kind
from feedbard.language import TARGET_LANGUAGE
from feedbard.llm_client import generate_response
from feedbard.logger import logger
from feedbard.paths import ASSETS_DIR, CODE_SHARDS_DIR, code_shard_path
from feedbard.pipeline.sanitizer import is_effectively_empty

CODE_PROMPT_PATH = ASSETS_DIR / "code_prompt.txt"


def code_id(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:10]


def load_code_shard(cid: str, language: str = TARGET_LANGUAGE) -> str | None:
    path = code_shard_path(cid)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("language", language) != language:
        return None
    return data.get("description")


def save_code_shard(cid: str, description: str, language: str = TARGET_LANGUAGE) -> None:
    CODE_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    code_shard_path(cid).write_text(
        json.dumps({"description": description, "language": language}, ensure_ascii=False),
        encoding="utf-8",
    )


def describe_code(
    code: str, lang: str | None, title: str = "", language: str = TARGET_LANGUAGE
) -> str:
    """One LLM call. Returns "" when the model declines, which the prompt
    tells it to signal with an empty answer rather than with an explanation
    that would otherwise be narrated as if the author had written it."""
    prompt = Template(CODE_PROMPT_PATH.read_text(encoding="utf-8")).render(
        CODE=code,
        CODE_LANGUAGE=lang or "unknown",
        LINE_COUNT=len([ln for ln in code.splitlines() if ln.strip()]),
        ARTICLE_TITLE=title,
        TARGET_LANGUAGE=language,
    )
    described, elapsed = generate_response(prompt)
    logger.info("Code block %s described in %.2f seconds", code_id(code), elapsed)
    if is_effectively_empty(described):
        logger.warning("describe_code %s: model returned nothing", code_id(code))
        return ""
    return described.strip()


def _describe_or_drop(block, title: str, language: str) -> None:
    cid = code_id(block.text)
    try:
        description = describe_code(block.text, block.lang, title, language)
    except (RuntimeError, ValueError):
        logger.warning(
            "describe_code failed for %s, dropping the block from narration", cid, exc_info=True
        )
        block.description = None
        return
    if not description:
        block.description = None
        return
    block.description = description
    save_code_shard(cid, description, language)


def describe_code_blocks(
    doc: Document, title: str = "", language: str = TARGET_LANGUAGE, max_workers: int = 8
) -> None:
    """Mutates doc.blocks in place. Call once per document, after extract()."""
    blocks = [b for b in doc.blocks if b.kind is Kind.CODE and b.text]

    loaded = 0
    todo = []
    for block in blocks:
        shard = load_code_shard(code_id(block.text), language)
        if shard is not None:
            block.description = shard
            loaded += 1
        else:
            todo.append(block)

    if loaded:
        logger.info("[%s] describe_code_blocks: resumed %d block(s) from shards", title, loaded)
    if not todo:
        return

    with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
        list(pool.map(lambda b: _describe_or_drop(b, title, language), todo))
