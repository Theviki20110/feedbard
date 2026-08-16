"""Table -> a paragraph a listener can follow.

A table is the one block the parser produces that has content but no prose:
`_emit_table` records the grid in `Block.rows` and leaves `Block.text` empty.
Read cell by cell a grid is unlistenable; dropped, it takes its numbers with
it. So an LLM restates it, in the narration language, and
`narration.attach_descriptions` puts the result back where the table stood.

Shards are keyed on the content hash of the grid rather than on the article,
exactly as visuals are: the same table quoted by two posts is described once,
and an interrupted run resumes without re-billing the model.

When the model is unusable, the fallback is the grid read out flatly. It is
worse to listen to than a description and better than silence: no number is
lost, which is the only guarantee that matters here.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Template

from feedbard.ingestion.html_parser import Document, Kind
from feedbard.llm_client import generate_response
from feedbard.logger import logger
from feedbard.paths import ASSETS_DIR, TABLE_SHARDS_DIR, table_shard_path
from feedbard.pipeline.lexicon import TARGET_LANGUAGE
from feedbard.pipeline.sanitizer import ModelRefusedError, check_usable

TABLE_PROMPT_PATH = ASSETS_DIR / "table_prompt.txt"

# Rows handed to the model. A grid past this size is a data dump the listener
# was never going to follow row by row, and the prompt asks for the findings
# rather than the cells, so the tail adds cost without adding narration.
MAX_PROMPT_ROWS = 50


def table_id(rows: list[list[str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def render_rows(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def flatten_rows(rows: list[list[str]]) -> str:
    """The grid as a sequence of spoken clauses. Column identity is lost, the
    values are not."""
    return ". ".join(", ".join(cell.strip() for cell in row if cell.strip()) for row in rows)


def load_table_shard(tid: str, language: str = TARGET_LANGUAGE) -> str | None:
    path = table_shard_path(tid)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("language", language) != language:
        return None
    return data.get("description")


def save_table_shard(tid: str, description: str, language: str = TARGET_LANGUAGE) -> None:
    TABLE_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    table_shard_path(tid).write_text(
        json.dumps({"description": description, "language": language}, ensure_ascii=False),
        encoding="utf-8",
    )


def describe_table(rows: list[list[str]], title: str = "", language: str = TARGET_LANGUAGE) -> str:
    """One LLM call. Raises ModelRefusedError on output that cannot be
    narrated; the caller decides what to do with a table it could not phrase."""
    shown = rows[:MAX_PROMPT_ROWS]
    prompt = Template(TABLE_PROMPT_PATH.read_text(encoding="utf-8")).render(
        TABLE=render_rows(shown),
        ROW_COUNT=len(shown),
        TRUNCATED=len(rows) > len(shown),
        ARTICLE_TITLE=title,
        TARGET_LANGUAGE=language,
    )
    described, elapsed = generate_response(prompt)
    logger.info("Table %s described in %.2f seconds", table_id(rows), elapsed)
    check_usable(described, "describe_table", 0)
    return described.strip()


def _describe_or_flatten(block, title: str, language: str) -> None:
    tid = table_id(block.rows)
    try:
        description = describe_table(block.rows, title, language)
    except (ModelRefusedError, RuntimeError, ValueError):
        logger.warning(
            "describe_table failed for %s, falling back to reading the rows out",
            tid,
            exc_info=True,
        )
        block.description = flatten_rows(block.rows)
        return
    block.description = description
    save_table_shard(tid, description, language)


def describe_tables(
    doc: Document, title: str = "", language: str = TARGET_LANGUAGE, max_workers: int = 8
) -> None:
    """Mutates doc.blocks in place. Call once per document, after extract()."""
    tables = [b for b in doc.blocks if b.kind is Kind.TABLE and b.rows]

    loaded = 0
    todo = []
    for block in tables:
        shard = load_table_shard(table_id(block.rows), language)
        if shard is not None:
            block.description = shard
            loaded += 1
        else:
            todo.append(block)

    if loaded:
        logger.info("[%s] describe_tables: resumed %d table(s) from shards", title, loaded)
    if not todo:
        return

    with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
        list(pool.map(lambda b: _describe_or_flatten(b, title, language), todo))
