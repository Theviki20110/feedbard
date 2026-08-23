"""Editorial judgements about an article's outline, made once per article.

Three decisions used to be made by regular expressions over the article's
text: where the bibliography starts, which blocks carry nothing a listener
needs, and which paragraph refers back to the figure above it. Each of those
is a judgement about *meaning*, and each was written as a list of English
phrases with a few Italian ones appended -- so a French or Japanese article
had its bibliography narrated in full, its "buy the book" list items read
out, and its figures left in the wrong order, silently.

One call per article replaces all of it. The model sees the outline (index,
kind, an excerpt) and returns indices, never text, so it cannot rewrite the
article by accident.

Nothing is removed from `doc.blocks`. A dropped block is marked and emptied
in place, because every per-block shard on disk is keyed by the block's
index: deleting one would shift every index after it and silently invalidate
the whole article's cached translation.

Failing open is deliberate. A bad call here loses part of an article; no call
at all just narrates a bibliography. So any error -- the model, the JSON, an
index that does not exist -- leaves the document exactly as parsed.
"""

from __future__ import annotations

import json
import re

from jinja2 import Template

from feedbard.ingestion.html_parser import Document, Kind
from feedbard.llm_client import generate_response
from feedbard.logger import logger
from feedbard.paths import ASSETS_DIR, TRIAGE_SHARDS_DIR, triage_shard_path

TRIAGE_PROMPT_PATH = ASSETS_DIR / "triage_prompt.txt"

# Per block, in the outline the model reads. Enough to recognize what a block
# is; a bibliography entry and a paragraph are distinguishable in far less.
EXCERPT_CHARS = 180

_JSON_RE = re.compile(r"\{.*\}", re.S)

EMPTY: dict = {"tail_from": None, "drop": [], "refers_back": []}


def _excerpt(doc: Document, block) -> str:
    if block.kind is Kind.VISUAL:
        visual = doc.visuals.get(block.vid or "")
        parts = [visual.caption or "", visual.alt or ""] if visual else []
        text = " / ".join(p for p in parts if p) or "(no caption)"
    elif block.kind is Kind.TABLE:
        rows = block.rows or []
        text = " | ".join(rows[0]) if rows else "(empty table)"
    else:
        text = block.text or ""
    return re.sub(r"\s+", " ", text)[:EXCERPT_CHARS].strip()


def build_outline(doc: Document) -> str:
    return "\n".join(
        f"{i}\t{b.kind.value}\t{_excerpt(doc, b)}" for i, b in enumerate(doc.blocks)
    )


def _valid_indices(values, limit: int) -> list[int]:
    return sorted({v for v in values if isinstance(v, int) and 0 <= v < limit})


def parse_verdict(raw: str, block_count: int) -> dict:
    """Model output -> a verdict with only in-range indices, or EMPTY."""
    match = _JSON_RE.search(raw or "")
    if not match:
        logger.warning("triage: no JSON object in the reply: %r", (raw or "")[:200])
        return dict(EMPTY)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("triage: reply was not valid JSON: %r", match.group(0)[:200])
        return dict(EMPTY)
    if not isinstance(data, dict):
        return dict(EMPTY)

    tail = data.get("tail_from")
    if not (isinstance(tail, int) and 0 <= tail < block_count):
        tail = None

    return {
        "tail_from": tail,
        "drop": _valid_indices(data.get("drop") or [], block_count),
        "refers_back": _valid_indices(data.get("refers_back") or [], block_count),
    }


def load_shard(title: str) -> dict | None:
    path = triage_shard_path(title)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("triage: %s is unreadable; ignoring it", path, exc_info=True)
        return None


def save_shard(title: str, verdict: dict) -> None:
    TRIAGE_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    triage_shard_path(title).write_text(json.dumps(verdict), encoding="utf-8")


def ask(doc: Document, title: str) -> dict:
    prompt = Template(TRIAGE_PROMPT_PATH.read_text(encoding="utf-8")).render(
        ARTICLE_TITLE=title, OUTLINE=build_outline(doc)
    )
    try:
        raw, elapsed = generate_response(prompt)
    except RuntimeError:
        logger.warning("triage: call failed; leaving the article as parsed", exc_info=True)
        return dict(EMPTY)
    logger.info("[%s] triage: judged in %.2f seconds", title, elapsed)
    return parse_verdict(raw, len(doc.blocks))


def _drop(block) -> None:
    """Empty a block without removing it, so later indices do not shift."""
    block.dropped = True
    block.text = ""
    block.description = None


def reposition_visuals(doc: Document) -> None:
    """Play a paragraph before the figure it refers back to.

    DOM position is not always the right one for listening: if a figure comes
    before the paragraph that comments on it, the listener hears the
    description with no context yet.
    """
    out = []
    i, n = 0, len(doc.blocks)
    while i < n:
        blk = doc.blocks[i]
        nxt = doc.blocks[i + 1] if i + 1 < n else None
        if (
            blk.kind is Kind.VISUAL
            and nxt is not None
            and nxt.kind in (Kind.PROSE, Kind.LIST)
            and nxt.deictic == "back"
        ):
            out.extend([nxt, blk])
            i += 2
            continue
        out.append(blk)
        i += 1
    doc.blocks = out


def apply(doc: Document, verdict: dict) -> dict:
    """Mutate `doc` in place. Returns what was actually applied, for logging."""
    tail = verdict.get("tail_from")
    dropped = list(verdict.get("drop") or [])
    refers_back = list(verdict.get("refers_back") or [])

    if tail is not None:
        doc.truncated_at = doc.blocks[tail].text or doc.blocks[tail].kind.value
        for block in doc.blocks[tail:]:
            _drop(block)

    for index in dropped:
        _drop(doc.blocks[index])

    for index in refers_back:
        doc.blocks[index].deictic = "back"
    reposition_visuals(doc)

    return {"tail_from": tail, "drop": dropped, "refers_back": refers_back}


def triage(doc: Document, title: str = "") -> dict:
    """Judge the outline once, cache the verdict, and apply it."""
    if not doc.blocks:
        return dict(EMPTY)

    verdict = load_shard(title)
    if verdict is None:
        verdict = ask(doc, title)
        save_shard(title, verdict)
    else:
        logger.info("[%s] triage: resumed the verdict from a shard", title)

    applied = apply(doc, verdict)
    logger.info(
        "[%s] triage: tail from %s, %d block(s) dropped, %d paragraph(s) moved before their figure",
        title,
        applied["tail_from"] if applied["tail_from"] is not None else "nowhere",
        len(applied["drop"]),
        len(applied["refers_back"]),
    )
    return applied
