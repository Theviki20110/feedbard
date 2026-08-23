from feedbard.ingestion.html_parser import Document, extract
from feedbard.logger import logger
from feedbard.pipeline.audio_renderer import generate_audio_from_blocks
from feedbard.pipeline.code_describer import describe_code_blocks
from feedbard.pipeline.narration import attach_descriptions
from feedbard.pipeline.publisher import publish
from feedbard.pipeline.sanitizer import sanitize_blocks
from feedbard.pipeline.table_describer import describe_tables
from feedbard.pipeline.translator import translate_blocks
from feedbard.pipeline.triage import triage
from feedbard.pipeline.visual_describer import describe_visuals


def _log_extraction(doc: Document, title: str) -> None:
    """What the parse kept and what it threw away.

    Both are judgement calls made by selector lists that no publisher is
    obliged to keep matching, so a run that quietly starts dropping half an
    article should say so somewhere.
    """
    logger.info(
        "[%s] extract: %d block(s), %d visual(s), %d footnote(s)",
        title,
        len(doc.blocks),
        len(doc.visuals),
        len(doc.notes),
    )
    if doc.dropped:
        logger.info(
            "[%s] extract: dropped boilerplate %s",
            title,
            ", ".join(f"{sel}x{n}" for sel, n in sorted(doc.dropped.items())),
        )


def process_item(item: dict) -> str:
    title = item["title"]
    logger.info("[%s] extract: parsing HTML", title)
    doc = extract(item["text"])
    _log_extraction(doc, title)

    # Before anything is billed per block: what triage drops is never
    # described, translated, or synthesized.
    logger.info("[%s] triage", title)
    triage(doc, title)

    logger.info("[%s] describe_visuals", title)
    describe_visuals(doc)

    logger.info("[%s] describe_tables", title)
    describe_tables(doc, title)

    logger.info("[%s] describe_code_blocks", title)
    describe_code_blocks(doc, title)

    # Before translation: this is what gives the figures, tables and code
    # blocks a place in the episode instead of leaving them as silent gaps
    # (or, for code, being read aloud as literal text).
    attached = attach_descriptions(doc)
    logger.info("[%s] attach_descriptions: %d non-prose block(s) narratable", title, attached)

    logger.info("[%s] translate_text", title)
    translated_blocks = translate_blocks(doc, title)

    logger.info("[%s] sanitize_text", title)
    speakable_blocks = sanitize_blocks(translated_blocks, title)

    logger.info("[%s] generate_audio_from_blocks", title)
    dest = generate_audio_from_blocks(speakable_blocks, title)

    logger.info("[%s] publish", title)
    publish(item)
    return dest


def process_feeds(items: list[dict]) -> list[str]:
    logger.info("process_feeds: %d item(s)", len(items))
    return [process_item(item) for item in items]
