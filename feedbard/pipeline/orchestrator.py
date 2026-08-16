from feedbard.ingestion.html_parser import extract
from feedbard.logger import logger
from feedbard.pipeline.audio_renderer import generate_audio_from_blocks
from feedbard.pipeline.publisher import publish
from feedbard.pipeline.sanitizer import sanitize_blocks
from feedbard.pipeline.translator import translate_blocks
from feedbard.pipeline.visual_describer import describe_visuals


def process_item(item: dict) -> str:
    title = item["title"]
    logger.info("[%s] extract: parsing HTML", title)
    doc = extract(item["text"])

    logger.info("[%s] describe_visuals", title)
    describe_visuals(doc)

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
