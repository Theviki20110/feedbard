from substack_feed.ingestion.html_parser import extract
from substack_feed.logger import logger
from substack_feed.pipeline.audio_renderer import generate_audio_from_blocks
from substack_feed.pipeline.translator import translate_blocks
from substack_feed.pipeline.visual_describer import describe_visuals


def process_item(item: dict) -> str:
    title = item["title"]
    logger.info("[%s] extract: parsing HTML", title)
    doc = extract(item["text"])

    logger.info("[%s] describe_visuals", title)
    describe_visuals(doc)

    logger.info("[%s] translate_text", title)
    translated_blocks = translate_blocks(doc)

    logger.info("[%s] generate_audio_from_blocks", title)
    dest = generate_audio_from_blocks(translated_blocks, title)
    return dest


def process_feeds(items: list[dict]) -> list[str]:
    logger.info("process_feeds: %d item(s)", len(items))
    return [process_item(item) for item in items]
