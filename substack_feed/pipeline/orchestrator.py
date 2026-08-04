import logging

from substack_feed.pipeline.text_aggregator import aggregate_data
from substack_feed.ingestion.html_parser import extract
from substack_feed.pipeline.translator import translate_blocks
from substack_feed.pipeline.audio_renderer import generate_audio_from_blocks
from substack_feed.pipeline.visual_describer import describe_visuals

def process_item(item: dict) -> str:
    title = item["title"]
    print("[%s] extract: parsing HTML", title)

    print("Extract")
    doc = extract(item["text"])

    print("Describe_visuals")
    describe_visuals(doc)

    print("Translate_text")
    translated_blocks = translate_blocks(doc)

    print("Generate_audio_from_blocks")
    dest = generate_audio_from_blocks(translated_blocks, title)
    return dest


def process_feeds(items: list[dict]) -> list[str]:
    print("process_feeds: %d item(s)", len(items))
    return [process_item(item) for item in items]