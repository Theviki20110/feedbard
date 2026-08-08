from substack_feed.ingestion.html_parser import extract
from substack_feed.pipeline.translator import translate_blocks
from substack_feed.pipeline.audio_renderer import generate_audio_from_blocks
from substack_feed.pipeline.visual_describer import describe_visuals

def process_item(item: dict) -> str:
    title = item["title"]
    print(f"[{title}] extract: parsing HTML")
    doc = extract(item["text"])

    print(f"[{title}] describe_visuals")
    describe_visuals(doc)

    print(f"[{title}] translate_text")
    translated_blocks = translate_blocks(doc)

    print(f"[{title}] generate_audio_from_blocks")
    dest = generate_audio_from_blocks(translated_blocks, title)
    return dest


def process_feeds(items: list[dict]) -> list[str]:
    print(f"process_feeds: {len(items)} item(s)")
    return [process_item(item) for item in items]