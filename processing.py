import logging

from aggregator import aggregate_data
from cleaning import extract
from translation import translate_text
from tts import generate_audio_from_text, render_for_tts
from visuals import describe_visuals

def process_item(item: dict) -> str:
    title = item["title"]
    print("[%s] extract: parsing HTML", title)

    print("Extract")
    doc = extract(item["text"])

    print("Describe_visuals")
    describe_visuals(doc)

    print("Aggregate_data")
    aggregated = aggregate_data(doc)

    print("Translate_text")
    translated = translate_text(aggregated)

    print("Render_for_tts")
    final_text = render_for_tts(doc, translated)

    print("Generate_audio_from_text")
    dest = generate_audio_from_text(final_text, title)
    return dest


def process_feeds(items: list[dict]) -> list[str]:
    print("process_feeds: %d item(s)", len(items))
    return [process_item(item) for item in items]