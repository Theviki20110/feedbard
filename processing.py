from aggregator import aggregate_data
from cleaning import extract
from translation import translate_text
from tts import generate_audio_from_text


def process_item(item: dict) -> str:
    blocks = extract(item["text"])
    aggregated = aggregate_data()
    translated = translate_text(aggregated)
    return generate_audio_from_text(translated, item["title"])


def process_feeds(items: list[dict]) -> list[str]:
    return [process_item(item) for item in items]