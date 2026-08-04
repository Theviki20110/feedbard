from jinja2 import Template

from substack_feed.llm_client import generate_response
from substack_feed.ingestion.html_parser import VIS_RE
from substack_feed.paths import ASSETS_DIR

TRANSLATOR_PROMPT_PATH = ASSETS_DIR / "translator_prompt.txt"

# Long articles cause the model to summarize/skip content instead of
# translating it in full. Chunking bounds each request to a size the model
# reliably translates in its entirety, and lets us catch a broken chunk
# immediately instead of discovering a silent mid-document drop at the end.
MAX_CHUNK_CHARS = 6000

def translate_chunk(chunk: str, target_language: str, index: int) -> str:
    expected = VIS_RE.findall(chunk)
    prompt = Template(TRANSLATOR_PROMPT_PATH.read_text()).render(
        SOURCE_TEXT=chunk, TARGET_LANGUAGE=target_language
    )
    translated_chunk, elapsed_time = generate_response(prompt)
    print(f"Translation chunk {index} to {target_language} completed in {elapsed_time:.2f} seconds")

    found = VIS_RE.findall(translated_chunk)
    if found != expected:
        missing = [v for v in expected if v not in found]
        extra = [v for v in found if v not in expected]
        raise RuntimeError(
            f"chunk {index}: expected {len(expected)} placeholders, found {len(found)}; "
            f"missing={missing} extra={extra}"
        )
    return translated_chunk


def translate_blocks(document, target_language: str = "Italian"):
    for index, block in enumerate(document.blocks):
        if block.text == "":
            continue
        block.translated_text = translate_chunk(block.text, target_language, index)
        print(block.translated_text)
    return document