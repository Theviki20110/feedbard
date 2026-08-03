from pathlib import Path

from jinja2 import Template

from llm_client import generate_response
from cleaning import VIS_RE

TRANSLATOR_PROMPT_PATH = Path(__file__).parent / "assets" / "translator_prompt.txt"

# Long articles cause the model to summarize/skip content instead of
# translating it in full. Chunking bounds each request to a size the model
# reliably translates in its entirety, and lets us catch a broken chunk
# immediately instead of discovering a silent mid-document drop at the end.
MAX_CHUNK_CHARS = 6000


def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Group paragraphs (blank-line separated) into chunks up to max_chars,
    never splitting a single paragraph."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for the "\n\n" join
        if current and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _translate_chunk(chunk: str, target_language: str, index: int) -> str:
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


def translate_text(text: str, target_language: str = "Italian") -> str:
    chunks = _split_into_chunks(text)
    translated_chunks = [
        _translate_chunk(chunk, target_language, i) for i, chunk in enumerate(chunks)
    ]
    return "\n\n".join(translated_chunks)
