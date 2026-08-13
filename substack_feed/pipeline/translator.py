import os
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Template

from substack_feed.ingestion.html_parser import VIS_RE
from substack_feed.llm_client import generate_response
from substack_feed.logger import logger
from substack_feed.paths import ASSETS_DIR, safe_filename

TRANSLATOR_PROMPT_PATH = ASSETS_DIR / "translator_prompt.txt"

# Long articles cause the model to summarize/skip content instead of
# translating it in full. Chunking bounds each request to a size the model
# reliably translates in its entirety, and lets us catch a broken chunk
# immediately instead of discovering a silent mid-document drop at the end.
MAX_CHUNK_CHARS = 6000

# Per-block translated text, written right after each LLM call succeeds. If
# the process is interrupted mid-article, restarting skips every block whose
# shard is already on disk instead of re-billing the model for it.
AUDIO_DIR = os.environ["AUDIO_DIR"]
TEXT_SHARDS_DIR = os.path.join(AUDIO_DIR, "text_shards")


def _shard_path(safe_title: str, index: int) -> str:
    return os.path.join(TEXT_SHARDS_DIR, f"{safe_title}_block{index}.txt")


def load_text_shard(safe_title: str, index: int) -> str | None:
    path = _shard_path(safe_title, index)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def save_text_shard(safe_title: str, index: int, translated_text: str) -> None:
    os.makedirs(TEXT_SHARDS_DIR, exist_ok=True)
    with open(_shard_path(safe_title, index), "w", encoding="utf-8") as f:
        f.write(translated_text)


def translate_chunk(chunk: str, target_language: str, index: int) -> str:
    expected = VIS_RE.findall(chunk)
    prompt = Template(TRANSLATOR_PROMPT_PATH.read_text()).render(
        SOURCE_TEXT=chunk, TARGET_LANGUAGE=target_language
    )
    translated_chunk, elapsed_time = generate_response(prompt)
    logger.info(
        "Translation chunk %d to %s completed in %.2f seconds", index, target_language, elapsed_time
    )

    found = VIS_RE.findall(translated_chunk)
    if found != expected:
        missing = [v for v in expected if v not in found]
        extra = [v for v in found if v not in expected]
        raise RuntimeError(
            f"chunk {index}: expected {len(expected)} placeholders, found {len(found)}; "
            f"missing={missing} extra={extra}"
        )
    return translated_chunk


def _translate_block(index: int, block, target_language: str, safe_title: str) -> None:
    block.translated_text = translate_chunk(block.text, target_language, index)
    save_text_shard(safe_title, index, block.translated_text)
    logger.debug(block.translated_text)


def translate_blocks(document, title: str, target_language: str = "Italian", max_workers: int = 8):
    safe_title = safe_filename(title)
    candidates = [(i, b) for i, b in enumerate(document.blocks) if b.text != ""]

    loaded = 0
    todo = []
    for i, b in candidates:
        shard = load_text_shard(safe_title, i)
        if shard is not None:
            b.translated_text = shard
            loaded += 1
        else:
            todo.append((i, b))

    if loaded:
        logger.info(
            "[%s] translate_blocks: resumed %d/%d block(s) from shards",
            title,
            loaded,
            len(candidates),
        )
    if not todo:
        return document

    with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
        list(
            pool.map(
                lambda item: _translate_block(item[0], item[1], target_language, safe_title), todo
            )
        )
    return document
