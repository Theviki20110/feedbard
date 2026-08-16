from concurrent.futures import ThreadPoolExecutor

from jinja2 import Template

from feedbard.llm_client import generate_response
from feedbard.logger import logger
from feedbard.paths import ASSETS_DIR, TEXT_SHARDS_DIR, text_shard_path
from feedbard.pipeline.lexicon import TARGET_LANGUAGE
from feedbard.pipeline.sanitizer import ModelRefusedError, check_usable, is_effectively_empty

TRANSLATOR_PROMPT_PATH = ASSETS_DIR / "translator_prompt.txt"

# Per-block translated text, written right after each LLM call succeeds. If
# the process is interrupted mid-article, restarting skips every block whose
# shard is already on disk instead of re-billing the model for it.

# A refusal is sampled, not deterministic: the same prompt run again often
# just answers normally. Bounded so a block that refuses for a real reason
# (unlike sanitize_chunk, translate has no mechanical fallback to drop to)
# still surfaces as ModelRefusedError instead of retrying forever.
MAX_REFUSAL_RETRIES = 2


def load_text_shard(title: str, index: int) -> str | None:
    path = text_shard_path(title, index)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def save_text_shard(title: str, index: int, translated_text: str) -> None:
    TEXT_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    text_shard_path(title, index).write_text(translated_text, encoding="utf-8")


def translate_chunk(chunk: str, target_language: str, index: int) -> str:
    # A block that is empty once zero-width characters are discounted gets a
    # helpful English "the source content appears to be empty" reply from the
    # model, which then reaches the speech engine as if it were article prose.
    if is_effectively_empty(chunk):
        return ""

    prompt = Template(TRANSLATOR_PROMPT_PATH.read_text()).render(
        SOURCE_TEXT=chunk, TARGET_LANGUAGE=target_language
    )

    for attempt in range(MAX_REFUSAL_RETRIES + 1):
        translated_chunk, elapsed_time = generate_response(prompt)
        logger.info(
            "Translation chunk %d to %s completed in %.2f seconds",
            index,
            target_language,
            elapsed_time,
        )
        try:
            # Commentary about the task reads exactly like article prose
            # downstream, so it has to be rejected here rather than
            # discovered in the audio.
            check_usable(translated_chunk, "translate", index, source=chunk)
        except ModelRefusedError:
            if attempt == MAX_REFUSAL_RETRIES:
                raise
            logger.warning(
                "translate block %d: refused on attempt %d/%d, retrying",
                index,
                attempt + 1,
                MAX_REFUSAL_RETRIES + 1,
            )
            continue

        # The ⟪⟫ markers around inline code (added by the HTML parser) are
        # left in place on purpose: they tell the sanitizer stage which spans
        # are identifiers and commands rather than prose. That stage
        # consumes them.
        return translated_chunk


def _translate_block(index: int, block, target_language: str, title: str) -> None:
    block.translated_text = translate_chunk(block.text, target_language, index)
    save_text_shard(title, index, block.translated_text)
    logger.debug(block.translated_text)


def translate_blocks(
    document, title: str, target_language: str = TARGET_LANGUAGE, max_workers: int = 8
):
    # A block that already carries translated text got it from
    # `narration.attach_descriptions`, which works in the narration language:
    # there is nothing to translate, and doing so would paraphrase it.
    candidates = [
        (i, b) for i, b in enumerate(document.blocks) if b.text != "" and b.translated_text is None
    ]

    loaded = 0
    todo = []
    for i, b in candidates:
        shard = load_text_shard(title, i)
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
            pool.map(lambda item: _translate_block(item[0], item[1], target_language, title), todo)
        )
    return document
