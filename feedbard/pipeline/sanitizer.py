"""Speech-preparation pass: turns translated text into something a TTS
engine can actually pronounce.

The rule is an allowlist, and it lives in `speakable`: a passage made only of
letters, marks, digits, spaces and punctuation is already speech, and is
passed through untouched. Anything else -- a symbol, a URL, a fragment of
code, an emoji, a rule drawn out of box characters -- is something only the
model can turn into words, because what it should *say* depends on what it
means here. So the passage goes to the model, and what comes back is checked
against the same allowlist that sent it.

There is no table of spoken forms anywhere in this pipeline, and no list of
symbols to keep up to date. There used to be one per language, which is what
made the whole thing an Italian pipeline wearing a `TARGET_LANGUAGE` flag.

Three attempts, then a scrub:

1. Already speakable -> returned as-is, no call.
2. One call. If the answer is speakable, done.
3. One more call, told what it left behind. If that is speakable, done.
4. `strip_unspeakable` deletes what is left. Reached only when the model has
   failed twice, and logged as the failure it is.
"""

from concurrent.futures import ThreadPoolExecutor

from jinja2 import Template

from feedbard.language import TARGET_LANGUAGE, display_name
from feedbard.llm_client import generate_response
from feedbard.logger import logger
from feedbard.paths import ASSETS_DIR, SPEECH_SHARDS_DIR, speech_shard_path
from feedbard.pipeline.speakable import (
    describe_unspeakable,
    is_effectively_empty,
    needs_llm,
    remove_invisible,
    strip_unspeakable,
)

SANITIZER_PROMPT_PATH = ASSETS_DIR / "sanitizer_prompt.txt"

# How much of the preceding block to hand the model as context. Enough to
# resolve "the sigmas introduced above", not enough to double the bill.
CONTEXT_CHARS = 800

# One retry, and only one. The second call is told exactly which characters
# survived, which is the only new information available; a third would repeat
# the second with nothing added.
MAX_ATTEMPTS = 2


# `is_effectively_empty` is re-exported from `speakable`: an empty answer is
# how every prompt in `assets/` tells a model to decline. That turns a refusal
# into a value the pipeline can act on -- the block is dropped -- instead of a
# paragraph of commentary that reads exactly like article prose and gets
# narrated to the listener as if the author had written it.


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------


def load_speech_shard(title: str, index: int) -> str | None:
    path = speech_shard_path(title, index)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def save_speech_shard(title: str, index: int, text: str) -> None:
    SPEECH_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    speech_shard_path(title, index).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# LLM pass
# --------------------------------------------------------------------------


def _render_prompt(chunk: str, title: str, previous_text: str, language: str, leftover: str) -> str:
    return Template(SANITIZER_PROMPT_PATH.read_text(encoding="utf-8")).render(
        SOURCE_TEXT=chunk,
        TARGET_LANGUAGE=display_name(language),
        ARTICLE_TITLE=title,
        PREVIOUS_TEXT=previous_text[-CONTEXT_CHARS:],
        LEFTOVER=leftover,
    )


def sanitize_chunk(
    chunk: str,
    index: int,
    title: str = "",
    previous_text: str = "",
    target_language: str = TARGET_LANGUAGE,
) -> str:
    """Rewrite one block for speech, or return it unchanged if it already is."""
    if is_effectively_empty(chunk):
        return ""

    # Invisible characters are deleted rather than described: there is nothing
    # for a model to decide about a character that makes no sound.
    chunk = remove_invisible(chunk)
    if not needs_llm(chunk):
        return chunk

    current = chunk
    leftover = describe_unspeakable(chunk)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = _render_prompt(chunk, title, previous_text, target_language, leftover)
        try:
            answer, elapsed = generate_response(prompt)
        except Exception as exc:
            # Not just RuntimeError: a transient network/API failure (Ollama
            # down, a throttled Anthropic/Bedrock call) must fall back to the
            # delete-what-cannot-be-spoken path the same way a bad reply does,
            # or it aborts the whole article instead of one block.
            logger.warning("sanitize block %d: call %d failed (%s)", index, attempt, exc)
            break

        logger.info("Sanitize chunk %d completed in %.2f seconds", index, elapsed)

        if is_effectively_empty(answer):
            # The prompt's way of declining. Nothing to narrate here, and
            # nothing to repair either.
            logger.warning("sanitize block %d: model returned nothing; dropping the block", index)
            return ""

        current = answer
        if not needs_llm(answer):
            return answer

        leftover = describe_unspeakable(answer)
        logger.warning(
            "sanitize block %d: attempt %d left %s", index, attempt, leftover or "a version number"
        )

    scrubbed = strip_unspeakable(current)
    # Error, not warning, and with both texts. This path deletes whole tokens
    # rather than replacing them with words, so what reaches the listener is a
    # mutilated sentence ("complessità da a O(n)") rather than a
    # mispronounced symbol. That is harder to catch by ear than the failure it
    # replaced, and the shard on disk keeps no record of what was removed, so
    # the log line is the only trace there is.
    logger.error(
        "sanitize block %d: giving up after %d attempt(s) and deleting what could not be "
        "spoken (%s)\n  before: %r\n  after:  %r",
        index,
        MAX_ATTEMPTS,
        describe_unspeakable(current) or "a version number",
        current,
        scrubbed,
    )
    return scrubbed


def _sanitize_block(
    index: int, block, title: str, previous_text: str, target_language: str
) -> None:
    block.speech_text = sanitize_chunk(
        block.translated_text, index, title, previous_text, target_language
    )
    save_speech_shard(title, index, block.speech_text)


def sanitize_blocks(
    document, title: str, target_language: str = TARGET_LANGUAGE, max_workers: int = 8
):
    """Populate block.speech_text for every translated block.

    Runs between translation and TTS. Blocks keep their translated_text
    untouched so a bad sanitizer run can be re-done from shards without
    re-billing translation.
    """
    candidates = [
        (i, b) for i, b in enumerate(document.blocks) if b.translated_text not in (None, "")
    ]

    # Context is the previously translated block, in document order, so it is
    # available regardless of which blocks resume from shards.
    previous: dict[int, str] = {}
    last = ""
    for i, b in candidates:
        previous[i] = last
        last = b.translated_text

    loaded = 0
    todo = []
    for i, b in candidates:
        shard = load_speech_shard(title, i)
        if shard is not None:
            b.speech_text = shard
            loaded += 1
        else:
            todo.append((i, b))

    if loaded:
        logger.info(
            "[%s] sanitize_blocks: resumed %d/%d block(s) from shards",
            title,
            loaded,
            len(candidates),
        )
    if not todo:
        return document

    with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
        list(
            pool.map(
                lambda item: _sanitize_block(
                    item[0], item[1], title, previous[item[0]], target_language
                ),
                todo,
            )
        )
    return document
