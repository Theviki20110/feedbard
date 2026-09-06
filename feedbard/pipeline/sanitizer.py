"""Speech-preparation pass: turns translated text into something a TTS
engine can actually pronounce.

Two layers, on purpose:

1. An LLM pass, which is the only thing that can decide what a symbol MEANS.
   Whether `σ_1` should be narrated as "sigma uno, il primo valore singolare"
   or just "sigma uno" depends on the surrounding prose, and no table lookup
   can settle it.
2. A deterministic scrub, which runs unconditionally afterwards. The LLM is
   the part of this pipeline that has already been observed to fail open --
   returning refusals, meta-commentary, or simply forgetting a glyph -- so
   the guarantee that no raw symbol reaches the speech engine cannot rest on
   it. The scrub is what makes that guarantee.
"""

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Template

from feedbard.ingestion.html_parser import SYM_CLOSE, SYM_OPEN
from feedbard.llm_client import generate_response
from feedbard.logger import logger
from feedbard.paths import ASSETS_DIR, SPEECH_SHARDS_DIR, speech_shard_path
from feedbard.pipeline.lexicon import TARGET_LANGUAGE, load_lexicon

SANITIZER_PROMPT_PATH = ASSETS_DIR / "sanitizer_prompt.txt"
REPAIR_PROMPT_PATH = ASSETS_DIR / "repair_prompt.txt"

# How much of the preceding block to hand the model as context. Enough to
# resolve "the sigmas introduced above", not enough to double the bill.
CONTEXT_CHARS = 800

# How far a repair may move the length of a passage. A mend is a word or two
# of connective tissue; anything past this is the model rewriting the passage,
# which is exactly what the repair pass must not do.
REPAIR_LENGTH_TOLERANCE = 0.15


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

# Blocks that are empty once invisible characters are discounted. Sending one
# of these to an LLM produces a helpful "I notice the source content appears
# to be empty..." reply, in English, which then gets narrated to the listener.
INVISIBLE_RE = re.compile(r"[​-‏  ‪-‮⁠﻿­]")


def is_effectively_empty(text: str) -> bool:
    return not INVISIBLE_RE.sub("", text or "").strip()


# The model answering ABOUT the task instead of performing it. Every pattern
# here was observed in a real shard on disk, in output that was otherwise
# indistinguishable from article prose.
REFUSAL_RE = re.compile(
    r"I'm ready to (?:translate|help)"
    r"|I understand the instructions"
    r"|no source content was (?:actually )?provided"
    r"|(?:source content|input) (?:field )?(?:appears to be|is) empty"
    r"|zero-width character"
    r"|Please provide the (?:actual )?text"
    r"|Could you please provide"
    r"|\bI cannot\b|\bI can't\b|\bI'm unable to\b"
    r"|Non posso (?:tradurre|riprodurre|aiutart)"
    r"|Posso invece aiutarti"
    r"|protetto da copyright"
    r"|Mi dispiace, (?:non|ma)",
    re.I,
)


class ModelRefusedError(RuntimeError):
    """The model returned commentary about the task instead of the output.

    Raised rather than logged: a refusal that reaches disk is indistinguishable
    from real content downstream, and gets read aloud as if the author had
    written it.
    """


def check_usable(text: str, stage: str, index: int, source: str = "") -> None:
    # `source` is diagnostic only, never validated: knowing what was SENT is
    # what makes a refusal reproducible without re-running the whole article.
    src_note = f" | source: {source[:200]!r}" if source else ""
    if is_effectively_empty(text):
        raise ModelRefusedError(f"{stage} block {index}: model returned empty output{src_note}")
    match = REFUSAL_RE.search(text)
    if match:
        raise ModelRefusedError(
            f"{stage} block {index}: model returned commentary instead of content "
            f"(matched {match.group(0)!r}): {text[:200]!r}{src_note}"
        )


# --------------------------------------------------------------------------
# Deterministic scrub
# --------------------------------------------------------------------------

# Spoken names live in `assets/lexicons/<language>.json`, loaded through
# `lexicon.load_lexicon`. They are the backstop table, not the primary path:
# the LLM pass is expected to have handled these in context and with a gloss.
# Anything reaching here is a symbol the model dropped, so a bare correct
# pronunciation beats a glyph the engine spells out or skips.

# Digits written as sub/superscripts: the engine either skips them or reads
# them as ordinary digits glued to the previous word ("sigma1").
_SUB = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙₐₑₒₓ"
_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ"

# A single-letter variable with a short subscript: `σ_1`, `x_i`, `W_q`.
# The lookbehind keeps snake_case identifiers (`reasoning_effort`) out: there
# the underscore is preceded by a word character, not by a standalone letter.
SUBSCRIPT_RE = re.compile(r"(?<!\w)([A-Za-zͰ-Ͽ])_(\w{1,3})(?!\w)")

# Pseudo-XML and template tags: `<think>`, `</think>`, `<|assistant|>`.
# Reduced to the bare name so the voice says "think" instead of narrating
# angle brackets, or silently swallowing the tag along with what follows it.
TAG_RE = re.compile(r"</?\|?([A-Za-z][\w.:-]*)\|?\s*/?>")

MD_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*$", re.MULTILINE)
MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
MD_BULLET_RE = re.compile(r"^\s{0,3}[*+-]\s+", re.MULTILINE)
MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")

# `\n` that survived as two literal characters rather than a line break.
LITERAL_ESCAPE_RE = re.compile(r"\\+[nrt]")

# A bare URL. Read aloud, a link is a minute of spelled-out slashes and hex
# that carries nothing a listener can act on, so it never survives to the
# speech engine. Markdown links are handled earlier by MD_LINK_RE, which keeps
# the label and drops only the target.
URL_RE = re.compile(
    r"[ \t]*<?(?:https?://|ftp://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)\]}]>?[ \t]*\.?",
    re.I,
)

# The URL is replaced by its host before the LLM pass rather than deleted
# outright. Deleting first loses information the model could use -- "available
# on GitHub" is better than "available" -- and leaves a hole the model has no
# way to know about. The marker uses the same delimiters the parser puts around
# inline code, which the sanitizer prompt already knows how to consume.
LINK_MARKER = SYM_OPEN + "link: {host}" + SYM_CLOSE
LINK_MARKER_RE = re.compile(
    re.escape(SYM_OPEN)
    + r"link: [^"
    + re.escape(SYM_CLOSE)
    + r"]*"
    + re.escape(SYM_CLOSE)
    + r"[ \t]*\.?"
)

HOST_RE = re.compile(r"^(?:https?://|ftp://)?(?:www\.)?([^/\s:]+)", re.I)

# Punctuation left stranded once the URL after it is gone: an empty pair of
# brackets, the connector that introduced the link ("available here:"), a
# doubled stop. The connector becomes a full stop rather than disappearing,
# so the sentence still lands instead of trailing off.
EMPTY_PAIR_RE = re.compile(r"[ \t]*[(\[][ \t]*[)\]]")
DANGLING_CONNECTOR_RE = re.compile(r"[ \t]*[:,;\-–—][ \t]*$", re.MULTILINE)


def mark_urls(text: str) -> str:
    """Replace every URL with a `⟪link: host⟫` marker, for the LLM pass.

    Runs before the model sees the passage: a link it cannot see is a link it
    cannot leave in, and the marker tells it a source was named there so it can
    phrase the sentence around it instead of around a hole.
    """

    def to_marker(match: re.Match) -> str:
        url = match.group(0).strip().strip("<>").rstrip(".")
        host = HOST_RE.match(url)
        return " " + LINK_MARKER.format(host=host.group(1) if host else "link")

    return URL_RE.sub(to_marker, text)


def strip_urls(text: str) -> str:
    """Delete links and mend the punctuation they leave behind.

    The deterministic half of link handling: whatever the LLM pass did not
    consume -- a raw URL, or a marker it ignored -- is removed here, and the
    stranded connector becomes a full stop. Mending it into fluent prose is not
    something a regex can do; that is `repair_chunk`'s job.
    """
    text = LINK_MARKER_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = EMPTY_PAIR_RE.sub("", text)
    text = DANGLING_CONNECTOR_RE.sub(".", text)
    return re.sub(r"([.,;])[ \t]*\1+", r"\1", text)


def scrub(text: str, language: str = TARGET_LANGUAGE) -> str:
    """Last line of defence before the speech engine.

    Idempotent and lossless in the sense that matters here: it replaces a
    symbol with words in `language` and never drops a claim -- URLs excepted,
    which are unspeakable by nature. Runs on every block, including the ones
    the LLM pass was skipped for.
    """
    lex = load_lexicon(language)

    def spell_subscript(match: re.Match) -> str:
        base, sub = match.group(1), match.group(2)
        return f"{lex.symbols.get(base, base)} {sub}"

    text = LITERAL_ESCAPE_RE.sub(" ", text)
    text = INVISIBLE_RE.sub("", text)

    text = MD_FENCE_RE.sub("", text)
    # Before the URL pass, so a Markdown link keeps its label: the target is
    # dropped here, and there is nothing left for URL_RE to find.
    text = MD_LINK_RE.sub(r"\1", text)
    text = MD_EMPHASIS_RE.sub(r"\2", text)
    text = MD_HEADING_RE.sub("", text)
    text = MD_BULLET_RE.sub("", text)
    text = text.replace("`", "")

    # Before the inline-code and symbol passes: both would dissolve a `⟪link:
    # host⟫` marker into prose, and the symbol pass would turn a raw link's
    # slashes into spoken words and hide it from URL_RE.
    text = strip_urls(text)

    # Inline-code markers from the HTML parser. The LLM pass is supposed to
    # consume them; anything left is a segment it did not rewrite, so keep
    # the content and drop the brackets.
    text = text.replace(SYM_OPEN, " ").replace(SYM_CLOSE, " ")

    # Padded, not bare: adjacent tags (`<|assistant|><think>`) would otherwise
    # fuse into one unpronounceable word.
    text = TAG_RE.sub(r" \1 ", text)
    text = SUBSCRIPT_RE.sub(spell_subscript, text)

    # Unicode has canonical decompositions for sub/superscript digits, so the
    # normal form gives us the plain digit without a 30-entry table. Space it
    # out so `σ₁` is heard as "sigma uno" and not as the word "sigma1".
    text = "".join(
        " " + unicodedata.normalize("NFKC", ch) if ch in _SUB or ch in _SUP else ch for ch in text
    )

    for glyph, spoken in lex.symbols.items():
        if glyph in text:
            text = text.replace(glyph, spoken)

    # Whatever comparison syntax is left is not part of a formula the LLM
    # recognized, so spell it the plain way rather than leave it mute. Order
    # comes from the lexicon file: `<=` has to be consumed before `<`.
    for frm, to in lex.sequences:
        if frm in text:
            text = text.replace(frm, to)

    # Pure markup, deleted rather than spoken, so nothing here is per-language.
    text = text.replace("|", " ").replace("\\", " ")
    text = text.replace("➜", " ").replace("▶", " ").replace("•", " ")
    text = text.replace("{", " ").replace("}", " ")
    # Not a heading (those went earlier): a shell comment marker or a URL
    # fragment separator, both of which the engine would spell as "hash".
    text = text.replace("#", " ")

    # Collapse the whitespace the substitutions above introduced, without
    # flattening paragraph breaks, which carry prosody.
    text = re.sub(r"[ \t]+", " ", text)
    # Stripping a delimiter mid-sentence strands a space before the following
    # punctuation, which some engines render as an audible pause.
    text = re.sub(r" +([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Blocks with none of this are already speakable, and an LLM round-trip on
# them buys nothing but latency, cost, and a chance to paraphrase away a
# number. Kept deliberately broad: a false positive costs one call, a false
# negative ships a glyph to the voice.
#
# A capitalized name directly followed by a dotted version number ("DeepSeek
# 4.5", "GPT-4.5") is the same problem in prose form as the math symbols
# above: read raw, "4.5" is ambiguous between a version's "point" and a
# decimal's locale-specific separator, and the deterministic scrub has no way
# to know which one it is -- only the sanitizer prompt's contextual judgment
# does (see "What NOT to convert" in sanitizer_prompt.txt).
NEEDS_LLM_RE = re.compile(
    r"[Ͱ-Ͽ∀-⋿←-⇿" + re.escape(SYM_OPEN + SYM_CLOSE + _SUB + _SUP) + r"`|\\{}<>=]"
    r"|```|^\s{0,3}#{1,6}\s|^\s{0,3}[*+-]\s|\*\*|\[[^\]]+\]\("
    r"|(?<!\w)[A-Za-z]_\w|https?://|\\n"
    r"|\b[A-Z][A-Za-z]*[ -]\d+\.\d+\b",
    re.MULTILINE,
)


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


def repair_is_acceptable(before: str, after: str) -> str | None:
    """Reject a repair that did more than mend punctuation.

    Returns the reason to reject, or None to accept. The repair pass is asked
    for a surgical edit, so anything that changes the size or the symbol
    inventory of the passage is the model rewriting instead of mending -- and
    the unrepaired passage, awkward as it reads, still carries the author's
    content.
    """
    if is_effectively_empty(after):
        return "empty output"
    if REFUSAL_RE.search(after):
        return "commentary instead of content"
    if URL_RE.search(after) or LINK_MARKER_RE.search(after):
        return "reintroduced a link"
    if len(after) > len(before) * (1 + REPAIR_LENGTH_TOLERANCE):
        return f"grew by more than {REPAIR_LENGTH_TOLERANCE:.0%}"
    if len(after) < len(before) * (1 - REPAIR_LENGTH_TOLERANCE):
        return f"shrank by more than {REPAIR_LENGTH_TOLERANCE:.0%}"
    return None


def repair_chunk(text: str, index: int, target_language: str, title: str = "") -> str:
    """Mend a passage a link removal left ungrammatical.

    Only reachable on the scrub-only path -- a block the LLM pass skipped, or
    one where it failed -- because a successful pass has already phrased the
    sentence around the marker. Any doubt about the result keeps `text`: this
    stage can only improve a passage, never replace it.
    """
    prompt = Template(REPAIR_PROMPT_PATH.read_text()).render(
        SOURCE_TEXT=text,
        TARGET_LANGUAGE=target_language,
        ARTICLE_TITLE=title,
    )
    try:
        repaired, elapsed = generate_response(prompt)
        logger.info("Repair chunk %d completed in %.2f seconds", index, elapsed)
    except Exception as exc:
        # Not just RuntimeError: a transient network/API failure must fall
        # back to the scrubbed text the same way a truncation would.
        logger.warning("repair block %d: LLM call failed (%s); keeping scrubbed text", index, exc)
        return text

    repaired = scrub(repaired, target_language)
    reason = repair_is_acceptable(text, repaired)
    if reason:
        logger.warning("repair block %d: rejected (%s); keeping scrubbed text", index, reason)
        return text
    return repaired


def _scrub_only(text: str, index: int, target_language: str, title: str, had_links: bool) -> str:
    """The fallback path: scrub, then repair if a link removal left a hole."""
    scrubbed = scrub(text, target_language)
    if not had_links or is_effectively_empty(scrubbed):
        return scrubbed
    return repair_chunk(scrubbed, index, target_language, title)


def sanitize_chunk(
    chunk: str,
    index: int,
    title: str = "",
    previous_text: str = "",
    target_language: str = TARGET_LANGUAGE,
) -> str:
    """Rewrite one block for speech. Falls back to the scrub alone if the
    model misbehaves: a mechanically pronounced block still carries the
    author's content, whereas a refusal does not."""
    if is_effectively_empty(chunk):
        return ""

    # The model never sees a URL, only `⟪link: host⟫`. It cannot leave in what
    # it cannot see, and it phrases the sentence around a named source rather
    # than around the hole a deletion would have left.
    marked = mark_urls(chunk)
    had_links = marked != chunk

    if not NEEDS_LLM_RE.search(marked):
        return _scrub_only(marked, index, target_language, title, had_links)

    prompt = Template(SANITIZER_PROMPT_PATH.read_text()).render(
        SOURCE_TEXT=marked,
        TARGET_LANGUAGE=target_language,
        ARTICLE_TITLE=title,
        PREVIOUS_TEXT=strip_urls(previous_text[-CONTEXT_CHARS:]),
    )

    try:
        sanitized, elapsed = generate_response(prompt)
        logger.info("Sanitize chunk %d completed in %.2f seconds", index, elapsed)
        check_usable(sanitized, "sanitize", index, source=marked)
    except Exception as exc:
        # Not just ModelRefusedError/RuntimeError: a transient network/API
        # failure must fall back to the scrub-only path the same way a
        # refusal does, or it aborts the whole article instead of one block.
        logger.warning(
            "sanitize block %d: LLM pass unusable (%s); falling back to scrub only", index, exc
        )
        return _scrub_only(marked, index, target_language, title, had_links)

    return scrub(sanitized, target_language)


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
