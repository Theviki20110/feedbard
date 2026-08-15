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

from substack_feed.ingestion.html_parser import SYM_CLOSE, SYM_OPEN, VIS_RE
from substack_feed.llm_client import generate_response
from substack_feed.logger import logger
from substack_feed.paths import ASSETS_DIR, SPEECH_SHARDS_DIR, speech_shard_path

SANITIZER_PROMPT_PATH = ASSETS_DIR / "sanitizer_prompt.txt"

# How much of the preceding block to hand the model as context. Enough to
# resolve "the sigmas introduced above", not enough to double the bill.
CONTEXT_CHARS = 800


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


def check_usable(text: str, stage: str, index: int) -> None:
    if is_effectively_empty(text):
        raise ModelRefusedError(f"{stage} block {index}: model returned empty output")
    match = REFUSAL_RE.search(text)
    if match:
        raise ModelRefusedError(
            f"{stage} block {index}: model returned commentary instead of content "
            f"(matched {match.group(0)!r}): {text[:200]!r}"
        )


# --------------------------------------------------------------------------
# Deterministic scrub
# --------------------------------------------------------------------------

# Spoken names for the Italian voice. This is the backstop table, not the
# primary path: the LLM pass is expected to have handled these in context and
# with a gloss. Anything reaching here is a symbol the model dropped, so a
# bare correct pronunciation beats a glyph the engine spells out or skips.
SPOKEN_IT = {
    "α": "alfa",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi greco",
    "ρ": "rho",
    "ς": "sigma",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "ϕ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Α": "Alfa",
    "Β": "Beta",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Ε": "Epsilon",
    "Ζ": "Zeta",
    "Η": "Eta",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Μ": "Mu",
    "Ξ": "Xi",
    "Π": "produttoria",
    "Ρ": "Rho",
    "Σ": "sommatoria",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Ω": "Omega",
    "∑": "sommatoria",
    "∏": "produttoria",
    "∫": "integrale",
    "∂": "derivata parziale",
    "∇": "nabla",
    "√": "radice quadrata di",
    "∞": "infinito",
    "∈": "appartiene a",
    "∉": "non appartiene a",
    "⊂": "è contenuto in",
    "∪": "unione",
    "∩": "intersezione",
    "→": "verso",
    "←": "da",
    "⇒": "implica",
    "↔": "corrisponde a",
    "≈": "circa uguale a",
    "≠": "diverso da",
    "≤": "minore o uguale a",
    "≥": "maggiore o uguale a",
    "≪": "molto minore di",
    "≫": "molto maggiore di",
    "±": "più o meno",
    "×": "per",
    "÷": "diviso",
    "·": "per",
    "−": "meno",
    "≡": "identico a",
    "∝": "proporzionale a",
    "°": " gradi",
    "µ": "micro",
    "℃": " gradi Celsius",
    "➜": " ",
    "▶": " ",
    "•": " ",
}

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


def _spell_subscript(match: re.Match) -> str:
    base, sub = match.group(1), match.group(2)
    return f"{SPOKEN_IT.get(base, base)} {sub}"


def scrub(text: str) -> str:
    """Last line of defence before the speech engine.

    Idempotent and lossless in the sense that matters here: it only ever
    replaces a symbol with words, never drops a claim. Runs on every block,
    including the ones the LLM pass was skipped for.
    """
    text = LITERAL_ESCAPE_RE.sub(" ", text)
    text = INVISIBLE_RE.sub("", text)

    # Inline-code markers from the HTML parser. The LLM pass is supposed to
    # consume them; anything left is a segment it did not rewrite, so keep
    # the content and drop the brackets.
    text = text.replace(SYM_OPEN, " ").replace(SYM_CLOSE, " ")

    text = MD_FENCE_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = MD_EMPHASIS_RE.sub(r"\2", text)
    text = MD_HEADING_RE.sub("", text)
    text = MD_BULLET_RE.sub("", text)
    text = text.replace("`", "")

    # Padded, not bare: adjacent tags (`<|assistant|><think>`) would otherwise
    # fuse into one unpronounceable word.
    text = TAG_RE.sub(r" \1 ", text)
    text = SUBSCRIPT_RE.sub(_spell_subscript, text)

    # Unicode has canonical decompositions for sub/superscript digits, so the
    # normal form gives us the plain digit without a 30-entry table. Space it
    # out so `σ₁` is heard as "sigma uno" and not as the word "sigma1".
    text = "".join(
        " " + unicodedata.normalize("NFKC", ch) if ch in _SUB or ch in _SUP else ch for ch in text
    )

    for glyph, spoken in SPOKEN_IT.items():
        if glyph in text:
            text = text.replace(glyph, spoken)

    # Whatever comparison or pipe syntax is left is not part of a formula the
    # LLM recognized, so spell it the plain way rather than leave it mute.
    text = text.replace("<->", " e viceversa ").replace("->", " verso ")
    text = text.replace("<=", " minore o uguale a ").replace(">=", " maggiore o uguale a ")
    text = text.replace("<", " minore di ").replace(">", " maggiore di ")
    text = text.replace("|", " ").replace("\\", " ")
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
NEEDS_LLM_RE = re.compile(
    r"[Ͱ-Ͽ∀-⋿←-⇿" + re.escape(SYM_OPEN + SYM_CLOSE + _SUB + _SUP) + r"`|\\{}<>=]"
    r"|```|^\s{0,3}#{1,6}\s|^\s{0,3}[*+-]\s|\*\*|\[[^\]]+\]\("
    r"|(?<!\w)[A-Za-z]_\w|https?://|\\n",
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


def sanitize_chunk(
    chunk: str,
    index: int,
    title: str = "",
    previous_text: str = "",
    target_language: str = "Italian",
) -> str:
    """Rewrite one block for speech. Falls back to the scrub alone if the
    model misbehaves: a mechanically pronounced block still carries the
    author's content, whereas a refusal does not."""
    if is_effectively_empty(chunk):
        return ""

    if not NEEDS_LLM_RE.search(chunk):
        return scrub(chunk)

    expected = VIS_RE.findall(chunk)
    prompt = Template(SANITIZER_PROMPT_PATH.read_text()).render(
        SOURCE_TEXT=chunk,
        TARGET_LANGUAGE=target_language,
        ARTICLE_TITLE=title,
        PREVIOUS_TEXT=previous_text[-CONTEXT_CHARS:],
    )

    try:
        sanitized, elapsed = generate_response(prompt)
        logger.info("Sanitize chunk %d completed in %.2f seconds", index, elapsed)
        check_usable(sanitized, "sanitize", index)

        found = VIS_RE.findall(sanitized)
        if found != expected:
            raise ModelRefusedError(
                f"sanitize block {index}: visual placeholders changed, "
                f"expected {expected}, found {found}"
            )
    except (ModelRefusedError, RuntimeError) as exc:
        logger.warning(
            "sanitize block %d: LLM pass unusable (%s); falling back to scrub only", index, exc
        )
        return scrub(chunk)

    return scrub(sanitized)


def _sanitize_block(index: int, block, title: str, previous_text: str) -> None:
    block.speech_text = sanitize_chunk(block.translated_text, index, title, previous_text)
    save_speech_shard(title, index, block.speech_text)


def sanitize_blocks(document, title: str, max_workers: int = 8):
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
                lambda item: _sanitize_block(item[0], item[1], title, previous[item[0]]),
                todo,
            )
        )
    return document
