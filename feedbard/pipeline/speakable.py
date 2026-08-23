"""What a speech engine can pronounce, decided by exclusion.

The pipeline used to answer this with a denylist: a table of glyphs that were
known to break the voice, each with a hand-written spoken form per language.
That table could only ever cover what someone had already thought of, and it
pinned the whole pipeline to the languages the table had been written for.

This is the inverse, and it is why nothing here names a language. Text is what
a writing system is made of -- letters, the marks that modify them, digits,
spaces, and punctuation. Anything else is not text: it is notation, markup,
an address, an emoji, a drawing made of box characters. The pipeline does not
need to know what those *mean* in order to know they cannot be read aloud, and
whatever they mean is decided later, in context, by the model.

The punctuation set is derived rather than listed. A character is punctuation
if the Unicode database says its name is one of the punctuation concepts
below, so `,` `、` `،` are all "comma" and every script's version arrives
without anyone adding it. Listing the characters instead would have quietly
meant "Latin".
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import cache

from feedbard.ingestion.html_parser import SYM_CLOSE, SYM_OPEN

# Letters (every script), the combining marks that Arabic, Devanagari and
# Vietnamese are unreadable without, decimal digits, and spaces.
ALLOWED_CATEGORIES = frozenset({"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Zs"})

# Punctuation by concept, matched against the character's Unicode name. These
# are English words because the Unicode database is written in English; they
# describe the writing system, not the article's language.
#
# Deliberately absent, and therefore treated as notation: NUMBER SIGN (`#`),
# ASTERISK, SOLIDUS (`/`), REVERSE SOLIDUS, PERCENT, AMPERSAND, COMMERCIAL AT,
# LOW LINE (`_`), and the square and curly brackets. Every one of them shows
# up in code, in a path, or in markup far more often than in prose.
PUNCTUATION_CONCEPTS = (
    "FULL STOP",
    "COMMA",
    "QUESTION MARK",
    "EXCLAMATION MARK",
    "SEMICOLON",
    "COLON",
    "APOSTROPHE",
    "QUOTATION MARK",
    "CORNER BRACKET",
    "PARENTHESIS",
    "HYPHEN",
    "DASH",
    "ELLIPSIS",
    "DANDA",
)

# Line structure carries prosody and is not a character the engine reads.
ALLOWED_CONTROLS = frozenset("\n\r\t")

# Characters that occupy no space and make no sound: zero-width joiners, soft
# hyphens, bidirectional overrides, stray control codes. They are deleted
# rather than sent to the model, because there is nothing for a model to
# decide -- they say nothing, in any language. Left in, they make a block look
# non-empty when it is blank, and split a word so the voice reads two.
INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})


def remove_invisible(text: str) -> str:
    return "".join(
        c
        for c in text or ""
        if c in ALLOWED_CONTROLS or unicodedata.category(c) not in INVISIBLE_CATEGORIES
    )


def is_effectively_empty(text: str) -> bool:
    """Nothing here would make a sound. Sending it to a model gets a helpful
    reply about the input being empty, which then gets narrated."""
    return not remove_invisible(text).strip()

# The parser's own inline-code delimiters. They are markup this pipeline put
# there, not something an author wrote, so they are stripped before the
# last-resort scrub rather than dragging their contents out with them.
OWN_MARKUP = (SYM_OPEN, SYM_CLOSE)

# Two or more dashes in a row: a command-line flag (`--model`), a rule drawn
# out of hyphens, an arrow typed as `-->`. Each individual dash is ordinary
# punctuation and passes the allowlist, so only the run gives them away.
DASH_RUN_RE = re.compile(r"[\u2010-\u2015\-]{2,}")

# A name followed by a dotted version number: `DeepSeek 4.5`, `GPT-4.5`.
# Every character in it is allowed, so only this rule routes it to the model.
# Read as a decimal it becomes a different number -- and in a locale where the
# dot is the thousands separator, a very different one. The capitalization of
# the name is what separates it from a plain metric ("salita al 95.3"), and it
# is tested with `str.isupper` rather than an `[A-Z]` class so that it holds
# for any bicameral script.
VERSIONED_NAME_RE = re.compile(r"(?<!\w)([^\W\d_][^\W\d_]*)[ -](\d+\.\d+)(?!\d)")


@cache
def is_speakable_char(char: str) -> bool:
    if char in ALLOWED_CONTROLS:
        return True
    if unicodedata.category(char) in ALLOWED_CATEGORIES:
        return True
    name = unicodedata.name(char, "")
    return any(concept in name for concept in PUNCTUATION_CONCEPTS)


def unspeakable_chars(text: str) -> list[str]:
    """Every distinct character in `text` a voice cannot read, in order."""
    seen: dict[str, None] = {}
    for char in text:
        if not is_speakable_char(char):
            seen.setdefault(char, None)
    return list(seen)


def has_versioned_name(text: str) -> bool:
    return any(m.group(1)[0].isupper() for m in VERSIONED_NAME_RE.finditer(text))


@cache
def script_of(char: str) -> str:
    """The script a character belongs to, taken from its Unicode name.

    `LATIN SMALL LETTER A` -> `LATIN`, `GREEK SMALL LETTER SIGMA` -> `GREEK`.
    From the Unicode database rather than a list of alphabets, so a script
    nobody here anticipated still classifies.
    """
    return unicodedata.name(char, "").split(None, 1)[0]


def foreign_letters(text: str) -> list[str]:
    """Single letters standing alone in prose written in another script.

    A letter is a text character, so the allowlist lets it through -- but a
    lone `σ` in Italian prose is a maths variable, not a word, and the engine
    will either mispronounce it or drop it silently. What makes it notable is
    not that it is Greek: it is that it stands alone *and* is foreign to this
    passage. The same `σ` inside Greek prose is an ordinary word.

    Standing alone is what keeps this from firing on ordinary writing. Japanese
    mixes three scripts inside a single sentence, so "a letter from a
    non-dominant script" would flag every Japanese article ever written; a
    one-character word is a variable in a way that a character inside a word
    is not.
    """
    scripts = Counter(script_of(c) for c in text if c.isalpha())
    if not scripts:
        return []
    dominant = scripts.most_common(1)[0][0]

    seen: dict[str, None] = {}
    for token in text.split():
        stripped = "".join(c for c in token if c.isalpha())
        if len(stripped) == 1 and script_of(stripped) != dominant:
            seen.setdefault(stripped, None)
    return list(seen)


def has_dash_run(text: str) -> bool:
    return bool(DASH_RUN_RE.search(text))


def needs_llm(text: str) -> bool:
    """Whether this passage has anything a speech engine cannot be handed.

    True means one LLM call. On a corpus of real translated blocks this fires
    on about one block in five, so the common case stays free.
    """
    return (
        bool(unspeakable_chars(text))
        or bool(foreign_letters(text))
        or has_dash_run(text)
        or has_versioned_name(text)
    )


def describe_unspeakable(text: str) -> str:
    """Why `needs_llm` fired, in words, for a log line and for the prompt.

    The model is told what tripped the check rather than left to find it: on a
    retry that is the only new information there is to give it.
    """
    parts = []
    chars = unspeakable_chars(text)
    if chars:
        parts.append(
            "characters that are not letters, digits, spaces or punctuation: "
            + ", ".join(f"{c} ({unicodedata.name(c, 'unnamed')})" for c in chars)
        )
    foreign = foreign_letters(text)
    if foreign:
        parts.append(
            "letters from a different script than the surrounding prose, which are notation "
            "rather than words here: " + ", ".join(foreign)
        )
    if has_dash_run(text):
        parts.append(
            "a run of two or more dashes, which is a flag or a drawn rule, not punctuation"
        )
    if has_versioned_name(text):
        parts.append("a name followed by a dotted version number, which must be read as a version")
    return "; ".join(parts)


def strip_unspeakable(text: str) -> str:
    """Last resort: drop what could not be turned into words.

    Whole whitespace-delimited tokens go, not individual characters. Deleting
    only the offending character out of `https://github.com/rasbt/evals` would
    leave `httpsgithubcomrasbtevals` for the voice to attempt; deleting the
    token leaves a gap, which is honest. The same reasoning covers a fragment
    of code, a box-drawing rule, and a stray emoji.

    Reached only when the model has already been asked twice and its answer
    still was not speakable, so this is a bad outcome being made survivable,
    never the normal path.
    """
    text = remove_invisible(text)
    for marker in OWN_MARKUP:
        text = text.replace(marker, " ")

    def drop_token(match: re.Match) -> str:
        token = match.group(0)
        return "" if unspeakable_chars(token) or has_dash_run(token) else token

    text = re.sub(r"\S+", drop_token, text)

    # The deletions leave doubled spaces, spaces before punctuation, and
    # sentences that now open or close on a stranded mark.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r"([.,;:])[ \t]*\1+", r"\1", text)
    text = re.sub(r"^[ \t]*[,;:.\-]+[ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
