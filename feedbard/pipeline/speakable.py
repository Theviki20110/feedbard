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

from babel import Locale
from babel.numbers import get_decimal_symbol, get_group_symbol

from feedbard.ingestion.html_parser import SYM_CLOSE, SYM_OPEN
from feedbard.language import TARGET_LANGUAGE, language_code

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


# What a word looks like. Letters, and the apostrophe or hyphen that joins two
# of them -- `l'accuratezza`, `e-mail`. Joining, not leading or trailing: a
# token that opens on a dash is a command-line flag, not a word.
WORD_JOINERS = "'\u2019\u02bc-\u2010\u2011\u2012\u2013\u2014\u2015"


@cache
def _number_re(language: str) -> re.Pattern:
    """How `language` writes a number, according to CLDR.

    Asked rather than assumed: Italian writes the decimal with a comma and
    Italian prose is where `4.5` gives itself away as a version rather than a
    quantity, while in English the same `4.5` is an ordinary decimal and
    reading it as one is correct. Hard-coding either answer would have made
    the rule right in one language and wrong in the next.
    """
    try:
        locale = Locale.parse(language_code(language))
        decimal = str(get_decimal_symbol(locale))
        group = str(get_group_symbol(locale))
    except Exception:  # noqa: BLE001 - an unknown locale is not worth a crash
        decimal, group = ".", ","
    d, g = re.escape(decimal), re.escape(group)
    return re.compile(rf"^[+-]?(?:\d+|\d{{1,3}}(?:{g}\d{{3}})+)(?:{d}\d+)?$")


def is_word(core: str) -> bool:
    """A run of letters, with the punctuation that can sit inside one.

    Internal punctuation is allowed because a word can legitimately carry it
    -- `l'accuratezza`, `e-mail` -- and because a script that writes without
    spaces puts its commas inside what whitespace hands us as a single token.
    A Japanese clause arrives here whole, and it is a sequence of words.

    The edges are what the rule turns on: a token that opens on punctuation is
    a flag (`--model`), not a word.
    """
    if not core or not (core[0].isalpha() and core[-1].isalpha()):
        return False
    return all(
        c.isalpha() or unicodedata.category(c).startswith("M") or is_speakable_char(c) for c in core
    )


def is_number(core: str, language: str) -> bool:
    return bool(_number_re(language).match(core))


@cache
def is_speakable_char(char: str) -> bool:
    if char in ALLOWED_CONTROLS:
        return True
    if unicodedata.category(char) in ALLOWED_CATEGORIES:
        return True
    name = unicodedata.name(char, "")
    return any(concept in name for concept in PUNCTUATION_CONCEPTS)


# Punctuation that can sit on either side of a token without being part of it:
# quotes, brackets, and the marks that end a sentence. Derived from the same
# concepts as the character rule, so it covers every script's forms -- the
# whole basic plane is scanned, because CJK punctuation lives above U+3000 and
# a shorter scan silently left every Japanese sentence looking like notation.
#
# Dashes are excluded on purpose: they are never stripped from an edge, so a
# token that opens on one stays recognizable as a flag rather than a word.
EDGE_PUNCTUATION = "".join(
    chr(c)
    for c in range(0x10000)
    if is_speakable_char(chr(c)) and not chr(c).isalnum() and unicodedata.category(chr(c)) != "Pd"
)


def unspeakable_chars(text: str) -> list[str]:
    """Every distinct character in `text` a voice cannot read, in order."""
    seen: dict[str, None] = {}
    for char in text:
        if not is_speakable_char(char):
            seen.setdefault(char, None)
    return list(seen)


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


def unspeakable_tokens(text: str, language: str = TARGET_LANGUAGE) -> list[str]:
    """The tokens in `text` that are neither a word, a number, nor punctuation.

    This is the whole rule. `30B`, `5:1`, `20-40`, `Qwen3.5`, `--model`,
    `AGENTS.md`, `https://github.com/x`, `≤`, `█` are all caught by it, and
    none of them is named anywhere: each simply fails to be one of the three
    things speech is made of.

    It replaced two hand-written patterns -- one for a run of dashes, one for
    a name carrying a version number -- that between them caught two cases and
    missed every other. Patterns like those accumulate: the denylist this
    module exists to avoid grew exactly that way, one observed failure at a
    time.
    """
    found: dict[str, None] = {}
    for token in text.split():
        core = token.strip(EDGE_PUNCTUATION)
        # A token of nothing but dashes is an em dash doing punctuation's job.
        # Dashes are never stripped from an edge, so `--model` keeps the pair
        # that gives it away as a flag rather than a word.
        if not core or core.strip(WORD_JOINERS) == "":
            continue
        # A number the way this language writes numbers is already speech.
        if is_number(core, language):
            continue
        # A digit touching anything else is notation: `30B` is a magnitude,
        # `5:1` a ratio, `20-40` a range, `Qwen3.5` a version. None of them is
        # named here -- what they have in common is that a digit is joined to
        # something that is not part of a number in this language.
        if any(c.isdigit() for c in core):
            found.setdefault(core, None)
            continue
        if not is_word(core):
            found.setdefault(core, None)
    return list(found)


def needs_llm(text: str, language: str = TARGET_LANGUAGE) -> bool:
    """Whether this passage has anything a speech engine cannot be handed.

    Two questions, and both are asked of the writing system rather than of a
    list: is every token a word, a number or punctuation, and is every letter
    one that belongs to the prose around it.
    """
    return bool(unspeakable_tokens(text, language)) or bool(foreign_letters(text))


def describe_unspeakable(text: str, language: str = TARGET_LANGUAGE) -> str:
    """Why `needs_llm` fired, in words, for a log line and for the prompt.

    The model is told what tripped the check rather than left to find it: on a
    retry that is the only new information there is to give it.
    """
    parts = []
    tokens = unspeakable_tokens(text, language)
    if tokens:
        parts.append(
            "tokens that are neither a word nor a number: " + ", ".join(repr(t) for t in tokens)
        )
        chars = unspeakable_chars(" ".join(tokens))
        if chars:
            parts.append(
                "including characters that are not letters, digits or punctuation: "
                + ", ".join(f"{c} ({unicodedata.name(c, 'unnamed')})" for c in chars)
            )
    foreign = foreign_letters(text)
    if foreign:
        parts.append(
            "letters from a different script than the surrounding prose, which are notation "
            "rather than words here: " + ", ".join(foreign)
        )
    return "; ".join(parts)


def strip_unspeakable(text: str, language: str = TARGET_LANGUAGE) -> str:
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
        return "" if unspeakable_tokens(token, language) else token

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
