"""Spoken forms for the symbols a speech engine cannot pronounce, per language.

The deterministic scrub in `sanitizer` has to replace a glyph with words, and
the words depend on the language the episode is narrated in: `σ` is "sigma" in
Italian and English but "sigma" in neither Greek nor Japanese, and `≤` is a
different phrase in every one of them. Keeping the table in code would pin the
whole pipeline to a single language, so it lives in `assets/lexicons/` instead,
one JSON file per language, added without touching this module.

A lexicon file has two sections:

* `symbols` -- single glyphs, replaced anywhere they appear.
* `sequences` -- ordered multi-character replacements (`<=` before `<`, so the
  compound comparison is not read as two separate ones). Order is significant
  and is taken from the file, so JSON list-of-pairs rather than an object.

Anything language-independent (`{`, `|`, bullet glyphs) stays in the scrub
itself: it is deleted, not spoken, so there is nothing to translate.
"""

import json
import os
from dataclasses import dataclass
from functools import cache

from dotenv import load_dotenv

from feedbard.logger import logger
from feedbard.paths import ASSETS_DIR

load_dotenv()

# The language every stage narrates in, unless a caller says otherwise.
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "Italian")

LEXICONS_DIR = ASSETS_DIR / "lexicons"

# Used when the requested language has no file yet. English words in a Swedish
# episode are wrong, but they are wrong and audible: dropping the symbol
# silently would delete a comparison or an operator from a claim, and the
# listener would never know a term went missing.
FALLBACK_LANGUAGE = "english"


@dataclass(frozen=True)
class Lexicon:
    language: str
    symbols: dict[str, str]
    sequences: tuple[tuple[str, str], ...]


def _path_for(language: str):
    return LEXICONS_DIR / f"{language.strip().lower()}.json"


def available_languages() -> list[str]:
    return sorted(p.stem for p in LEXICONS_DIR.glob("*.json"))


@cache
def load_lexicon(language: str = TARGET_LANGUAGE) -> Lexicon:
    """Load the spoken-form table for `language`, falling back to English.

    Cached per language, which also means the fallback warning is logged once
    per run rather than once per block.
    """
    path = _path_for(language)
    if not path.exists():
        logger.warning(
            "lexicon: no spoken-form table for %r (looked for %s); falling back to %s. "
            "Available: %s",
            language,
            path,
            FALLBACK_LANGUAGE,
            ", ".join(available_languages()) or "none",
        )
        path = _path_for(FALLBACK_LANGUAGE)

    data = json.loads(path.read_text(encoding="utf-8"))
    return Lexicon(
        language=path.stem,
        symbols=dict(data.get("symbols", {})),
        sequences=tuple((frm, to) for frm, to in data.get("sequences", [])),
    )
