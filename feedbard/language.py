"""One narration language, resolved once, in every form a stage needs it.

`TARGET_LANGUAGE` used to be repeated in four env vars that had to agree by
hand: the name the prompts interpolate ("Italian"), the short code the HTTP
TTS server and the ASR round-trip want ("it"), the BCP-47 tag Polly wants
("it-IT"). Nothing checked that they matched, so a half-updated `.env`
narrated one language with another's voice.

They are all derivable from a single value, so they are derived here.
`TARGET_LANGUAGE` accepts whatever an operator would naturally write -- an
English language name (`Italian`), a bare code (`it`), or a full tag
(`pt-BR`) -- and every other form is computed from it.

An unresolvable value is not fatal: the raw string still reaches the prompts,
which is the one consumer that can do something sensible with a name this
module does not recognize.
"""

import os

import langcodes
from dotenv import load_dotenv

from feedbard.logger import logger

load_dotenv()

# What the operator wrote. Kept verbatim because it is what every stage still
# passes around as `target_language`, and what the shards record, so changing
# its spelling here would invalidate every cached description on disk.
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "Italian")


def _resolve(value: str) -> langcodes.Language | None:
    """`Italian` / `it` / `it-IT` / `pt-BR` -> a Language, or None."""
    try:
        tag = langcodes.Language.get(value, normalize=True)
        if tag.is_valid():
            return tag
    except Exception:  # noqa: BLE001 - langcodes raises several unrelated types
        pass
    try:
        return langcodes.find(value)
    except LookupError:
        return None


_resolved = _resolve(TARGET_LANGUAGE)

if _resolved is None:
    logger.warning(
        "language: could not resolve TARGET_LANGUAGE=%r to a language tag; "
        "speech language codes fall back to the raw value and are probably wrong. "
        "Write an English language name (Italian), a code (it), or a tag (pt-BR).",
        TARGET_LANGUAGE,
    )
    LANGUAGE_NAME = TARGET_LANGUAGE
    LANGUAGE_CODE = TARGET_LANGUAGE
    LANGUAGE_TAG = TARGET_LANGUAGE
else:
    # The bare language, without the territory: `pt-BR` names the same
    # language as `pt`, and a prompt asking for "Portuguese (Brazil)" reads
    # as a different instruction than one asking for "Portuguese".
    _bare = langcodes.Language.make(language=_resolved.language)

    # Interpolated into every prompt. English because that is the language the
    # prompts themselves are written in.
    LANGUAGE_NAME = _bare.display_name("en")
    # HTTP TTS server and the ASR round-trip in audio_renderer.
    LANGUAGE_CODE = _resolved.language
    # Polly, which wants a territory even when the operator did not name one.
    LANGUAGE_TAG = langcodes.Language.make(
        language=_resolved.language,
        territory=_resolved.territory or _resolved.maximize().territory,
    ).to_tag()



def language_code(language: str) -> str:
    """The short code for any spelling of `language`, for libraries that want
    a locale rather than a name."""
    resolved = _resolve(language)
    return resolved.language if resolved is not None else language


def display_name(language: str) -> str:
    """The English name for any spelling of `language`, for prompt templates.

    Stages pass the operator's raw string around, because that is what keys
    the shards on disk. A prompt, though, has to read as an instruction: with
    `TARGET_LANGUAGE=it` the templates would otherwise ask the model to write
    "in it".
    """
    resolved = _resolve(language)
    if resolved is None:
        return language
    return langcodes.Language.make(language=resolved.language).display_name("en")
