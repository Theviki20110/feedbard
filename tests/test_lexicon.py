import logging

import pytest

from feedbard.pipeline import lexicon
from feedbard.pipeline.lexicon import Lexicon, available_languages, load_lexicon


def test_available_languages_lists_the_shipped_lexicons():
    assert {"italian", "english"} <= set(available_languages())


def test_load_lexicon_reads_symbols_and_sequences():
    lex = load_lexicon("Italian")
    assert isinstance(lex, Lexicon)
    assert lex.language == "italian"
    assert lex.symbols["σ"] == "sigma"
    assert ("<=", " minore o uguale a ") in lex.sequences


def test_load_lexicon_is_case_and_whitespace_insensitive():
    assert load_lexicon("Italian") == load_lexicon(" italian ")


def test_load_lexicon_is_cached_per_exact_argument():
    # functools.cache keys on the literal argument, so this only holds for two
    # calls with the identical string -- see the case-insensitivity test above
    # for why two different spellings are still equal in content.
    assert load_lexicon("Italian") is load_lexicon("Italian")


def test_sequences_preserve_file_order():
    # "<=" has to be replaced before "<", or the compound comparison would be
    # read as two separate symbols.
    lex = load_lexicon("Italian")
    froms = [frm for frm, _ in lex.sequences]
    assert froms.index("<=") < froms.index("<")


def test_unknown_language_falls_back_to_english(caplog):
    with caplog.at_level(logging.WARNING):
        lex = load_lexicon("Klingon")
    assert lex.language == lexicon.FALLBACK_LANGUAGE
    assert lex.symbols == load_lexicon("English").symbols
    assert "Klingon" in caplog.text


def test_fallback_warning_is_not_repeated_for_a_cached_language(caplog):
    load_lexicon.cache_clear()
    with caplog.at_level(logging.WARNING):
        load_lexicon("Klingon")
        load_lexicon("Klingon")
    assert caplog.text.count("no spoken-form table") == 1
    load_lexicon.cache_clear()


@pytest.mark.parametrize("language", ["Italian", "English"])
def test_every_shipped_lexicon_is_well_formed(language):
    lex = load_lexicon(language)
    assert lex.symbols
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in lex.symbols.items())
    assert all(len(pair) == 2 for pair in lex.sequences)
