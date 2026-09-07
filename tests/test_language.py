"""TARGET_LANGUAGE is written once and every other form is derived from it.

These cover the derivation itself; that the derived values reach the TTS and
ASR calls is covered in test_audio_renderer.py.
"""

import pytest

from feedbard import language


@pytest.mark.parametrize("raw", ["Italian", "it", "it-IT"])
def test_every_spelling_of_a_language_resolves_to_the_same_forms(raw):
    assert language.display_name(raw) == "Italian"


@pytest.mark.parametrize(
    "raw, name, code, tag",
    [
        ("Japanese", "Japanese", "ja", "ja-JP"),
        ("de", "German", "de", "de-DE"),
        ("pt-BR", "Portuguese", "pt", "pt-BR"),
    ],
)
def test_a_language_yields_a_name_a_code_and_a_tag(raw, name, code, tag):
    resolved = language._resolve(raw)
    assert resolved is not None
    assert language.display_name(raw) == name
    assert resolved.language == code


def test_a_regional_tag_keeps_its_territory_but_not_in_its_name():
    # Polly wants pt-BR; a prompt asking for "Portuguese (Brazil)" reads as a
    # different instruction than one asking for "Portuguese".
    assert language.display_name("pt-BR") == "Portuguese"
    assert language._resolve("pt-BR").territory == "BR"


def test_a_language_with_no_territory_still_gets_a_tag():
    # Polly requires one even when the operator did not write one.
    resolved = language._resolve("it")
    assert resolved.territory is None
    assert resolved.maximize().territory == "IT"


def test_an_unresolvable_value_falls_through_to_the_prompts_unchanged():
    # Not fatal: the prompt is the one consumer that can do something with a
    # name this module does not recognize.
    assert language._resolve("Frobnistan") is None
    assert language.display_name("Frobnistan") == "Frobnistan"
