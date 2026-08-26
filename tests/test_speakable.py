"""The allowlist: what a speech engine can be handed, decided by exclusion.

Nothing in this module knows a language, so most of what matters is that the
same rule gives the right answer in scripts nobody wrote it for.
"""

import pytest

from feedbard.pipeline.speakable import (
    describe_unspeakable,
    foreign_letters,
    needs_llm,
    strip_unspeakable,
    unspeakable_chars,
)


@pytest.mark.parametrize(
    "text",
    [
        "Il modello di ragionamento produce output migliori, ma più lenti.",
        "The reasoning model produces better output, but slower.",
        "推論モデルは、より良い出力を、より遅く生成する。",
        "Модель рассуждения даёт лучший результат, но медленнее.",
        "النموذج ينتج مخرجات أفضل، لكنه أبطأ.",
        "तर्क मॉडल बेहतर परिणाम देता है।",
        "Il modello raggiunge il 92 per cento (una misura alta): notevole!",
    ],
)
def test_prose_in_any_script_is_already_speech(text):
    assert not needs_llm(text)
    assert unspeakable_chars(text) == []


@pytest.mark.parametrize(
    "text, offender",
    [
        ("il codice è su https://github.com/rasbt/evals", "/"),
        ("usa il flag --model per scegliere", None),
        ("il parametro reasoning_effort è attivo", "_"),
        ("una riga ───█ di arte ascii", "─"),
        ("la soglia è 30° gradi", "°"),
        ("il tasso è ≤ 0,5", "≤"),
        ("scrivi {'chiave': 'valore'} nel file", "{"),
        ("ottimo lavoro 🎉 davvero", "🎉"),
        ("il titolo ## Sezione due", "#"),
        ("accuratezza del 92%", "%"),
    ],
)
def test_anything_that_is_not_text_is_caught(text, offender):
    assert needs_llm(text)
    if offender is not None:
        assert offender in unspeakable_chars(text)


def test_a_foreign_letter_is_notation_but_its_own_script_is_prose():
    # `σ` in Italian prose is a variable and has to be spelled out; the same
    # `σ` in Greek prose is a word. Dominance decides, so no alphabet is named.
    assert needs_llm("il tasso σ è piccolo")
    assert foreign_letters("il tasso σ è piccolo") == ["σ"]
    assert not needs_llm("Ο συντελεστής σ είναι μικρός")


@pytest.mark.parametrize(
    "text",
    [
        "DeepSeek 4.5 è uscito",
        "GPT-4.5 ha vinto",
        # Glued straight on, which is how most model names are written. The
        # rule used to demand a space or a hyphen and missed every one of
        # these -- 114 blocks of the shipped corpus.
        "il modello V3.2 è nuovo",
        "Qwen3.5 batte K2.5",
        "MiniMax M2.1 e GLM-4.7",
    ],
)
def test_a_versioned_name_is_caught_although_every_character_is_allowed(text):
    # "4.5" read as a decimal is a different number, and in a locale where the
    # dot separates thousands it is a very different one.
    assert unspeakable_chars(text) == []
    assert needs_llm(text)


@pytest.mark.parametrize(
    "text",
    [
        "l'accuratezza è salita al 95.3 per cento",
        "circa 3.5 milioni di token",
        "il valore 0.5 resta invariato",
    ],
)
def test_a_plain_decimal_is_left_alone(text):
    # The capitalization of the name is what separates a version from a
    # metric: with the separator optional, "al 95.3" would otherwise match.
    assert not needs_llm(text)


def test_the_reason_is_reported_for_the_prompt_and_the_log():
    reason = describe_unspeakable("il codice https://x.com/a e σ con DeepSeek 4.5")
    assert "SOLIDUS" in reason and "σ" in reason and "version" in reason


# --- last resort ------------------------------------------------------------


def test_strip_drops_whole_tokens_not_single_characters():
    # Deleting only the offending characters would leave
    # "httpsgithubcomrasbtevals" for the voice to attempt.
    out = strip_unspeakable("il codice è qui: https://github.com/rasbt/evals per i dettagli")
    assert out == "il codice è qui: per i dettagli"
    assert "github" not in out


def test_strip_keeps_the_content_of_the_pipelines_own_markers():
    # ⟪⟫ is markup this pipeline added, not something an author wrote.
    assert strip_unspeakable("usa ⟪ollama⟫ per iniziare") == "usa ollama per iniziare"


def test_strip_output_is_speakable_and_idempotent():
    for text in ["una riga ───█ di arte", "curl https://x.sh | bash", "```json\n{}\n```"]:
        once = strip_unspeakable(text)
        assert not unspeakable_chars(once), once
        assert strip_unspeakable(once) == once
