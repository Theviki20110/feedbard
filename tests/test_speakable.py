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
    unspeakable_tokens,
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
    "text, token",
    [
        # A version, glued to its name or not. Read as a quantity it is a
        # different number, and where the dot groups thousands, a wrong one.
        ("DeepSeek 4.5 è uscito", "4.5"),
        ("Qwen3.5 batte tutti", "Qwen3.5"),
        ("il modello V3.2 è nuovo", "V3.2"),
        # A magnitude: the letter is a word, not a letter.
        ("un modello da 30B parametri", "30B"),
        ("contesto da 32k token", "32k"),
        # A relation between two numbers, whatever joins them.
        ("il rapporto resta di 5:1", "5:1"),
        ("servono tra 20-40 epoche", "20-40"),
        # A flag: it opens on punctuation, so it is not a word.
        ("usa il flag --model per scegliere", "--model"),
    ],
)
def test_a_token_that_is_neither_a_word_nor_a_number_is_notation(text, token):
    # None of these is named in the code. Each one simply fails to be one of
    # the three things speech is made of.
    assert needs_llm(text)
    assert token in unspeakable_tokens(text)


@pytest.mark.parametrize(
    "text",
    [
        # Written the way the language writes numbers, so already speech.
        "l'accuratezza è salita a 95,3 punti",
        "nel 2024 sono usciti molti modelli",
        "circa 3 milioni di token",
        # A word carrying the punctuation a word can carry.
        "una e-mail e l'accuratezza restano",
        "un inciso — come questo — nel mezzo",
    ],
)
def test_words_and_numbers_are_left_alone(text):
    assert not needs_llm(text)


def test_the_decimal_separator_comes_from_the_language_not_from_a_guess():
    # `4.5` is a decimal in English and reads correctly as one. In Italian the
    # dot groups thousands, so the same token is a version or an untranslated
    # number -- either way something only the model can resolve. Asking CLDR
    # gives the right answer in both without either being written down here.
    assert not needs_llm("the score rose to 4.5 points", "English")
    assert needs_llm("il punteggio sale a 4.5 punti", "Italian")


def test_the_reason_is_reported_for_the_prompt_and_the_log():
    reason = describe_unspeakable("il codice https://x.com/a e σ con 30B parametri")
    assert "https://x.com/a" in reason and "30B" in reason
    assert "SOLIDUS" in reason
    assert "σ" in reason


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
