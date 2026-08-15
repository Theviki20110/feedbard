import pytest

from substack_feed.pipeline.sanitizer import (
    ModelRefusedError,
    check_usable,
    is_effectively_empty,
    scrub,
)

# Every string below was taken from a shard that actually reached the speech
# engine, so these are regressions, not hypotheticals.
REAL_SHARDS = [
    "Per Claude tramite ⟪ollama launch claude⟫, la chiave è che ⟪ollama⟫ deve vedere",
    "Questi tag ⟪<think>⟫ e ⟪</think>⟫ sono elementi cosmetici",
    "Inizia con ⟪<|assistant|><think>⟫ quando il ragionamento è abilitato",
    'attraverso thinking: ⟪{"type": "disabled"}⟫ nell\'API ufficiale',
    "11. Mac <-> DGX",
    "OLLAMA_HOST=http://127.0.0.1:11434 \\nollama launch claude --model qwen3.6:35b",
    "# Avviso sul copyright\n\n1. **Brevi estratti** (pochi paragrafi)",
    '```json\n{\n  "privacy": true\n}\n```',
    "curl https://x.sh | bash",
    "➜  uv run bench.py --model north-mini",
]

UNSPEAKABLE = set("⟪⟫<>{}|\\`#➜") | set("αβγδθλμπσωΣΠ∇∂≤≥×÷→") | set("₀₁₂₃⁰¹²³")


@pytest.mark.parametrize("text", REAL_SHARDS)
def test_scrub_leaves_nothing_unspeakable(text):
    assert not (set(scrub(text)) & UNSPEAKABLE)


@pytest.mark.parametrize("text", REAL_SHARDS)
def test_scrub_is_idempotent(text):
    once = scrub(text)
    assert scrub(once) == once


def test_greek_and_subscripts_become_words():
    out = scrub("I pesi σ_1, σ_2 e σ_n con θ, x² e Σ_i")
    assert "sigma 1" in out and "sigma n" in out
    assert "theta" in out and "sommatoria" in out


def test_unicode_subscripts_are_spaced():
    # "sigma1" would be pronounced as one nonsense word.
    assert scrub("σ₁") == "sigma 1"


def test_snake_case_identifiers_survive():
    # The subscript rule must not fire inside `reasoning_effort`.
    assert "reasoning_effort" in scrub("il parametro reasoning_effort è attivo")


def test_plain_prose_is_untouched():
    text = "Il modello di ragionamento produce output migliori, ma più lenti."
    assert scrub(text) == text


def test_visual_placeholders_survive():
    assert "⟦VIS:0123456789⟧" in scrub("Come mostrato qui ⟦VIS:0123456789⟧ nel grafico.")


def test_invisible_only_block_is_empty():
    assert is_effectively_empty("​  \n")
    assert not is_effectively_empty("ciao")


@pytest.mark.parametrize(
    "refusal",
    [
        "I'm ready to translate. However, the source content appears to be empty.",
        "Non posso tradurre un intero libro protetto da copyright (528 pagine).",
        "I understand the instructions completely. Please provide the text.",
    ],
)
def test_refusals_are_rejected(refusal):
    with pytest.raises(ModelRefusedError):
        check_usable(refusal, "translate", 0)


def test_real_content_passes():
    check_usable("Il modello raggiunge il 92% di accuratezza sul benchmark.", "translate", 0)
