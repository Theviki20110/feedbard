import pytest

from feedbard.pipeline import sanitizer
from feedbard.pipeline.lexicon import load_lexicon
from feedbard.pipeline.sanitizer import (
    ModelRefusedError,
    check_usable,
    is_effectively_empty,
    mark_urls,
    repair_is_acceptable,
    sanitize_chunk,
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


# --- URLs ------------------------------------------------------------------

# Every one of these reached the speech engine and was spelled out aloud.
URL_SHARDS = [
    "Uno gennaio, Deep Delta Learning, https://arxiv.org/abs/2601.00417",
    "il codice è disponibile qui: https://github.com/rasbt/local-coding-agent-evals",
    "Ollama fornisce un'integrazione: https://docs.ollama.com/integrations/claude-code.",
    "se hai clonato gli script da https://github.com/rasbt/x, esegui quanto segue",
    "vedi la documentazione (https://example.com/docs) per i dettagli",
    "il sito www.example.com riporta i benchmark",
]


@pytest.mark.parametrize("text", URL_SHARDS)
def test_urls_are_removed(text):
    out = scrub(text)
    assert "http" not in out and "www." not in out and "://" not in out
    assert ".com" not in out and ".org" not in out


def test_url_removal_keeps_the_sentence_readable():
    out = scrub("il codice è disponibile qui: https://github.com/rasbt/evals")
    assert out == "il codice è disponibile qui."
    assert "  " not in out


def test_url_removal_keeps_surrounding_content():
    out = scrub("Uno gennaio, Deep Delta Learning, https://arxiv.org/abs/2601.00417")
    assert "Deep Delta Learning" in out


def test_markdown_link_label_survives():
    assert scrub("vedi [il repository](https://github.com/x/y) per i dettagli") == (
        "vedi il repository per i dettagli"
    )


@pytest.mark.parametrize("text", URL_SHARDS)
def test_url_removal_is_idempotent(text):
    once = scrub(text)
    assert scrub(once) == once


# --- Language ---------------------------------------------------------------


def test_equals_sign_is_spoken():
    # `=` used to survive the scrub and reach the engine mute.
    assert "=" not in scrub("utilizza enable_thinking=True")
    assert "uguale a" in scrub("utilizza enable_thinking=True")


def test_spoken_forms_follow_the_requested_language():
    text = "I pesi σ_1 con θ, e a <= b"
    it, en = scrub(text, "Italian"), scrub(text, "English")
    assert "minore o uguale a" in it and "less than or equal to" in en
    assert "alfa" in scrub("α", "Italian") and "alpha" in scrub("α", "English")


def test_unknown_language_falls_back_without_crashing():
    out = scrub("a <= b con σ", "Klingon")
    assert not (set(out) & UNSPEAKABLE)


def test_lexicons_cover_the_same_symbols():
    it, en = load_lexicon("Italian"), load_lexicon("English")
    assert set(it.symbols) == set(en.symbols)
    assert [f for f, _ in it.sequences] == [f for f, _ in en.sequences]


# --- Link markers, the LLM's view of a URL ----------------------------------


def test_mark_urls_keeps_the_host_and_drops_the_path():
    out = mark_urls("il codice è qui: https://github.com/rasbt/evals")
    assert "⟪link: github.com⟫" in out
    assert "rasbt" not in out and "https" not in out


def test_mark_urls_normalises_www_and_schemes():
    assert "⟪link: example.com⟫" in mark_urls("vedi www.example.com per i dettagli")
    assert "⟪link: arxiv.org⟫" in mark_urls("paper: https://arxiv.org/abs/2601.00417")


def test_mark_urls_leaves_prose_alone():
    text = "Il modello raggiunge il 92% di accuratezza."
    assert mark_urls(text) == text


def test_scrub_removes_a_marker_the_model_ignored():
    out = scrub("il codice è disponibile qui: ⟪link: github.com⟫")
    assert "link" not in out and "github" not in out
    assert out == "il codice è disponibile qui."


# --- Repair pass ------------------------------------------------------------

SCRUBBED = "se hai clonato gli script da, possiamo eseguire quanto segue."


@pytest.mark.parametrize(
    "repaired,reason",
    [
        ("", "empty output"),
        ("Non posso aiutarti con questa richiesta.", "commentary"),
        ("se hai clonato gli script, esegui questo: https://github.com/x/y", "link"),
        (SCRUBBED + " " + "Aggiungo un intero paragrafo di contesto inventato qui.", "grew"),
        ("se hai clonato gli script.", "shrank"),
    ],
)
def test_repair_guards_reject_a_rewrite(repaired, reason):
    assert repair_is_acceptable(SCRUBBED, repaired) is not None


def test_repair_guard_accepts_a_mend():
    mended = "se hai clonato gli script, possiamo eseguire quanto segue."
    assert repair_is_acceptable(SCRUBBED, mended) is None


def test_repair_guard_rejects_dropped_visual_placeholder():
    before = "Come mostrato ⟦VIS:0123456789⟧ nel grafico, il valore sale."
    assert repair_is_acceptable(before, "Come mostrato nel grafico, il valore sale ancora.")


def _fake_llm(reply):
    return lambda prompt: (reply, 0.0)


def test_fallback_path_repairs_a_link_hole(monkeypatch):
    # The sanitize call fails, so the block takes the scrub-only path; the
    # repair call then mends what the link removal left behind.
    calls = []

    def generate_response(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("bedrock unavailable")
        return "se hai clonato gli script, possiamo eseguire quanto segue.", 0.0

    monkeypatch.setattr(sanitizer, "generate_response", generate_response)
    chunk = (
        "se hai clonato gli script da https://github.com/rasbt/evals, "
        "possiamo eseguire quanto segue."
    )
    out = sanitize_chunk(chunk, index=0)
    assert out == "se hai clonato gli script, possiamo eseguire quanto segue."
    assert len(calls) == 2


def test_fallback_keeps_scrubbed_text_when_repair_misbehaves(monkeypatch):
    def generate_response(prompt):
        if "repair" in prompt.lower():
            return "Certo! Ecco il testo corretto: vedi https://github.com/x/y", 0.0
        raise RuntimeError("bedrock unavailable")

    monkeypatch.setattr(sanitizer, "generate_response", generate_response)
    out = sanitize_chunk("il codice è qui: https://github.com/rasbt/evals", index=0)
    assert "http" not in out and "github" not in out
    assert out == "il codice è qui."


def test_no_repair_call_without_links(monkeypatch):
    monkeypatch.setattr(sanitizer, "generate_response", _fake_llm("mai chiamato"))
    calls = []

    def generate_response(prompt):
        calls.append(prompt)
        raise RuntimeError("bedrock unavailable")

    monkeypatch.setattr(sanitizer, "generate_response", generate_response)
    out = sanitize_chunk("il parametro enable_thinking=True è attivo", index=0)
    assert "uguale a" in out
    assert len(calls) == 1  # sanitize only, no repair
