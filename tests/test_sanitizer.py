"""The speech pass: check, ask, ask again, then delete what is left.

Only blocks that fail the allowlist reach the model at all, so the cheapest
thing this stage does is nothing. What matters here is the loop around the
call: that a clean answer is taken, a dirty one is challenged once, and a
model that never produces speakable text cannot put a symbol on the wire.
"""

import logging

import pytest

from feedbard.ingestion.html_parser import Block, Document, Kind
from feedbard.pipeline import sanitizer
from feedbard.pipeline.sanitizer import is_effectively_empty, sanitize_blocks, sanitize_chunk
from feedbard.pipeline.speakable import unspeakable_chars

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


@pytest.fixture
def shards(monkeypatch, tmp_path):
    monkeypatch.setattr(sanitizer, "SPEECH_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(sanitizer, "speech_shard_path", lambda t, i: tmp_path / f"{t}_{i}.txt")


def _model(*replies):
    """A model returning `replies` in order, recording the prompts it saw."""
    calls, it = [], iter(replies)

    def generate_response(prompt):
        calls.append(prompt)
        return next(it), 0.0

    generate_response.calls = calls
    return generate_response


# --- the cheap path ---------------------------------------------------------


def test_prose_that_is_already_speech_never_reaches_the_model(monkeypatch):
    monkeypatch.setattr(sanitizer, "generate_response", _model())
    text = "Il modello raggiunge il 92 per cento di accuratezza sul benchmark."
    assert sanitize_chunk(text, 0) == text


def test_an_empty_block_never_reaches_the_model(monkeypatch):
    monkeypatch.setattr(sanitizer, "generate_response", _model())
    assert sanitize_chunk("​  \n", 0) == ""
    assert is_effectively_empty("​  \n")
    assert not is_effectively_empty("ciao")


def test_invisible_characters_are_deleted_rather_than_described(monkeypatch):
    # Nothing for a model to decide: they make no sound in any language.
    monkeypatch.setattr(sanitizer, "generate_response", _model())
    assert sanitize_chunk("il mo­dello ​funziona", 0) == "il modello funziona"


# --- the loop ---------------------------------------------------------------


@pytest.mark.parametrize("text", REAL_SHARDS)
def test_a_speakable_answer_is_taken_on_the_first_call(monkeypatch, text):
    clean = "Il comando avvia il modello locale."
    model = _model(clean)
    monkeypatch.setattr(sanitizer, "generate_response", model)

    assert sanitize_chunk(text, 0) == clean
    assert len(model.calls) == 1


def test_a_dirty_answer_is_challenged_once_and_the_retry_is_kept(monkeypatch):
    model = _model("ancora ⟪sporco⟫", "adesso pulito.")
    monkeypatch.setattr(sanitizer, "generate_response", model)

    assert sanitize_chunk("usa ⟪ollama⟫", 0) == "adesso pulito."
    assert len(model.calls) == 2


def test_the_retry_is_told_what_the_previous_answer_left_behind(monkeypatch):
    model = _model("il codice è su https://github.com/x", "il codice è su GitHub.")
    monkeypatch.setattr(sanitizer, "generate_response", model)

    sanitize_chunk("il codice è su ⟪github⟫", 0)
    assert "SOLIDUS" in model.calls[1]


def test_a_model_that_never_cleans_up_cannot_put_a_symbol_on_the_wire(monkeypatch):
    model = _model("ancora ⟪sporco⟫ https://x.com/y", "sempre ⟪sporco⟫ https://x.com/y")
    monkeypatch.setattr(sanitizer, "generate_response", model)

    out = sanitize_chunk("usa ⟪ollama⟫ da https://x.com/y", 0)
    assert len(model.calls) == sanitizer.MAX_ATTEMPTS
    assert unspeakable_chars(out) == []
    assert "sporco" in out, "the last resort drops the offending token, not the whole passage"


def test_a_failing_call_falls_back_to_deleting_rather_than_raising(monkeypatch):
    def boom(prompt):
        raise RuntimeError("bedrock is down")

    monkeypatch.setattr(sanitizer, "generate_response", boom)
    out = sanitize_chunk("il codice è su https://github.com/x per i dettagli", 0)
    assert unspeakable_chars(out) == []
    assert out == "il codice è su per i dettagli"


def test_a_network_error_falls_back_the_same_way(monkeypatch):
    # Not every LLM-call failure is a RuntimeError: a dropped connection or an
    # HTTP error from the backend must take the same fallback, instead of
    # aborting the whole article over one block.
    def boom(prompt):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(sanitizer, "generate_response", boom)
    out = sanitize_chunk("il codice è su https://github.com/x per i dettagli", 0)
    assert unspeakable_chars(out) == []
    assert out == "il codice è su per i dettagli"


def test_an_empty_answer_drops_the_block(monkeypatch):
    # How the prompt tells the model to decline, instead of explaining itself
    # in a sentence that would be narrated as article prose.
    model = _model("   ")
    monkeypatch.setattr(sanitizer, "generate_response", model)

    assert sanitize_chunk("usa ⟪ollama⟫", 0) == ""
    assert len(model.calls) == 1


@pytest.mark.parametrize("text", REAL_SHARDS)
def test_whatever_happens_the_output_is_speakable(monkeypatch, text):
    monkeypatch.setattr(sanitizer, "generate_response", _model("ancora ⟪sporco⟫", "⟪ancora⟫"))
    assert unspeakable_chars(sanitize_chunk(text, 0)) == []


# --- blocks and shards ------------------------------------------------------


def _doc(*texts):
    blocks = [Block(Kind.PROSE, text=t) for t in texts]
    for b in blocks:
        b.translated_text = b.text
    return Document(blocks=blocks, visuals={}, notes={})


def test_sanitize_blocks_fills_speech_text_and_writes_a_shard(monkeypatch, shards):
    monkeypatch.setattr(sanitizer, "generate_response", _model())
    doc = _doc("Prima frase.", "Seconda frase.")

    sanitize_blocks(doc, "Articolo")

    assert [b.speech_text for b in doc.blocks] == ["Prima frase.", "Seconda frase."]
    assert (sanitizer.speech_shard_path("Articolo", 0)).read_text() == "Prima frase."


def test_a_second_run_resumes_from_shards(monkeypatch, shards):
    monkeypatch.setattr(sanitizer, "generate_response", _model())
    sanitize_blocks(_doc("Prima frase."), "Articolo")

    def _unreachable(prompt):
        raise AssertionError("the shard was on disk: the model must not be called")

    monkeypatch.setattr(sanitizer, "generate_response", _unreachable)
    doc = _doc("Prima frase.")
    sanitize_blocks(doc, "Articolo")
    assert doc.blocks[0].speech_text == "Prima frase."


def test_the_preceding_block_is_offered_as_context(monkeypatch, shards):
    model = _model("pulito.")
    monkeypatch.setattr(sanitizer, "generate_response", model)

    doc = _doc("Il primo paragrafo introduce sigma.", "il valore ⟪sigma⟫ cresce")
    sanitize_blocks(doc, "Articolo", max_workers=1)

    assert "Il primo paragrafo introduce sigma." in model.calls[0]


def test_giving_up_is_reported_at_error_level_with_both_texts(monkeypatch, caplog):
    """The one trace there is that a sentence was mutilated.

    This path deletes whole tokens rather than replacing them with words, so
    the listener hears a broken sentence rather than a mispronounced symbol --
    harder to catch by ear, and the shard on disk keeps no record of what was
    removed.
    """
    monkeypatch.setattr(sanitizer, "generate_response", _model("ancora ⟪sporco⟫", "⟪ancora⟫"))

    with caplog.at_level(logging.ERROR):
        out = sanitize_chunk("complessità da O(n²) a O(n)", 7)

    record = next(r for r in caplog.records if r.levelno >= logging.ERROR)
    assert "before:" in record.getMessage() and "after:" in record.getMessage()
    assert repr(out) in record.getMessage()
