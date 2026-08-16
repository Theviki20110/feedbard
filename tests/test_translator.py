import pytest

from feedbard.ingestion.html_parser import Block, Document, Kind
from feedbard.pipeline import translator
from feedbard.pipeline.sanitizer import ModelRefusedError
from feedbard.pipeline.translator import translate_blocks, translate_chunk


def _doc(blocks):
    return Document(blocks=list(blocks), visuals={}, notes={})


def _unreachable(prompt):
    raise AssertionError("generate_response should not have been called")


@pytest.fixture
def shards(monkeypatch, tmp_path):
    def shard_path(title, index):
        return tmp_path / f"{title}_{index}.txt"

    monkeypatch.setattr(translator, "TEXT_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(translator, "text_shard_path", shard_path)
    return tmp_path


# --------------------------------------------------------------------------
# translate_chunk
# --------------------------------------------------------------------------


def test_empty_chunk_skips_the_model_entirely(monkeypatch):
    monkeypatch.setattr(translator, "generate_response", _unreachable)
    assert translate_chunk("​ \n", "Italian", 0) == ""


def test_translate_chunk_returns_the_model_output(monkeypatch):
    monkeypatch.setattr(translator, "generate_response", lambda prompt: ("ciao mondo", 0.1))
    assert translate_chunk("hello world", "Italian", 0) == "ciao mondo"


def test_translate_chunk_passes_language_and_source_into_the_prompt(monkeypatch):
    seen = {}

    def generate_response(prompt):
        seen["prompt"] = prompt
        return "ok", 0.0

    monkeypatch.setattr(translator, "generate_response", generate_response)
    translate_chunk("hello", "French", 0)

    assert "hello" in seen["prompt"]
    assert "French" in seen["prompt"]


def test_translate_chunk_rejects_a_refusal(monkeypatch):
    monkeypatch.setattr(
        translator,
        "generate_response",
        lambda prompt: ("I'm ready to translate. Please provide the text.", 0.0),
    )
    with pytest.raises(ModelRefusedError):
        translate_chunk("hello", "Italian", 0)


def test_translate_chunk_retries_a_refusal_and_keeps_the_later_success(monkeypatch):
    # A refusal is sampled, not deterministic: the same prompt run again
    # often just answers normally, so one bad draw should not fail the block.
    responses = iter(
        [
            ("Mi dispiace, ma non posso procedere con questa richiesta.", 0.0),
            ("ciao mondo", 0.1),
        ]
    )
    monkeypatch.setattr(translator, "generate_response", lambda prompt: next(responses))

    assert translate_chunk("hello", "Italian", 0) == "ciao mondo"


def test_translate_chunk_gives_up_after_exhausting_refusal_retries(monkeypatch):
    calls = []

    def always_refuses(prompt):
        calls.append(prompt)
        return "I'm ready to translate. Please provide the text.", 0.0

    monkeypatch.setattr(translator, "generate_response", always_refuses)

    with pytest.raises(ModelRefusedError):
        translate_chunk("hello", "Italian", 0)

    assert len(calls) == translator.MAX_REFUSAL_RETRIES + 1


def test_inline_code_markers_survive_translation(monkeypatch):
    # The sanitizer stage consumes ⟪⟫ markers; the translator must not eat them.
    monkeypatch.setattr(translator, "generate_response", lambda prompt: ("usa ⟪git clone⟫", 0.0))
    assert translate_chunk("use ⟪git clone⟫", "Italian", 0) == "usa ⟪git clone⟫"


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------


def test_a_second_run_resumes_from_the_shard(monkeypatch, shards):
    calls = []

    def generate_response(prompt):
        calls.append(prompt)
        return "tradotto", 0.0

    monkeypatch.setattr(translator, "generate_response", generate_response)

    translate_blocks(_doc([Block(Kind.PROSE, text="hello")]), title="Articolo")
    translate_blocks(_doc([Block(Kind.PROSE, text="hello")]), title="Articolo")

    assert len(calls) == 1


def test_shard_is_written_after_a_successful_translation(monkeypatch, shards):
    monkeypatch.setattr(translator, "generate_response", lambda prompt: ("tradotto", 0.0))

    translate_blocks(_doc([Block(Kind.PROSE, text="hello")]), title="Articolo")

    assert (shards / "Articolo_0.txt").read_text() == "tradotto"


# --------------------------------------------------------------------------
# translate_blocks: which blocks are candidates
# --------------------------------------------------------------------------


def test_blocks_with_no_text_are_skipped(monkeypatch, shards):
    calls = []

    def generate_response(prompt):
        calls.append(prompt)
        return "x", 0.0

    monkeypatch.setattr(translator, "generate_response", generate_response)

    doc = _doc([Block(Kind.VISUAL, vid="aaaaaaaaaa")])  # empty text
    translate_blocks(doc, title="Articolo")

    assert calls == []
    assert doc.blocks[0].translated_text is None


def test_blocks_already_translated_by_narration_are_not_retranslated(monkeypatch, shards):
    # attach_descriptions writes translated_text directly, in the narration
    # language: re-translating it would paraphrase a figure/table description.
    monkeypatch.setattr(translator, "generate_response", _unreachable)

    block = Block(Kind.VISUAL, vid="aaaaaaaaaa")
    block.translated_text = "Il grafico mostra una curva."
    doc = _doc([block])

    translate_blocks(doc, title="Articolo")

    assert doc.blocks[0].translated_text == "Il grafico mostra una curva."


def test_multiple_blocks_are_all_translated_and_indexed_correctly(monkeypatch, shards):
    def generate_response(prompt):
        return (f"tradotto: {prompt.splitlines()[0]}", 0.0)

    monkeypatch.setattr(translator, "generate_response", generate_response)

    doc = _doc([Block(Kind.PROSE, text=f"paragraph {i}") for i in range(5)])
    translate_blocks(doc, title="Articolo", max_workers=3)

    assert all(b.translated_text is not None for b in doc.blocks)
    assert (shards / "Articolo_4.txt").exists()
