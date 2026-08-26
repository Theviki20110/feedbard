import pytest

from feedbard.ingestion.html_parser import Block, Document, Kind
from feedbard.pipeline import code_describer
from feedbard.pipeline.code_describer import code_id, describe_code_blocks

DIAGRAM = "Full RoPE:    [r r r r r r r r]\nPartial RoPE: [r r r - - - -]"
CODE = "def add(a, b):\n    return a + b"


def _doc(text, lang=None):
    return Document(blocks=[Block(Kind.CODE, text=text, lang=lang)], visuals={}, notes={})


def test_code_id_is_stable_and_content_addressed():
    assert code_id(DIAGRAM) == code_id(str(DIAGRAM))
    assert code_id(DIAGRAM) != code_id(CODE)


def test_description_is_attached_to_the_block(monkeypatch, tmp_path):
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")
    monkeypatch.setattr(
        code_describer,
        "generate_response",
        lambda prompt: ("Il diagramma mostra la RoPE completa che ruota tutte le dimensioni.", 0.0),
    )

    doc = _doc(DIAGRAM)
    describe_code_blocks(doc, title="Un articolo")
    assert "RoPE" in doc.blocks[0].description


def test_a_second_run_resumes_from_the_shard(monkeypatch, tmp_path):
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")
    calls = []

    def generate_response(prompt):
        calls.append(prompt)
        return "Una descrizione.", 0.0

    monkeypatch.setattr(code_describer, "generate_response", generate_response)

    describe_code_blocks(_doc(CODE))
    describe_code_blocks(_doc(CODE))
    assert len(calls) == 1


def test_a_declined_block_is_dropped_rather_than_read_raw(monkeypatch, tmp_path):
    # Unlike a table's numbers, raw code carries nothing a listener can use:
    # a description that failed should leave the block silent, not fall back
    # to reading the source characters aloud.
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")
    monkeypatch.setattr(
        code_describer,
        "generate_response",
        lambda prompt: ("", 0.0),
    )

    doc = _doc(CODE)
    describe_code_blocks(doc)
    assert doc.blocks[0].description is None


def test_an_llm_error_drops_the_block(monkeypatch, tmp_path):
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")

    def boom(prompt):
        raise RuntimeError("bedrock unavailable")

    monkeypatch.setattr(code_describer, "generate_response", boom)

    doc = _doc(CODE)
    describe_code_blocks(doc)
    assert doc.blocks[0].description is None


def test_blocks_without_text_are_skipped(monkeypatch):
    monkeypatch.setattr(
        code_describer, "generate_response", lambda prompt: (_ for _ in ()).throw(AssertionError)
    )
    doc = _doc("")
    describe_code_blocks(doc)
    assert doc.blocks[0].description is None


# --- the model handing the block back --------------------------------------

FENCE = '```python\nimport numpy as np\nmodel = load_model("north-mini")\n```'
TRANSCRIPT = "$ ollama run qwen3.6:35b\npulling manifest\npulling 8a1b2c3d... 100%  4.7 GB"


@pytest.mark.parametrize("source", [FENCE, TRANSCRIPT, CODE])
def test_a_reply_that_reproduces_a_source_line_is_an_echo(source):
    assert code_describer.is_echo(source, source)
    assert code_describer.is_echo(source, f"Ecco il blocco:\n{source}")


@pytest.mark.parametrize(
    "description",
    [
        "Il blocco carica il modello north-mini e stampa il testo generato.",
        "Scarica il modello con ollama e mostra l'avanzamento del download.",
        "Definisce una funzione che somma i due argomenti.",
        # Naming an identifier is not echoing it: a description is allowed to
        # say what the thing is called.
        "Il blocco importa numpy e usa load_model per caricare il modello.",
    ],
)
def test_a_real_description_is_not_an_echo(description):
    assert not code_describer.is_echo(FENCE, description)
    assert not code_describer.is_echo(CODE, description)


def test_an_echo_is_asked_again_and_the_second_reply_is_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")
    replies = iter([(FENCE, 0.0), ("Il blocco carica il modello e stampa.", 0.0)])
    monkeypatch.setattr(code_describer, "generate_response", lambda prompt: next(replies))

    doc = _doc(FENCE)
    describe_code_blocks(doc)
    assert doc.blocks[0].description == "Il blocco carica il modello e stampa."


def test_a_block_that_is_only_ever_echoed_is_dropped_and_not_cached(monkeypatch, tmp_path):
    # Neither empty nor a refusal, so every other check passes: without this
    # guard the source reached the speech engine as if it were prose.
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")
    monkeypatch.setattr(code_describer, "generate_response", lambda prompt: (FENCE, 0.0))

    doc = _doc(FENCE)
    describe_code_blocks(doc)

    assert doc.blocks[0].description is None
    assert list(tmp_path.glob("*.json")) == [], "an echo must never reach the shard cache"
