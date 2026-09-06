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


def test_a_refusal_drops_the_block_rather_than_reading_it_raw(monkeypatch, tmp_path):
    # Unlike a table's numbers, raw code carries nothing a listener can use:
    # a description that failed should leave the block silent, not fall back
    # to reading the source characters aloud.
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")
    monkeypatch.setattr(
        code_describer,
        "generate_response",
        lambda prompt: ("Mi dispiace, non posso aiutarti con questa richiesta.", 0.0),
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


def test_a_network_error_also_drops_the_block(monkeypatch, tmp_path):
    # Not every LLM-call failure is a ModelRefusedError/RuntimeError: a
    # dropped connection or an HTTP error from the backend must degrade the
    # same way, instead of aborting the whole article.
    monkeypatch.setattr(code_describer, "CODE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(code_describer, "code_shard_path", lambda cid: tmp_path / f"{cid}.json")

    def boom(prompt):
        raise ConnectionError("connection refused")

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
