from feedbard.ingestion.html_parser import Block, Document, Kind
from feedbard.pipeline import table_describer
from feedbard.pipeline.table_describer import (
    describe_tables,
    flatten_rows,
    render_rows,
    table_id,
)

ROWS = [["modello", "punteggio"], ["7B", "41.2"], ["70B", "53.9"]]


def _doc(rows):
    return Document(blocks=[Block(Kind.TABLE, rows=rows)], visuals={}, notes={})


def test_table_id_is_stable_and_content_addressed():
    assert table_id(ROWS) == table_id([list(r) for r in ROWS])
    assert table_id(ROWS) != table_id([["modello", "punteggio"], ["7B", "41.3"]])


def test_render_rows_keeps_the_grid_readable_for_the_model():
    assert render_rows(ROWS).splitlines()[0] == "modello | punteggio"


def test_flatten_rows_keeps_every_value():
    flat = flatten_rows(ROWS)
    for cell in ("modello", "7B", "41.2", "70B", "53.9"):
        assert cell in flat


def test_description_is_attached_to_the_block(monkeypatch, tmp_path):
    monkeypatch.setattr(table_describer, "TABLE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(table_describer, "table_shard_path", lambda tid: tmp_path / f"{tid}.json")
    monkeypatch.setattr(
        table_describer,
        "generate_response",
        lambda prompt: ("Il modello grande segna cinquantatre virgola nove.", 0.0),
    )

    doc = _doc(ROWS)
    describe_tables(doc, title="Un articolo")
    assert doc.blocks[0].description == "Il modello grande segna cinquantatre virgola nove."


def test_a_second_run_resumes_from_the_shard(monkeypatch, tmp_path):
    monkeypatch.setattr(table_describer, "TABLE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(table_describer, "table_shard_path", lambda tid: tmp_path / f"{tid}.json")
    calls = []

    def generate_response(prompt):
        calls.append(prompt)
        return "Una descrizione.", 0.0

    monkeypatch.setattr(table_describer, "generate_response", generate_response)

    describe_tables(_doc(ROWS))
    describe_tables(_doc(ROWS))
    assert len(calls) == 1


def test_a_refusal_falls_back_to_reading_the_rows(monkeypatch, tmp_path):
    # The numbers have to survive a model that will not play along.
    monkeypatch.setattr(table_describer, "TABLE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(table_describer, "table_shard_path", lambda tid: tmp_path / f"{tid}.json")
    monkeypatch.setattr(
        table_describer,
        "generate_response",
        lambda prompt: ("Mi dispiace, non posso aiutarti con questa richiesta.", 0.0),
    )

    doc = _doc(ROWS)
    describe_tables(doc)
    assert "41.2" in doc.blocks[0].description
    assert "53.9" in doc.blocks[0].description


def test_an_llm_error_falls_back_to_reading_the_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(table_describer, "TABLE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(table_describer, "table_shard_path", lambda tid: tmp_path / f"{tid}.json")

    def boom(prompt):
        raise RuntimeError("bedrock unavailable")

    monkeypatch.setattr(table_describer, "generate_response", boom)

    doc = _doc(ROWS)
    describe_tables(doc)
    assert "41.2" in doc.blocks[0].description


def test_a_network_error_also_falls_back_to_reading_the_rows(monkeypatch, tmp_path):
    # Not every LLM-call failure is a ModelRefusedError/RuntimeError: a
    # dropped connection or an HTTP error from the backend must degrade the
    # same way, instead of aborting the whole article.
    monkeypatch.setattr(table_describer, "TABLE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(table_describer, "table_shard_path", lambda tid: tmp_path / f"{tid}.json")

    def boom(prompt):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(table_describer, "generate_response", boom)

    doc = _doc(ROWS)
    describe_tables(doc)
    assert "41.2" in doc.blocks[0].description


def test_large_tables_are_truncated_for_the_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(table_describer, "TABLE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(table_describer, "table_shard_path", lambda tid: tmp_path / f"{tid}.json")
    seen = []

    def generate_response(prompt):
        seen.append(prompt)
        return "Una descrizione.", 0.0

    monkeypatch.setattr(table_describer, "generate_response", generate_response)

    rows = [[f"riga{i}", str(i)] for i in range(table_describer.MAX_PROMPT_ROWS + 20)]
    describe_tables(_doc(rows))
    assert "riga0" in seen[0]
    assert f"riga{table_describer.MAX_PROMPT_ROWS + 10}" not in seen[0]
