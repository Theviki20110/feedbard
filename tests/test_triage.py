"""One call per article decides what the regexes used to decide.

The stage fails open on purpose: a bad verdict loses part of an article, no
verdict at all just narrates a bibliography. So most of what is tested here is
what happens when the model, or its JSON, is wrong.
"""

import json

import pytest

from feedbard.ingestion.html_parser import Block, Document, Kind, Visual
from feedbard.pipeline import triage as triage_mod
from feedbard.pipeline.triage import EMPTY, apply, build_outline, parse_verdict, triage


def _doc(*kinds_and_texts):
    blocks = []
    visuals = {}
    for kind, text in kinds_and_texts:
        if kind is Kind.VISUAL:
            vid = f"v{len(visuals):08d}"
            visuals[vid] = Visual(
                vid=vid, src="s3://x", fetch_url="https://img.example/x.png", caption=text
            )
            blocks.append(Block(Kind.VISUAL, vid=vid))
        else:
            blocks.append(Block(kind, text=text))
    return Document(blocks=blocks, visuals=visuals, notes={})


@pytest.fixture
def shard(monkeypatch, tmp_path):
    monkeypatch.setattr(triage_mod, "TRIAGE_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(triage_mod, "triage_shard_path", lambda title: tmp_path / f"{title}.json")
    return tmp_path


def _model(reply):
    return lambda prompt: (reply, 0.0)


# --- the outline the model reads --------------------------------------------


def test_the_outline_carries_the_index_the_kind_and_an_excerpt():
    doc = _doc((Kind.PROSE, "Il modello è veloce."), (Kind.VISUAL, "Figura 1: la curva"))
    lines = build_outline(doc).splitlines()
    assert lines[0] == "0\tprose\tIl modello è veloce."
    assert lines[1] == "1\tvisual\tFigura 1: la curva"


# --- parsing ----------------------------------------------------------------


def test_a_well_formed_verdict_is_taken():
    raw = '{"tail_from": 3, "drop": [1], "refers_back": [2]}'
    assert parse_verdict(raw, 5) == {"tail_from": 3, "drop": [1], "refers_back": [2]}


def test_json_wrapped_in_prose_is_still_read():
    raw = 'Sure! {"tail_from": null, "drop": [0], "refers_back": []} Hope that helps.'
    assert parse_verdict(raw, 2)["drop"] == [0]


@pytest.mark.parametrize("raw", ["not json", "", "[]", "{oops", "null"])
def test_an_unusable_reply_changes_nothing(raw):
    assert parse_verdict(raw, 5) == EMPTY


def test_out_of_range_indices_are_discarded_rather_than_trusted():
    # An index the model invented would otherwise blank a block at random or
    # raise mid-article.
    raw = '{"tail_from": 99, "drop": [1, 42, -1, "x"], "refers_back": [7]}'
    assert parse_verdict(raw, 3) == {"tail_from": None, "drop": [1], "refers_back": []}


# --- applying ---------------------------------------------------------------


def test_a_dropped_block_is_emptied_in_place_not_removed():
    # Every per-block shard on disk is keyed by index: removing a block would
    # shift every later index and invalidate the article's cached translation.
    doc = _doc((Kind.PROSE, "uno"), (Kind.PROSE, "due"), (Kind.PROSE, "tre"))
    apply(doc, {"tail_from": None, "drop": [1], "refers_back": []})

    assert len(doc.blocks) == 3
    assert doc.blocks[1].dropped and doc.blocks[1].text == ""
    assert doc.blocks[2].text == "tre"


def test_the_tail_and_everything_after_it_is_dropped():
    doc = _doc((Kind.PROSE, "corpo"), (Kind.HEADING, "Bibliografia"), (Kind.LIST, "Smith 2024"))
    apply(doc, {"tail_from": 1, "drop": [], "refers_back": []})

    assert not doc.blocks[0].dropped
    assert doc.blocks[1].dropped and doc.blocks[2].dropped
    assert doc.truncated_at == "Bibliografia"


def test_a_paragraph_that_refers_back_is_played_before_its_figure():
    doc = _doc((Kind.VISUAL, "Figura 1"), (Kind.PROSE, "come mostrato sopra, la curva sale"))
    apply(doc, {"tail_from": None, "drop": [], "refers_back": [1]})

    assert [b.kind for b in doc.blocks] == [Kind.PROSE, Kind.VISUAL]


def test_a_paragraph_that_points_forward_is_left_where_it_is():
    doc = _doc((Kind.VISUAL, "Figura 1"), (Kind.PROSE, "il diagramma seguente mostra"))
    apply(doc, {"tail_from": None, "drop": [], "refers_back": []})

    assert [b.kind for b in doc.blocks] == [Kind.VISUAL, Kind.PROSE]


# --- the stage --------------------------------------------------------------


def test_the_verdict_is_cached_and_the_second_run_makes_no_call(monkeypatch, shard):
    calls = []

    def generate_response(prompt):
        calls.append(prompt)
        return '{"tail_from": null, "drop": [1], "refers_back": []}', 0.0

    monkeypatch.setattr(triage_mod, "generate_response", generate_response)

    for _ in range(2):
        doc = _doc((Kind.PROSE, "uno"), (Kind.PROSE, "due"))
        triage(doc, "Articolo")
        assert doc.blocks[1].dropped

    assert len(calls) == 1
    assert json.loads((shard / "Articolo.json").read_text())["drop"] == [1]


def test_a_failing_call_leaves_the_article_exactly_as_parsed(monkeypatch, shard):
    def boom(prompt):
        raise RuntimeError("bedrock is down")

    monkeypatch.setattr(triage_mod, "generate_response", boom)
    doc = _doc((Kind.PROSE, "uno"), (Kind.PROSE, "due"))
    triage(doc, "Articolo")

    assert not any(b.dropped for b in doc.blocks)
    assert doc.truncated_at is None


def test_an_empty_document_makes_no_call(monkeypatch, shard):
    def _unreachable(prompt):
        raise AssertionError("nothing to judge: the model must not be called")

    monkeypatch.setattr(triage_mod, "generate_response", _unreachable)
    assert triage(Document(blocks=[], visuals={}, notes={}), "Articolo") == EMPTY


def test_a_dropped_block_never_gets_a_description(monkeypatch, shard):
    from feedbard.pipeline.narration import attach_descriptions

    monkeypatch.setattr(
        triage_mod,
        "generate_response",
        _model('{"tail_from": null, "drop": [0], "refers_back": []}'),
    )
    doc = _doc((Kind.VISUAL, "Figura 1"))
    doc.visuals["v00000000"].klass = "essential"
    doc.visuals["v00000000"].description = "Il grafico sale."

    triage(doc, "Articolo")
    assert attach_descriptions(doc) == 0
    assert doc.blocks[0].translated_text is None
