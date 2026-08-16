"""process_item wires the pipeline stages together; process_feeds runs it over
a batch. These tests fake every stage so they exercise only the wiring: order,
data hand-off, and return value -- not what any individual stage does (each
has its own test module).
"""

import pytest

from feedbard.ingestion.html_parser import Document
from feedbard.pipeline import orchestrator


def _item(title="Un articolo", text="<p>ciao</p>"):
    return {"title": title, "text": text, "author": "Autore", "published_at": ""}


class _CallOrder(list):
    """A list of stage names in call order, plus the stand-in Document the
    faked `extract` returns -- handy for tests that need to hand it to a
    replacement stage of their own."""

    doc = None


@pytest.fixture
def stages(monkeypatch):
    """Patch every stage to a no-op that records its name and passes its
    first argument through unchanged, so process_item's wiring can be
    observed without any stage's real behaviour."""
    order = _CallOrder()

    def make(name, ret=None):
        def stage(*args, **kwargs):
            order.append(name)
            return ret(*args, **kwargs) if callable(ret) else ret

        return stage

    # _log_extraction reads doc.blocks/.visuals/.notes for real, so the
    # stand-in has to be a real (empty) Document rather than a bare object.
    doc = Document(blocks=[], visuals={}, notes={})
    order.doc = doc
    monkeypatch.setattr(orchestrator, "extract", make("extract", lambda html: doc))
    monkeypatch.setattr(orchestrator, "describe_visuals", make("describe_visuals"))
    monkeypatch.setattr(orchestrator, "describe_tables", make("describe_tables"))
    monkeypatch.setattr(orchestrator, "attach_descriptions", make("attach_descriptions", 0))
    monkeypatch.setattr(
        orchestrator, "translate_blocks", make("translate_blocks", lambda d, t: d)
    )
    monkeypatch.setattr(orchestrator, "sanitize_blocks", make("sanitize_blocks", lambda d, t: d))
    monkeypatch.setattr(
        orchestrator, "generate_audio_from_blocks", make("generate_audio_from_blocks", "dest.mp3")
    )
    monkeypatch.setattr(orchestrator, "publish", make("publish"))
    return order


def test_process_item_runs_the_stages_in_pipeline_order(stages):
    orchestrator.process_item(_item())
    assert stages == [
        "extract",
        "describe_visuals",
        "describe_tables",
        "attach_descriptions",
        "translate_blocks",
        "sanitize_blocks",
        "generate_audio_from_blocks",
        "publish",
    ]


def test_visuals_and_tables_are_described_before_descriptions_are_attached(stages):
    # attach_descriptions reads Visual.description / Block.description, which
    # only exist once the describers have run.
    orchestrator.process_item(_item())
    assert stages.index("describe_visuals") < stages.index("attach_descriptions")
    assert stages.index("describe_tables") < stages.index("attach_descriptions")


def test_descriptions_are_attached_before_translation(stages):
    # Otherwise a figure/table description written in the narration language
    # would be handed to the translator and paraphrased.
    orchestrator.process_item(_item())
    assert stages.index("attach_descriptions") < stages.index("translate_blocks")


def test_publish_runs_after_audio_is_rendered(stages):
    orchestrator.process_item(_item())
    assert stages.index("generate_audio_from_blocks") < stages.index("publish")


def test_process_item_returns_the_rendered_path(stages):
    assert orchestrator.process_item(_item()) == "dest.mp3"


def test_process_item_publishes_the_original_item_not_the_document(monkeypatch, stages):
    seen = {}
    monkeypatch.setattr(orchestrator, "publish", lambda item: seen.setdefault("item", item))
    item = _item(title="Titolo Specifico")

    orchestrator.process_item(item)
    assert seen["item"] is item


def test_process_feeds_processes_every_item_in_order(stages):
    items = [_item("Uno"), _item("Due"), _item("Tre")]
    dests = orchestrator.process_feeds(items)
    assert dests == ["dest.mp3", "dest.mp3", "dest.mp3"]
    # extract runs once per item
    assert stages.count("extract") == 3


def test_process_feeds_stops_at_the_first_failing_item(monkeypatch, stages):
    """Documents current behaviour: process_feeds has no per-item error
    isolation, so one broken article aborts every item after it in the same
    batch -- a real gap, not a feature, tracked here so a future fix changes
    this test on purpose rather than by surprise."""
    calls = []

    def flaky_extract(html):
        calls.append(html)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return stages.doc

    monkeypatch.setattr(orchestrator, "extract", flaky_extract)
    items = [_item("Uno", text="a"), _item("Due", text="b"), _item("Tre", text="c")]

    with pytest.raises(RuntimeError, match="boom"):
        orchestrator.process_feeds(items)

    assert calls == ["a", "b"]  # the third item was never attempted
