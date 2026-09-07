"""The join between the describers and the episode.

Every test here guards the same failure: a figure or a table that the
pipeline paid a model to describe, and that then never reached the audio.
"""

from feedbard.ingestion.html_parser import Block, Document, Kind, Visual, extract
from feedbard.pipeline.narration import attach_descriptions


def _doc(blocks, visuals=()):
    return Document(blocks=list(blocks), visuals={v.vid: v for v in visuals}, notes={})


def _visual(vid, klass, description):
    return Visual(vid=vid, src=f"s3://{vid}", fetch_url="", klass=klass, description=description)


def test_essential_visual_becomes_narratable():
    doc = _doc(
        [Block(Kind.PROSE, text="prosa"), Block(Kind.VISUAL, vid="aaaaaaaaaa")],
        [_visual("aaaaaaaaaa", "essential", "Il grafico mostra una curva che sale.")],
    )
    assert attach_descriptions(doc) == 1
    assert doc.blocks[1].translated_text == "Il grafico mostra una curva che sale."


def test_illustrative_visual_is_narrated_too():
    doc = _doc(
        [Block(Kind.VISUAL, vid="bbbbbbbbbb")],
        [_visual("bbbbbbbbbb", "illustrative", "Un ritratto dell'autore.")],
    )
    assert attach_descriptions(doc) == 1


def test_decorative_visual_stays_silent():
    # A banner narrated is an interruption, not information.
    doc = _doc(
        [Block(Kind.VISUAL, vid="cccccccccc")],
        [_visual("cccccccccc", "decorative", None)],
    )
    assert attach_descriptions(doc) == 0
    assert doc.blocks[0].translated_text is None


def test_visual_whose_description_failed_stays_silent():
    doc = _doc(
        [Block(Kind.VISUAL, vid="dddddddddd")],
        [_visual("dddddddddd", "essential", None)],
    )
    assert attach_descriptions(doc) == 0


def test_table_description_becomes_narratable():
    block = Block(Kind.TABLE, rows=[["modello", "punteggio"], ["7B", "41.2"]])
    block.description = "Il modello da sette miliardi segna quarantuno virgola due."
    doc = _doc([block])
    assert attach_descriptions(doc) == 1
    assert doc.blocks[0].translated_text.startswith("Il modello")


def test_code_description_becomes_narratable():
    block = Block(Kind.CODE, text="def add(a, b):\n    return a + b", lang="python")
    block.description = "Il codice definisce una funzione che somma due numeri."
    doc = _doc([block])
    assert attach_descriptions(doc) == 1
    assert doc.blocks[0].translated_text == "Il codice definisce una funzione che somma due numeri."


def test_code_whose_description_failed_stays_silent_not_raw():
    # Unlike VISUAL/TABLE, a CODE block's own `text` is never empty (it's the
    # raw code): a failed description has to mark the block handled, or the
    # raw code would fall through to the translator and be read aloud as-is.
    block = Block(Kind.CODE, text="[r r r r r r r r]")
    block.description = None
    doc = _doc([block])
    assert attach_descriptions(doc) == 0
    assert doc.blocks[0].translated_text == ""
    assert doc.blocks[0].text == "[r r r r r r r r]"


def test_prose_is_left_for_the_translator():
    doc = _doc([Block(Kind.PROSE, text="testo originale")])
    assert attach_descriptions(doc) == 0
    assert doc.blocks[0].translated_text is None


def test_block_count_is_unchanged():
    # Shards are keyed on the block index, so a stage that inserted or removed
    # a block would silently re-align every checkpoint already on disk.
    blocks = [
        Block(Kind.PROSE, text="prosa"),
        Block(Kind.VISUAL, vid="eeeeeeeeee"),
        Block(Kind.TABLE, rows=[["a", "b"]], description="una tabella"),
    ]
    doc = _doc(blocks, [_visual("eeeeeeeeee", "essential", "una figura")])
    attach_descriptions(doc)
    assert len(doc.blocks) == 3


HTML_WITH_FIGURE_AND_TABLE = """
<article>
  <p>Il paragrafo introduttivo.</p>
  <figure><img src="https://example.com/chart_1200x600.png"><figcaption>Fig 1</figcaption></figure>
  <table><tr><th>modello</th><th>punteggio</th></tr><tr><td>7B</td><td>41.2</td></tr></table>
</article>
"""


def test_extracted_figure_and_table_reach_the_audio_path():
    """End to end over the parse: both non-prose blocks arrive with no text of
    their own, and both leave this stage with something to say."""
    doc = extract(HTML_WITH_FIGURE_AND_TABLE)

    visual_block = next(b for b in doc.blocks if b.kind is Kind.VISUAL)
    table_block = next(b for b in doc.blocks if b.kind is Kind.TABLE)
    assert visual_block.text == "" and table_block.text == ""

    doc.visuals[visual_block.vid].klass = "essential"
    doc.visuals[visual_block.vid].description = "La curva sale fino a quaranta."
    table_block.description = "Il modello da sette miliardi segna quarantuno virgola due."

    assert attach_descriptions(doc) == 2
    # This is what the renderer reads for a block the sanitizer left alone.
    narrated = [
        b.speech_text if b.speech_text is not None else b.translated_text for b in doc.blocks
    ]
    assert "La curva sale fino a quaranta." in narrated
    assert "Il modello da sette miliardi segna quarantuno virgola due." in narrated
