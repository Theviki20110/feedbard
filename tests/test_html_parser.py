from feedbard.ingestion.html_parser import Kind, extract

FOOTNOTE_HTML = """
<article>
  <p>Il modello è stato addestrato su un corpus filtrato<a href="#footnote-1">1</a>.</p>
  <p>Un secondo paragrafo senza note.</p>
  <div class="footnote">
    <a class="footnote-number" id="footnote-1">1</a>
    <div class="footnote-content">Il filtro rimuove i duplicati esatti.</div>
  </div>
</article>
"""


def test_footnote_body_is_folded_into_the_citing_block():
    # Collected and then dropped, the note used to disappear from the episode
    # while its marker was stripped for prosody: content lost in silence.
    doc = extract(FOOTNOTE_HTML)
    prose = [b for b in doc.blocks if b.kind is Kind.PROSE]
    assert "Il filtro rimuove i duplicati esatti." in prose[0].text
    assert "Il filtro" not in prose[1].text


def test_footnote_marker_never_reaches_the_text_inline():
    doc = extract(FOOTNOTE_HTML)
    prose = [b for b in doc.blocks if b.kind is Kind.PROSE]
    assert "filtrato Note:" in prose[0].text or "filtrato. Note:" in prose[0].text


def test_a_block_without_notes_is_untouched():
    doc = extract("<article><p>Solo prosa.</p></article>")
    assert doc.blocks[0].text == "Solo prosa."


TABLE_HTML = """
<article>
  <table>
    <tr><th>modello</th><th>punteggio</th></tr>
    <tr><td>7B</td><td>41.2</td></tr>
  </table>
</article>
"""


def test_table_keeps_its_grid_and_has_no_text_of_its_own():
    doc = extract(TABLE_HTML)
    table = next(b for b in doc.blocks if b.kind is Kind.TABLE)
    assert table.rows == [["modello", "punteggio"], ["7B", "41.2"]]
    # The describer is what gives it a voice; the parser deliberately does not.
    assert table.text == ""
    assert table.description is None
