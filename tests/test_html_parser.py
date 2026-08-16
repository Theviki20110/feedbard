from feedbard.ingestion.html_parser import (
    SYM_CLOSE,
    SYM_OPEN,
    Kind,
    extract,
    geometry,
    normalize_image_url,
    stable_id,
)

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


# --------------------------------------------------------------------------
# Article root selection
# --------------------------------------------------------------------------


def test_falls_back_to_article_tag_when_no_known_container_matches():
    doc = extract("<html><body><article><p>Contenuto.</p></article></body></html>")
    assert [b.text for b in doc.blocks if b.kind is Kind.PROSE] == ["Contenuto."]


def test_prefers_the_substack_selector_over_a_generic_article_tag():
    html = """
    <article>
      <p>Non è questo il corpo.</p>
      <div class="available-content"><p>Questo è il corpo vero.</p></div>
    </article>
    """
    doc = extract(html)
    prose = [b.text for b in doc.blocks if b.kind is Kind.PROSE]
    assert prose == ["Questo è il corpo vero."]


def test_nested_generic_containers_are_descended_not_flattened():
    # Substack nests plenty of plain <div>s; their children must keep their
    # own block type instead of being merged into one opaque blob.
    html = """
    <article><div><div><p>Primo.</p><h2>Titolo</h2><p>Secondo.</p></div></div></article>
    """
    doc = extract(html)
    kinds = [b.kind for b in doc.blocks]
    assert kinds == [Kind.PROSE, Kind.HEADING, Kind.PROSE]


# --------------------------------------------------------------------------
# Boilerplate stripping
# --------------------------------------------------------------------------


def test_boilerplate_nodes_are_dropped_at_the_block_level():
    html = """
    <article>
      <p>Prosa vera.</p>
      <div class="subscription-widget-wrap"><p>Iscriviti ora per continuare!</p></div>
      <nav><p>Menu di navigazione.</p></nav>
    </article>
    """
    doc = extract(html)
    texts = [b.text for b in doc.blocks if b.kind is Kind.PROSE]
    assert texts == ["Prosa vera."]
    assert doc.dropped[".subscription-widget-wrap"] == 1


def test_hr_script_style_svg_and_button_produce_no_block():
    html = """
    <article>
      <p>Prosa.</p>
      <hr><script>evil()</script><style>.x{}</style><svg></svg><button>Clicca</button>
    </article>
    """
    doc = extract(html)
    assert len(doc.blocks) == 1
    assert doc.blocks[0].kind is Kind.PROSE


# --------------------------------------------------------------------------
# Tail heading truncation
# --------------------------------------------------------------------------


def test_tail_heading_stops_extraction_and_is_not_itself_emitted():
    html = """
    <article>
      <p>Corpo dell'articolo.</p>
      <h2>Bibliography</h2>
      <p>Un riferimento che non deve essere narrato.</p>
    </article>
    """
    doc = extract(html)
    assert [b.text for b in doc.blocks] == ["Corpo dell'articolo."]
    assert doc.truncated_at == "Bibliography"


def test_tail_heading_match_is_case_insensitive_and_matches_italian_too():
    doc = extract("<article><p>Corpo.</p><h3>Bibliografia</h3><p>Nota.</p></article>")
    assert doc.truncated_at == "Bibliografia"
    assert len(doc.blocks) == 1


# --------------------------------------------------------------------------
# Headings and prose
# --------------------------------------------------------------------------


def test_heading_level_is_recorded():
    doc = extract("<article><h3>Un titolo</h3></article>")
    heading = doc.blocks[0]
    assert heading.kind is Kind.HEADING
    assert heading.level == 3
    assert heading.text == "Un titolo"


def test_prose_whitespace_is_collapsed():
    doc = extract("<article><p>Testo   con   spazi\n\nmultipli.</p></article>")
    assert doc.blocks[0].text == "Testo con spazi multipli."


def test_empty_prose_produces_no_block():
    doc = extract("<article><p>   </p><p>Reale.</p></article>")
    assert [b.text for b in doc.blocks] == ["Reale."]


# --------------------------------------------------------------------------
# Anchors and inline code
# --------------------------------------------------------------------------


def test_generic_anchor_text_is_discarded():
    html = '<article><p>Il codice è <a href="https://x.example">qui</a>.</p></article>'
    doc = extract(html)
    assert doc.blocks[0].text == "Il codice è ."


def test_meaningful_anchor_text_is_kept_without_the_url():
    html = '<article><p>Vedi <a href="https://x.example">il paper originale</a>.</p></article>'
    doc = extract(html)
    assert doc.blocks[0].text == "Vedi il paper originale."


def test_inline_code_is_wrapped_in_symbol_delimiters():
    doc = extract("<article><p>Usa <code>enable_thinking=True</code> per attivarlo.</p></article>")
    expected = f"Usa {SYM_OPEN}enable_thinking=True{SYM_CLOSE} per attivarlo."
    assert doc.blocks[0].text == expected


# --------------------------------------------------------------------------
# Code blocks
# --------------------------------------------------------------------------


def test_plain_pre_code_block_detects_language_from_class():
    html = '<article><pre><code class="language-python">x = 1</code></pre></article>'
    doc = extract(html)
    block = doc.blocks[0]
    assert block.kind is Kind.CODE
    assert block.lang == "python"
    assert "x = 1" in block.text


def test_highlighted_code_block_reads_language_from_data_attrs():
    html = """
    <article>
      <div class="highlighted_code_block" data-attrs='{"language": "bash"}'>
        <pre><span class="line">uv run pytest</span></pre>
      </div>
    </article>
    """
    doc = extract(html)
    block = doc.blocks[0]
    assert block.kind is Kind.CODE
    assert block.lang == "bash"
    assert block.text == "uv run pytest"
    assert block.lines == 1


def test_code_block_with_no_language_hint_still_extracts_text():
    doc = extract("<article><pre>echo hello</pre></article>")
    assert doc.blocks[0].lang is None
    assert "echo hello" in doc.blocks[0].text


# --------------------------------------------------------------------------
# Quotes and lists
# --------------------------------------------------------------------------


def test_blockquote_becomes_a_quote_block():
    doc = extract("<article><blockquote>Una citazione importante.</blockquote></article>")
    assert doc.blocks[0].kind is Kind.QUOTE
    assert doc.blocks[0].text == "Una citazione importante."


def test_list_items_are_joined_with_newlines():
    doc = extract("<article><ul><li>Primo</li><li>Secondo</li></ul></article>")
    block = doc.blocks[0]
    assert block.kind is Kind.LIST
    assert block.text == "Primo\nSecondo"


def test_nested_list_items_do_not_leak_into_the_parent_list():
    html = "<article><ul><li>Primo<ul><li>Annidato</li></ul></li><li>Secondo</li></ul></article>"
    doc = extract(html)
    lists = [b for b in doc.blocks if b.kind is Kind.LIST]
    # find_all(..., recursive=False) on the outer <ul> must not also pick up
    # the nested <li>, which would duplicate or scramble its text.
    assert lists[0].text.count("Primo") == 1


# --------------------------------------------------------------------------
# Visual extraction: geometry, classification, captions
# --------------------------------------------------------------------------


def test_visual_reads_dimensions_and_caption_from_data_attrs():
    html = """
    <article>
      <figure>
        <img data-attrs='{"src": "https://cdn.example/a.png", "width": 800, "height": 400,
                          "alt": "un grafico"}' src="https://cdn.example/thumb_a.png">
        <figcaption>Figura 1: andamento del punteggio</figcaption>
      </figure>
    </article>
    """
    doc = extract(html)
    visual_block = next(b for b in doc.blocks if b.kind is Kind.VISUAL)
    v = doc.visuals[visual_block.vid]
    assert v.src == "https://cdn.example/a.png"
    assert v.width == 800 and v.height == 400
    assert v.alt == "un grafico"
    assert v.caption == "Figura 1: andamento del punteggio"


def test_image_inside_a_paragraph_becomes_its_own_visual_block():
    html = (
        '<article><p>Testo prima <img src="https://cdn.example/chart_1000x500.png"> '
        "e dopo.</p></article>"
    )
    doc = extract(html)
    kinds = [b.kind for b in doc.blocks]
    assert Kind.VISUAL in kinds
    prose = next(b for b in doc.blocks if b.kind is Kind.PROSE)
    assert "img" not in prose.text.lower()
    assert "Testo prima" in prose.text and "e dopo" in prose.text


def test_wide_aspect_ratio_is_classified_as_formula():
    doc = extract('<article><img src="https://cdn.example/eq_2000x400.png"></article>')
    v = next(iter(doc.visuals.values()))
    assert v.hint == "formula"


def test_moderately_wide_short_image_is_classified_as_banner():
    doc = extract('<article><img src="https://cdn.example/head_1200x400.png"></article>')
    v = next(iter(doc.visuals.values()))
    assert v.hint == "banner"


def test_ordinary_aspect_ratio_is_classified_as_figure():
    doc = extract('<article><img src="https://cdn.example/plot_1200x1000.png"></article>')
    v = next(iter(doc.visuals.values()))
    assert v.hint == "figure"


def test_caption_keywords_override_geometry_towards_formula():
    html = """
    <article><figure>
      <img src="https://cdn.example/plot_1200x1000.png">
      <figcaption>Formal definition of the loss</figcaption>
    </figure></article>
    """
    doc = extract(html)
    v = next(iter(doc.visuals.values()))
    assert v.hint == "formula"


def test_noise_only_caption_is_discarded():
    html = """
    <article><figure>
      <img src="https://cdn.example/plot_1200x1000.png">
      <figcaption>(from [1, 3])</figcaption>
    </figure></article>
    """
    doc = extract(html)
    v = next(iter(doc.visuals.values()))
    assert v.caption == ""


def test_hero_image_is_flagged_from_top_image_attr():
    html = """
    <article><img data-attrs='{"src": "https://cdn.example/a.png", "topImage": true}'
                   src="https://cdn.example/thumb.png"></article>
    """
    doc = extract(html)
    v = next(iter(doc.visuals.values()))
    assert v.hero is True


def test_same_image_reused_twice_shares_one_visual_entry():
    html = """
    <article>
      <img src="https://cdn.example/a_1200x900.png">
      <p>Testo di mezzo.</p>
      <img src="https://cdn.example/a_1200x900.png">
    </article>
    """
    doc = extract(html)
    visual_blocks = [b for b in doc.blocks if b.kind is Kind.VISUAL]
    assert len(visual_blocks) == 2
    assert visual_blocks[0].vid == visual_blocks[1].vid
    assert len(doc.visuals) == 1


# --------------------------------------------------------------------------
# Deictic repositioning
# --------------------------------------------------------------------------


def test_visual_moves_after_a_paragraph_that_refers_back_to_it():
    html = """
    <article>
      <img src="https://cdn.example/a_1200x900.png">
      <p>Come mostrato sopra, il modello converge rapidamente.</p>
    </article>
    """
    doc = extract(html)
    assert [b.kind for b in doc.blocks] == [Kind.PROSE, Kind.VISUAL]


def test_visual_stays_put_before_a_paragraph_that_refers_forward():
    html = """
    <article>
      <img src="https://cdn.example/a_1200x900.png">
      <p>Come mostrato sotto, il modello converge rapidamente.</p>
    </article>
    """
    doc = extract(html)
    assert [b.kind for b in doc.blocks] == [Kind.VISUAL, Kind.PROSE]


def test_visual_with_no_deictic_reference_keeps_dom_order():
    html = """
    <article>
      <img src="https://cdn.example/a_1200x900.png">
      <p>Un paragrafo qualunque senza riferimenti.</p>
    </article>
    """
    doc = extract(html)
    assert [b.kind for b in doc.blocks] == [Kind.VISUAL, Kind.PROSE]


# --------------------------------------------------------------------------
# Pure helper functions
# --------------------------------------------------------------------------


def test_normalize_image_url_unquotes_an_encapsulated_source():
    encapsulated = "https://substackcdn.com/image/fetch/w_1200/https%3A%2F%2Fbucket.s3%2Fa.png"
    assert normalize_image_url(encapsulated) == "https://bucket.s3/a.png"


def test_normalize_image_url_finds_a_second_embedded_https():
    embedded = "https://cdn.example/proxy?url=https://origin.example/a.png"
    assert normalize_image_url(embedded) == "https://origin.example/a.png"


def test_normalize_image_url_leaves_a_plain_url_untouched():
    plain = "https://cdn.example/a.png"
    assert normalize_image_url(plain) == plain


def test_geometry_reads_dimensions_from_the_filename():
    w, h, hint = geometry("https://cdn.example/chart_1200x600.png")
    assert (w, h) == (1200, 600)
    assert hint == "figure"


def test_geometry_returns_none_when_no_dimensions_are_encoded():
    assert geometry("https://cdn.example/chart.png") == (None, None, "figure")


def test_stable_id_is_deterministic_and_content_addressed():
    a = stable_id("https://cdn.example/a.png")
    b = stable_id("https://cdn.example/a.png")
    c = stable_id("https://cdn.example/b.png")
    assert a == b
    assert a != c
    assert len(a) == 10
