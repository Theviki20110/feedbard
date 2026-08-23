"""
Deterministic DOM -> IR pass into typed blocks.

Input is an article-sized HTML fragment, whatever produced it: a publisher
API, a feed's content:encoded, or a readability pass over a page (see
`ingestion/fetcher.py`). The selectors below name the containers and the
boilerplate of the platforms seen so far -- an unknown one still parses,
falling back to `article`/`main` and carrying whatever widgets it had.

Goal: walk the tree ONCE and produce, at the same time,
  (a) the ordered block stream, with opaque placeholders in place of visuals
  (b) the resource registry, indexed by stable id

No LLM at this stage. The position of each resource in the stream is a
structural fact: flattening before recording it destroys it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, Tag

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Delimiters chosen so a downstream LLM isn't tempted to "fix" them:
# braces and math brackets get rewritten, these don't.
SYM_OPEN, SYM_CLOSE = "\u27ea", "\u27eb"  # double angle brackets, for inline symbols

# A visual keeps its own block in the stream rather than a placeholder inside
# a neighbouring one: the position is already recorded by the block order, and
# a token embedded in prose has to survive every LLM pass that touches it.

ARTICLE_SELECTORS = (
    # Substack
    "div.available-content",
    "div.body.markup",
    # Ghost / WordPress / common themes
    "div.gh-content",
    "div.entry-content",
    "div.post-content",
    "[itemprop='articleBody']",
    # Generic, and what a readability pass leaves behind
    "article",
    "main",
)

# Boilerplate: discarded at the BLOCK level, never via regex on the text.
BOILERPLATE_SELECTORS = (
    ".subscription-widget-wrap",
    ".subscription-widget-wrap-editor",
    ".subscription-widget",
    ".button-wrapper",  # <p> with the "Subscribe now" button
    ".digest-post-embed",  # cross-promo card mid-article
    ".post-ufi",
    ".comments-section",
    ".paywall",
    ".image-link-expand",  # restack / fullscreen buttons inside figures
    ".pencraft",
    # Ghost / WordPress / common themes
    ".sharedaddy",
    ".post-share",
    ".related-posts",
    ".newsletter-signup",
    ".wp-block-post-comments",
    "aside",
    "footer",
    "nav",
    "header",
    "form",
)

# Aspect ratio is a cheap PRE-ROUTER, not a reliable classifier:
# in this article a 3462x1056 strip (ratio 3.28) is a curve chart,
# and a 1306x690 one (ratio 1.89) is a formula. It only picks the starting
# prompt; the authoritative type comes from the VLM, which sees the image.
# Threshold set high on purpose: better to miss a few formulas than to label
# charts as formulas, because the "paraphrase the formula" prompt on a chart
# produces made-up output.
FORMULA_ASPECT_MIN = 4.0


class Kind(str, Enum):
    HEADING = "heading"
    PROSE = "prose"
    VISUAL = "visual"
    CODE = "code"
    TABLE = "table"
    QUOTE = "quote"
    LIST = "list"


@dataclass
class Block:
    kind: Kind
    text: str = ""
    vid: str | None = None
    level: int = 0
    lang: str | None = None
    lines: int = 0
    rows: list[list[str]] = field(default_factory=list)
    note_ids: list[str] = field(default_factory=list)
    deictic: str | None = None
    # Set by `pipeline.triage` on a block the model judged to carry nothing a
    # listener needs. The block stays in the list, emptied, so that every
    # per-block shard on disk keeps the index it was written under.
    dropped: bool = False
    # Narratable prose for a block that has none of its own. Filled in
    # downstream for TABLE (by table_describer); a VISUAL's equivalent lives on
    # the Visual, which is keyed on content and so shared between articles.
    description: str | None = None
    translated_text: str | None = None
    # Speech-ready form of translated_text: symbols spelled out, markup gone.
    # Kept separate so a sanitizer re-run never costs a re-translation.
    speech_text: str | None = None


@dataclass
class Visual:
    vid: str
    src: str  # canonical S3 source, used for the id
    fetch_url: str  # full-resolution variant, to hand to the VLM
    caption: str = ""
    alt: str = ""
    width: int | None = None
    height: int | None = None
    hint: str = "figure"  # formula | figure | banner
    hero: bool = False  # topImage: header image
    # filled in downstream by the vision pass, not here
    klass: str | None = None  # see visual_describer.KLASSES
    description: str | None = None


@dataclass
class Document:
    blocks: list[Block]
    visuals: dict[str, Visual]
    notes: dict[str, str]
    dropped: dict[str, int] = field(default_factory=dict)
    truncated_at: str | None = None


# --------------------------------------------------------------------------
# URL and geometry
# --------------------------------------------------------------------------


def normalize_image_url(url: str) -> str:
    """Fallback for when data-attrs is missing. Substack encapsulates the
    original asset, percent-encoded, at the tail of the CDN transform path."""
    i = url.find("https%3A%2F%2F")
    if i != -1:
        return unquote(url[i:])
    i = url.find("https://", len("https://"))
    if i != -1:
        return url[i:]
    return url


_DIM_RE = re.compile(r"_(\d{2,5})x(\d{2,5})\.(?:png|jpe?g|gif|webp)\b", re.I)


def _classify(w: int | None, h: int | None) -> str:
    if not w or not h:
        return "figure"
    r = w / h
    if r >= FORMULA_ASPECT_MIN:
        return "formula"
    if r >= 2.4 and h < 500:
        return "banner"
    return "figure"


def geometry(url: str) -> tuple[int | None, int | None, str]:
    """Native dimensions are in the Substack filename. A very wide ratio
    is a LaTeX formula strip, not a figure: telling them apart routes them
    to different prompts, without downloading a single byte."""
    m = _DIM_RE.search(url)
    if not m:
        return None, None, "figure"
    w, h = int(m.group(1)), int(m.group(2))
    return w, h, _classify(w, h)


def stable_id(canonical_src: str) -> str:
    return hashlib.sha256(canonical_src.encode("utf-8")).hexdigest()[:10]


def _as_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------


class Extractor:
    def __init__(self, html: str, parser: str = "lxml"):
        self.soup = BeautifulSoup(html, parser)
        self.blocks: list[Block] = []
        self.visuals: dict[str, Visual] = {}
        self.notes: dict[str, str] = {}
        self.dropped: dict[str, int] = {}
        self.truncated_at: str | None = None
        self._pending_notes: list[str] = []

    # -- entry ---------------------------------------------------------------

    def run(self) -> Document:
        root = self._find_article()
        self._strip_boilerplate(root)
        self._collect_footnote_bodies(root)
        self._pending_notes.clear()  # footnote bodies don't count
        for node in root.children:
            if isinstance(node, Tag):
                self._dispatch(node)
        self._inline_notes()
        return Document(self.blocks, self.visuals, self.notes, self.dropped, self.truncated_at)

    def _find_article(self) -> Tag:
        for sel in ARTICLE_SELECTORS:
            node = self.soup.select_one(sel)
            if node is not None:
                return node
        return self.soup.body or self.soup

    def _strip_boilerplate(self, root: Tag) -> None:
        for sel in BOILERPLATE_SELECTORS:
            hits = root.select(sel)
            if hits:
                self.dropped[sel] = len(hits)
            for node in hits:
                node.decompose()

    def _collect_footnote_bodies(self, root: Tag) -> None:
        """The id is on the <a class="footnote-number">, NOT on the container
        div. Taking it from the div leaves the notes keyless and silently
        drops them."""
        for node in root.select("div.footnote"):
            num = node.select_one("a.footnote-number")
            fid = ""
            if num is not None:
                fid = (num.get("id") or "").replace("footnote-", "").strip() or num.get_text(
                    strip=True
                )
                num.extract()
            body = node.select_one(".footnote-content") or node
            if fid:
                self.notes[fid] = self._collapse(self._inline(body))
            node.decompose()

    # -- dispatch --------------------------------------------------------------

    def _dispatch(self, node: Tag) -> None:
        name = node.name.lower()
        classes = node.get("class") or []

        if "highlighted_code_block" in classes:
            return self._emit_code_wrapper(node)

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return self._emit_heading(node, int(name[1]))
        if name == "p":
            return self._emit_prose(node)
        if name in ("figure", "picture"):
            return self._emit_visual(node)
        if name == "img":
            return self._emit_visual(node, img=node)
        if name == "pre":
            return self._emit_code(node)
        if name == "table":
            return self._emit_table(node)
        if name == "blockquote":
            return self._emit_quote(node)
        if name in ("ul", "ol"):
            return self._emit_list(node)
        if name in ("hr", "script", "style", "svg", "button"):
            return

        # Generic container (Substack nests plenty of these): descend one
        # level instead of flattening it, otherwise I lose the children's typing.
        for child in node.children:
            if isinstance(child, Tag):
                self._dispatch(child)

    # -- emitters --------------------------------------------------------------

    def _emit_heading(self, node: Tag, level: int) -> None:
        text = self._collapse(self._inline(node))
        if not text:
            return
        self.blocks.append(Block(Kind.HEADING, text=text, level=level))

    def _emit_prose(self, node: Tag) -> None:
        for img in node.find_all("img"):
            parent = img.parent
            self._emit_visual(
                parent if parent is not None and parent.name == "figure" else img, img=img
            )
            img.decompose()

        text = self._collapse(self._inline(node))
        if not text:
            return
        self.blocks.append(Block(Kind.PROSE, text=text, note_ids=self._drain_notes()))

    def _emit_visual(self, node: Tag, img: Tag | None = None) -> None:
        img = img or node.find("img")
        if img is None:
            return

        # data-attrs carries the canonical S3 source plus alt, title, and
        # native dimensions: it's the source of truth, un-encapsulation is
        # the fallback.
        attrs: dict = {}
        raw = img.get("data-attrs")
        if raw:
            try:
                attrs = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                attrs = {}

        displayed = img.get("src") or img.get("data-src") or ""
        # The <a> wrapping the figure points to the variant without a w_ limit.
        link = node.find("a", class_="image-link") if isinstance(node, Tag) else None
        fullres = (link.get("href") or "") if link is not None else ""

        canonical = attrs.get("src") or normalize_image_url(fullres or displayed)
        if not canonical:
            return
        vid = stable_id(canonical)

        w, h, hint = geometry(canonical)
        if w is None:
            w, h = _as_int(attrs.get("width")), _as_int(attrs.get("height"))
            hint = _classify(w, h)

        cap_node = node.find("figcaption") if isinstance(node, Tag) else None
        caption = self._collapse(self._inline(cap_node)) if cap_node else ""

        self.visuals[vid] = Visual(
            vid=vid,
            src=canonical,
            fetch_url=fullres or displayed or canonical,
            caption=caption,
            alt=(attrs.get("alt") or attrs.get("title") or img.get("alt") or "").strip(),
            width=w,
            height=h,
            hint=hint,
            hero=bool(attrs.get("topImage")),
        )
        self.blocks.append(Block(Kind.VISUAL, vid=vid))

    def _emit_code_wrapper(self, node: Tag) -> None:
        """The language lives in the wrapper's data-attrs; the <pre>'s class
        is just 'shiki'."""
        lang = None
        try:
            lang = json.loads(node.get("data-attrs") or "{}").get("language")
        except (json.JSONDecodeError, TypeError):
            pass
        pre = node.find("pre")
        if pre is not None:
            self._emit_code(pre, lang)

    def _emit_code(self, node: Tag, lang: str | None = None) -> None:
        # Shiki can wrap each line in a <span class="line">.
        line_spans = node.select("span.line")
        code = "\n".join(s.get_text("") for s in line_spans) if line_spans else node.get_text("")
        if lang is None:
            blob = " ".join(
                (node.get("class") or []) + ((node.code.get("class") or []) if node.code else [])
            )
            m = re.search(r"language-([a-z0-9+#]+)", blob, re.I)
            lang = m.group(1) if m else None
        self.blocks.append(
            Block(
                Kind.CODE,
                lang=lang,
                text=code,
                lines=len([ln for ln in code.splitlines() if ln.strip()]),
            )
        )

    def _emit_table(self, node: Tag) -> None:
        rows: list[list[str]] = []
        for tr in node.find_all("tr"):
            cells = [self._collapse(self._inline(td)) for td in tr.find_all(("th", "td"))]
            if any(cells):
                rows.append(cells)
        if rows:
            self.blocks.append(Block(Kind.TABLE, rows=rows))

    def _emit_quote(self, node: Tag) -> None:
        text = self._collapse(self._inline(node))
        if text:
            self.blocks.append(Block(Kind.QUOTE, text=text, note_ids=self._drain_notes()))

    def _emit_list(self, node: Tag) -> None:
        items = [self._collapse(self._inline(li)) for li in node.find_all("li", recursive=False)]
        items = [i for i in items if i]
        if items:
            self.blocks.append(
                Block(Kind.LIST, text="\n".join(items), note_ids=self._drain_notes())
            )

    # -- inline serialization ---------------------------------------------

    def _inline(self, node) -> str:
        """Anchors reduced to just the useful text, notes collected and
        removed from the stream, symbols in <code> marked for the lexicon
        stage."""
        if node is None:
            return ""
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()

        if name == "a":
            href = node.get("href", "")
            if href.startswith("#footnote"):
                self._pending_notes.append(href.split("-")[-1])
                return ""  # [3] mid-sentence breaks prosody
            return node.get_text(" ", strip=True)  # anchor text, URL discarded

        if name == "code":
            inner = node.get_text("", strip=True)
            return f"{SYM_OPEN}{inner}{SYM_CLOSE}" if inner else ""

        if name in ("sup", "img", "svg", "button", "script", "style"):
            return ""
        if name == "br":
            return " "

        return "".join(self._inline(c) for c in node.children)

    @staticmethod
    def _collapse(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()

    def _drain_notes(self) -> list[str]:
        ids, self._pending_notes = self._pending_notes, []
        return ids

    # -- notes ---------------------------------------------------------------

    def _inline_notes(self) -> None:
        """Fold each footnote body into the block that cites it.

        The `[3]` marker is dropped at inline-serialization time because a
        number mid-sentence wrecks the prosody, which used to leave the note
        itself with nothing pointing at it and no place in the stream. Appending
        the body to the citing block keeps the note where the author put it,
        after the sentence that needs it, and costs no extra structure: the
        block count is unchanged, so every per-block shard still lines up.

        The body is appended as its own sentence, with no introducing label:
        any word that could introduce it ("Note:") would have to be written in
        some language, and the note reads perfectly well as a sentence that
        follows the one citing it.
        """
        for blk in self.blocks:
            bodies = [self.notes[nid] for nid in blk.note_ids if self.notes.get(nid)]
            if not bodies:
                continue
            head = blk.text if blk.text.endswith((".", "!", "?", ":", ";")) else blk.text + "."
            blk.text = " ".join([head, *bodies]).strip()

def extract(html: str) -> Document:
    return Extractor(html).run()
