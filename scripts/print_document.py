"""Dev tool: pretty-print the Document produced by extract(), to inspect
block ordering, visual placement, and what got dropped as boilerplate."""

import sys

from substack_feed.ingestion.html_parser import Document, Kind, extract


def _truncate(text: str, width: int = 1) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def print_document(doc: Document) -> None:
    for i, b in enumerate(doc.blocks):
        if b.kind is Kind.HEADING:
            print(f"[{i:>3}] HEADING h{b.level}: {_truncate(b.text)}")
        elif b.kind is Kind.PROSE:
            notes = f" notes={b.note_ids}" if b.note_ids else ""
            deictic = f" deictic={b.deictic}" if b.deictic else ""
            print(f"[{i:>3}] PROSE: {_truncate(b.text)}{notes}{deictic}")
        elif b.kind is Kind.VISUAL:
            v = doc.visuals.get(b.vid)
            if v is None:
                print(f"[{i:>3}] VISUAL vid={b.vid} <missing from registry>")
            else:
                print(
                    f"[{i:>3}] VISUAL vid={v.vid} hint={v.hint} hero={v.hero} "
                    f"klass={v.klass} size={v.width}x{v.height} "
                    f"caption={_truncate(v.caption, 50)!r}"
                )
        elif b.kind is Kind.CODE:
            print(f"[{i:>3}] CODE lang={b.lang} lines={b.lines}")
        elif b.kind is Kind.TABLE:
            rows = len(b.rows)
            cols = len(b.rows[0]) if b.rows else 0
            print(f"[{i:>3}] TABLE {rows}x{cols}")
        elif b.kind is Kind.QUOTE:
            print(f"[{i:>3}] QUOTE: {_truncate(b.text)}")
        elif b.kind is Kind.LIST:
            items = b.text.split("\n")
            print(f"[{i:>3}] LIST ({len(items)} items): {_truncate(' | '.join(items))}")
        else:
            print(f"[{i:>3}] {b.kind}: {_truncate(b.text)}")

    if doc.dropped:
        print("\n-- dropped (boilerplate selectors) --")
        for selector, count in doc.dropped.items():
            print(f"  {selector}: {count}")

    if doc.truncated_at:
        print(f"\n-- truncated at heading: {doc.truncated_at!r} --")

    print(f"\n-- {len(doc.blocks)} blocks, {len(doc.visuals)} visuals, {len(doc.notes)} notes --")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("uso: print_document.py <path_html>")
    html = open(sys.argv[1], encoding="utf-8").read()
    print_document(extract(html))


if __name__ == "__main__":
    main()
