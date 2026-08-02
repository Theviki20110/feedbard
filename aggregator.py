from xml.dom.minidom import Document
from cleaning import VIS_TOKEN, Kind

class IntegrityError(RuntimeError):
    pass


def to_placeholder_stream(doc: Document) -> str:
    """Text stream with placeholders. This is what you pass to the cleaning
    and lexicon stage, not the raw text."""
    parts: list[str] = []
    for b in doc.blocks:
        if b.kind is Kind.VISUAL and b.vid:
            parts.append(VIS_TOKEN.format(vid=b.vid))
        elif b.kind is Kind.CODE:
            parts.append(f"[CODE:{b.lang or 'plain'}:{b.lines}]")
        elif b.kind is Kind.TABLE:
            parts.append(f"[TABLE:{len(b.rows)}x{len(b.rows[0]) if b.rows else 0}]")
        elif b.text:
            parts.append(b.text)
    return "\n\n".join(parts)

def aggregate_data(doc: Document) -> str:
    """Placeholder stream for the doc, ready for translation."""
    return to_placeholder_stream(doc)