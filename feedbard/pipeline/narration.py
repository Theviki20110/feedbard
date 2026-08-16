"""Puts the non-prose blocks back into the narration.

`extract()` leaves a VISUAL block with no text at all and a TABLE block with
its grid but no prose. Neither is something a speech engine can read, so
without this stage both are positions in the article the listener never
learns about -- and the vision pass runs, is billed, and its output goes
nowhere. This module is the join: it takes what the describers produced and
writes it into the block that stood there.

The description is written to `translated_text`, not to `text`, for two
reasons. The describers already write in the narration language, so the
translation stage has nothing to do here and skips these blocks. The
sanitizer, however, still runs over them: a description that reads a formula
out loud is exactly the kind of text that arrives full of symbols.

Nothing is inserted or removed -- every description lands in a block the
parser already emitted -- so block indices, and with them every per-block
shard on disk, stay aligned with what earlier runs wrote.
"""

from __future__ import annotations

from feedbard.ingestion.html_parser import Document, Kind
from feedbard.pipeline.visual_describer import SKIPPED_KLASS


def attach_descriptions(doc: Document) -> int:
    """Materialise visual and table descriptions as narratable text.

    Returns the number of blocks that gained a voice. Call after
    `describe_visuals` and `describe_tables`, before `translate_blocks`.
    """
    attached = 0
    for block in doc.blocks:
        if block.kind is Kind.VISUAL:
            visual = doc.visuals.get(block.vid or "")
            # A decorativo visual is a banner or a logo: the vision pass judged
            # it to carry nothing the article doesn't already say, so narrating
            # it would only interrupt the prose.
            if visual is None or visual.klass == SKIPPED_KLASS:
                continue
            text = (visual.description or "").strip()
        elif block.kind is Kind.TABLE:
            text = (block.description or "").strip()
        else:
            continue

        if not text:
            continue
        block.translated_text = text
        attached += 1

    return attached
