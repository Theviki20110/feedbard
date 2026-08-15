"""Assembles the Audiobookshelf library from the pipeline's own artefacts.

Audiobookshelf reads a book library as `{Author}/{Title}/`, taking the author
and title from the folder names and the cover from an image file sitting
beside the audio. That is the only filesystem layout that gives each article
its own artwork -- a podcast library allows exactly one cover for the whole
feed -- which is why articles are published as books here.

Copies rather than moves, so DATA_DIR remains the pipeline's state of record
and republishing never depends on what the media server did to its library.
"""

import shutil
from pathlib import Path

from mutagen.id3 import APIC, ID3, TALB, TDRC, TIT2, TPE1, TPE2
from mutagen.mp3 import MP3

from substack_feed.logger import logger
from substack_feed.paths import (
    LIBRARY_DIR,
    episode_path,
    find_cover,
    item_dir,
    library_name,
    published_cover_path,
    published_episode_path,
)

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def _year_of(published_at: str) -> str:
    """Substack timestamps are ISO-8601; ID3's TDRC wants a plain year."""
    return published_at[:4] if published_at[:4].isdigit() else ""


def write_tags(
    mp3_path: Path,
    title: str,
    author: str,
    published_at: str = "",
    cover: Path | None = None,
) -> None:
    """Tag the episode so the library shows a title instead of a filename.

    Audiobookshelf falls back to the folder name when tags are absent, but
    the concatenated TTS output carries none at all, so anything reading the
    file outside ABS would see it as untitled.
    """
    audio = MP3(mp3_path, ID3=ID3)
    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags
    tags.delall("TIT2")
    tags.delall("TPE1")
    tags.delall("TPE2")
    tags.delall("TALB")
    tags.delall("TDRC")
    tags.add(TIT2(encoding=3, text=title))
    tags.add(TPE1(encoding=3, text=author))
    tags.add(TPE2(encoding=3, text=author))
    # ABS groups by album; one album per article keeps each item self-contained.
    tags.add(TALB(encoding=3, text=title))
    year = _year_of(published_at)
    if year:
        tags.add(TDRC(encoding=3, text=year))

    if cover is not None and cover.suffix.lower() in _MIME:
        tags.delall("APIC")
        tags.add(
            APIC(
                encoding=3,
                mime=_MIME[cover.suffix.lower()],
                type=3,  # front cover
                desc="Cover",
                data=cover.read_bytes(),
            )
        )
    audio.save(v2_version=3)  # v2.3 is what the widest set of players reads


def publish_item(
    title: str,
    author: str,
    published_at: str = "",
    overwrite: bool = False,
) -> Path | None:
    """Copy one finished episode and its cover into the library tree."""
    source = episode_path(title)
    if not source.exists():
        logger.warning("[%s] publish: no episode at %s", title, source)
        return None

    dest = published_episode_path(author, title)
    if dest.exists() and not overwrite:
        logger.info("[%s] publish: already in library, skipping", title)
        return dest

    item_dir(author, title).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

    cover = find_cover(title)
    dest_cover = None
    if cover is not None:
        ext = cover.suffix.lower() if cover.suffix.lower() in _MIME else ".jpg"
        dest_cover = published_cover_path(author, title, ext)
        shutil.copy2(cover, dest_cover)
    else:
        logger.warning("[%s] publish: no cover found, item will use ABS defaults", title)

    try:
        write_tags(dest, title, author, published_at, dest_cover)
    except Exception:
        # A tagless file still plays and still shows up under the right
        # folder name; losing the whole publish over metadata would be worse.
        logger.warning("[%s] publish: tagging failed", title, exc_info=True)

    logger.info("[%s] publish: %s", title, dest)
    return dest


def publish(item: dict, overwrite: bool = False) -> Path | None:
    """Publish from a feed item as built by feeds.reader."""
    title = item.get("title") or ""
    author = item.get("author") or "Unknown"
    if not title:
        logger.warning("publish: item without a title, skipping")
        return None
    return publish_item(title, author, item.get("published_at", ""), overwrite)


def library_summary() -> str:
    if not LIBRARY_DIR.exists():
        return f"{LIBRARY_DIR} (empty)"
    items = sorted(p for p in LIBRARY_DIR.glob("*/*") if p.is_dir())
    lines = [f"{LIBRARY_DIR} - {len(items)} item(s)"]
    lines += [f"  {p.parent.name} / {p.name}" for p in items]
    return "\n".join(lines)


__all__ = ["publish", "publish_item", "write_tags", "library_name", "library_summary"]
