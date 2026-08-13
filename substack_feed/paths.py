"""Shared filesystem locations. Single source of truth for the assets dir,
so moving a module doesn't break its `__file__`-relative prompt paths."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"


def safe_filename(title: str) -> str:
    """Filesystem-safe stem shared by every shard/output keyed on an item's
    title, so checkpoints from different pipeline stages line up on disk."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:150]
