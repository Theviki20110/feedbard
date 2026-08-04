"""Shared filesystem locations. Single source of truth for the assets dir,
so moving a module doesn't break its `__file__`-relative prompt paths."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
