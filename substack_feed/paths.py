"""Single source of truth for every filesystem location the pipeline uses.

Layout, rooted at DATA_DIR:

    data/
      episodes/            finished MP3, one per article
      covers/              episode artwork, one per article
      figures/             article images, fetched for description
      shards/
        text/              translated text, per block
        speech/            speech-ready text, per block
        audio/             synthesized audio, per block
        visual/            image classification + description, per image

The split follows the lifetime of the artefacts, not the stage that writes
them. `episodes/` and `covers/` are the finished goods. `shards/` is
resumable scratch -- deleting any of it costs money to rebuild but loses
nothing. `figures/` sits between the two: an input cache, safe to delete,
expensive to refetch.

Episodes, covers, and text/speech/audio shards are all keyed on the same
`slug()` of the article title, so an episode's artefacts can be found across
stages without a lookup table. Figures and visual shards are keyed on the
image content hash instead, which is what lets the same figure reused across
two articles be described once.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# --- deliverable: what the podcast container consumes ----------------------
EPISODES_DIR = DATA_DIR / "episodes"
COVERS_DIR = DATA_DIR / "covers"

# --- inputs ---------------------------------------------------------------
FIGURES_DIR = DATA_DIR / "figures"

# --- resumable intermediates ----------------------------------------------
SHARDS_DIR = DATA_DIR / "shards"
TEXT_SHARDS_DIR = SHARDS_DIR / "text"
SPEECH_SHARDS_DIR = SHARDS_DIR / "speech"
AUDIO_SHARDS_DIR = SHARDS_DIR / "audio"
VISUAL_SHARDS_DIR = SHARDS_DIR / "visual"

ALL_DIRS = (
    EPISODES_DIR,
    COVERS_DIR,
    FIGURES_DIR,
    TEXT_SHARDS_DIR,
    SPEECH_SHARDS_DIR,
    AUDIO_SHARDS_DIR,
    VISUAL_SHARDS_DIR,
)


def slug(title: str) -> str:
    """Filesystem-safe stem shared by every artefact keyed on an article, so
    checkpoints from different pipeline stages line up on disk and an episode
    can be matched to its cover by name alone."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:150]


def ensure_dirs() -> None:
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def episode_path(title: str) -> Path:
    return EPISODES_DIR / f"{slug(title)}.mp3"


def cover_path(title: str, ext: str = ".jpg") -> Path:
    return COVERS_DIR / f"{slug(title)}{ext}"


def find_cover(title: str) -> Path | None:
    """Covers keep the source image's extension, so the lookup is by stem."""
    stem = slug(title)
    for path in sorted(COVERS_DIR.glob(f"{stem}.*")):
        return path
    return None


def figure_path(vid: str, ext: str = ".bin") -> Path:
    return FIGURES_DIR / f"{vid}{ext}"


def find_figure(vid: str) -> Path | None:
    for path in sorted(FIGURES_DIR.glob(f"{vid}.*")):
        return path
    return None


def text_shard_path(title: str, index: int) -> Path:
    return TEXT_SHARDS_DIR / f"{slug(title)}_block{index}.txt"


def speech_shard_path(title: str, index: int) -> Path:
    return SPEECH_SHARDS_DIR / f"{slug(title)}_block{index}.txt"


def audio_shard_path(title: str, index: int) -> Path:
    return AUDIO_SHARDS_DIR / f"{slug(title)}_block{index}.wav"


def visual_shard_path(vid: str) -> Path:
    return VISUAL_SHARDS_DIR / f"{vid}.json"
