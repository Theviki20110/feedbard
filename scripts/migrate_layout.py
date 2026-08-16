"""One-shot move of the pre-refactor layout onto the DATA_DIR tree.

Moves rather than copies, but never overwrites and never deletes a source it
did not move: a second run is a no-op, and an interrupted run can be resumed.
Covers are renamed from the remote image's basename to the article slug,
which is what lets an episode find its artwork.

    uv run python scripts/migrate_layout.py --dry-run
    uv run python scripts/migrate_layout.py
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedbard.paths import (  # noqa: E402
    AUDIO_SHARDS_DIR,
    COVERS_DIR,
    EPISODES_DIR,
    SPEECH_SHARDS_DIR,
    TEXT_SHARDS_DIR,
    VISUAL_SHARDS_DIR,
    ensure_dirs,
)

OLD_AUDIO = Path("audio")
OLD_IMAGES = Path("images")

# (source dir, glob, destination dir). Covers are handled separately: their
# filenames come from a remote URL and carry no article identity.
MOVES = [
    (OLD_AUDIO, "*.mp3", EPISODES_DIR),
    (OLD_AUDIO / "text_shards", "*.txt", TEXT_SHARDS_DIR),
    (OLD_AUDIO / "speech_shards", "*.txt", SPEECH_SHARDS_DIR),
    (OLD_AUDIO / "audio_shards", "*.wav", AUDIO_SHARDS_DIR),
    (OLD_AUDIO / "visual_shards", "*.json", VISUAL_SHARDS_DIR),
    (OLD_IMAGES, "*", COVERS_DIR),
]


def migrate(dry_run: bool) -> int:
    if not dry_run:
        ensure_dirs()

    moved = skipped = 0
    for src_dir, pattern, dest_dir in MOVES:
        if not src_dir.is_dir():
            print(f"  (absent) {src_dir}")
            continue
        sources = sorted(p for p in src_dir.glob(pattern) if p.is_file())
        print(f"  {src_dir}/{pattern} -> {dest_dir}  ({len(sources)} file)")
        for src in sources:
            dest = dest_dir / src.name
            if dest.exists():
                skipped += 1
                continue
            if not dry_run:
                shutil.move(str(src), str(dest))
            moved += 1

    print(f"\n{'would move' if dry_run else 'moved'}: {moved}   already present: {skipped}")
    if not dry_run:
        print(
            "\nCovers kept their remote filenames and are NOT yet matched to an "
            "episode slug. Rename them to '<episode stem>.<ext>' by hand, or let "
            "the next pipeline run refetch them under the right name."
        )
    return moved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
    args = parser.parse_args()
    migrate(args.dry_run)
