"""Publish already-rendered episodes into the Audiobookshelf library tree.

The pipeline publishes each item as it finishes; this is for backfilling
episodes rendered before publishing existed, or for republishing after a
cover or a title changed.

Author and date are looked up from the feeds, since the rendered artefacts
carry neither. Pass --author to skip the lookup, or when the post has fallen
off the feed.

    uv run python scripts/publish.py --list
    uv run python scripts/publish.py --all
    uv run python scripts/publish.py Notes_on_Midtraining --overwrite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedbard.feeds.reader import find_posts_by_slug  # noqa: E402
from feedbard.paths import EPISODES_DIR, find_cover  # noqa: E402
from feedbard.pipeline.publisher import library_summary, publish_item  # noqa: E402


def available() -> list[str]:
    """Episode stems on disk. These are slugs, so underscores stand in for
    every character the filesystem would not take -- most often a space."""
    return sorted(p.stem for p in EPISODES_DIR.glob("*.mp3"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*", help="episode stems to publish (see --list)")
    parser.add_argument("--all", action="store_true", help="publish every rendered episode")
    parser.add_argument("--author", default="", help="skip the feed lookup and use this author")
    parser.add_argument("--overwrite", action="store_true", help="replace an item already there")
    parser.add_argument("--list", action="store_true", help="show what is on disk, publish nothing")
    args = parser.parse_args()

    if args.list:
        print("Rendered episodes:")
        for stem in available():
            print(f"  {stem}   cover={'yes' if find_cover(stem) else 'NO'}")
        print()
        print(library_summary())
        return

    targets = available() if args.all else args.slugs
    if not targets:
        parser.error("pass one or more episode stems, or --all (see --list)")

    # Without the real title the library folder would read "Notes_on_Midtraining".
    posts = {} if args.author else find_posts_by_slug(set(targets))

    for stem in targets:
        post = posts.get(stem)
        if post is None:
            if not args.author:
                print(f"  ! {stem}: not found in any feed, pass --author to publish it anyway")
                continue
            publish_item(stem, args.author, "", overwrite=args.overwrite)
        else:
            publish_item(
                post["title"],
                args.author or post["author"],
                post["published_at"],
                overwrite=args.overwrite,
            )

    print()
    print(library_summary())


if __name__ == "__main__":
    main()
