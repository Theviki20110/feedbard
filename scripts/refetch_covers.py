"""Refetch cover art for episodes that are missing it.

Covers downloaded before the layout refactor were named after the remote
file, so the article they belong to cannot be recovered from disk. This walks
the configured feeds, matches each post's title against the rendered
episodes by slug, and saves the artwork under the name the library expects.

    uv run python scripts/refetch_covers.py --dry-run
    uv run python scripts/refetch_covers.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedbard.feeds.reader import (  # noqa: E402
    get_post_entries_v2,
    read_feed_urls,
    save_cover,
)
from feedbard.ingestion.fetcher import fetch_article  # noqa: E402
from feedbard.paths import EPISODES_DIR, find_cover, slug  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report matches, download nothing")
    args = parser.parse_args()

    # Episodes are named by slug, and so is the cover we need to produce, so
    # the slug is the only key the two sides share.
    wanted = {p.stem for p in EPISODES_DIR.glob("*.mp3") if find_cover(p.stem) is None}
    if not wanted:
        print("every episode already has a cover")
        return
    print(f"missing covers: {len(wanted)}")

    found = 0
    for feed_url in read_feed_urls():
        for entry in get_post_entries_v2(feed_url):
            title = entry["title"]
            if slug(title) not in wanted:
                continue

            # The feed's image is enough for most publishers; the article is
            # fetched only for the posts that matched, and only because some
            # sources expose a better per-post cover than the feed does.
            image_url = entry["image_url"]
            try:
                image_url = fetch_article(entry).cover_image or image_url
            except Exception as exc:  # noqa: BLE001 - one bad post must not stop the sweep
                print(f"  ! {entry['url']}: {exc}")

            print(f"  {'would save' if args.dry_run else 'saving'}: {title}")
            if not args.dry_run:
                saved = save_cover(image_url, title)
                print(f"    -> {saved}")
            found += 1
            wanted.discard(slug(title))

    print(f"\nmatched {found}; still missing: {sorted(wanted)}")


if __name__ == "__main__":
    main()
