import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from feedbard.ingestion.fetcher import fetch_article
from feedbard.logger import logger
from feedbard.paths import COVERS_DIR, cover_path, find_cover, slug

load_dotenv()

FEEDS_LIST_PATH = os.environ["FEEDS_LIST_PATH"]


def read_feed_urls(path: str = FEEDS_LIST_PATH) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _extract_og_image(post_url: str) -> str | None:
    try:
        r = requests.get(post_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return None
    soup = BeautifulSoup(r.content, "html.parser")
    tag = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    return tag.get("content") if tag else None


def _entry_image(item: dict) -> str | None:
    """Artwork as the feed itself declares it, in decreasing order of intent:
    an explicit media element, then an enclosure, then whatever the page
    advertises to social networks."""
    for media in item.get("media_content", []) or []:
        if media.get("url"):
            return media["url"]
    for thumb in item.get("media_thumbnail", []) or []:
        if thumb.get("url"):
            return thumb["url"]
    for link_obj in item.get("links", []) or []:
        if link_obj.get("rel") == "enclosure" and link_obj.get("href"):
            return link_obj["href"]
    return None


def _entry_content(item: dict) -> str:
    """The post body as carried by the feed, if it is carried at all.

    `content` (content:encoded) is the full post where a publisher offers one;
    `summary` is a teaser far more often than not. Both are handed to the
    fetcher, which decides on length whether what it got is the whole article.
    """
    for block in item.get("content", []) or []:
        if block.get("value"):
            return block["value"]
    return item.get("summary") or ""


def _entry_author(item: dict, feed_meta: dict) -> str:
    """Per-post byline first (`dc:creator` lands in `author`), then the feed's
    own author, then nothing -- the fetcher falls back to the host."""
    for detail in (item.get("author_detail") or {}, feed_meta.get("author_detail") or {}):
        name = (detail.get("name") or "").strip()
        # feedparser puts the address in `name` when the field is bare email.
        if name and "@" not in name:
            return name
    author = (item.get("author") or "").strip()
    return "" if "@" in author else author


def get_post_entries_v2(feed_url: str) -> list[dict]:
    """Parse a feed into entries carrying everything the feed already knows.

    Title, byline, date and (for untruncated feeds) the body all come free
    with the poll. Passing them downstream is what lets the generic path skip
    a per-post request that only Substack ever needed.
    """
    r = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    feed_meta = feed.feed
    generator = feed_meta.get("generator") or ""

    entries = []
    for item in feed.entries:
        link = item.get("link")
        if not link:
            continue
        image_url = _entry_image(item) or _extract_og_image(link)
        entries.append(
            {
                "url": link,
                "image_url": image_url,
                "title": (item.get("title") or "").strip(),
                "author": _entry_author(item, feed_meta),
                "published_at": item.get("published") or item.get("updated") or "",
                "content_html": _entry_content(item),
                "generator": generator,
                "feed_title": (feed_meta.get("title") or "").strip(),
            }
        )
    return entries


def save_cover(image_url: str, title: str) -> str | None:
    """Store the post's artwork under the article slug.

    Naming it after the article, rather than after the remote file, is what
    lets the podcast container pair an episode with its cover: both sides are
    derived from the same title, so no lookup table has to be kept in sync.
    """
    if not image_url or not title:
        return None

    existing = find_cover(title)
    if existing is not None:
        return str(existing)

    ext = os.path.splitext(urlparse(image_url).path)[1] or ".jpg"
    dest_path = cover_path(title, ext)

    try:
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        # A missing cover costs the episode its artwork, not its audio.
        logger.warning("cover fetch failed for %s", image_url, exc_info=True)
        return None

    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(r.content)
    return str(dest_path)


def find_posts_by_slug(slugs: set[str]) -> dict[str, dict]:
    """Walk the configured feeds and return metadata for the wanted articles.

    Rendered artefacts are named by slug and carry no author or date, so
    republishing one after the fact means going back to the feed for them.
    The feed entry alone answers that -- no article body is fetched -- and the
    sweep stops as soon as every slug is accounted for, since the common case
    is backfilling a handful of episodes across feeds of hundreds of posts.
    """
    wanted = set(slugs)
    found: dict[str, dict] = {}

    for feed_url in read_feed_urls():
        if not wanted:
            break
        for entry in get_post_entries_v2(feed_url):
            if not wanted:
                break
            key = slug(entry["title"])
            if key not in wanted:
                continue

            found[key] = {
                "url": entry["url"],
                "title": entry["title"],
                "author": entry["author"] or entry["feed_title"] or "Unknown",
                "published_at": entry["published_at"],
                "image_url": entry["image_url"],
            }
            wanted.discard(key)

    return found


def _build_feed_item(entry: dict) -> dict:
    article = fetch_article(entry)
    title = article.title or entry["title"]
    # The article's own artwork beats the feed's, which is a fallback for
    # publishers that do not expose a per-post cover.
    image_url = article.cover_image or entry["image_url"]
    return {
        "url": entry["url"],
        "title": title,
        "text": article.html,
        "author": article.author,
        "subtitle": article.subtitle,
        "published_at": article.published_at,
        "cover_path": save_cover(image_url, title),
    }


def get_feeds(entries: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(_build_feed_item, entries))
