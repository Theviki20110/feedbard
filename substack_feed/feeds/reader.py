import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from substack_feed.ingestion.substack_fetcher import get_text_from_html
from substack_feed.logger import logger
from substack_feed.paths import COVERS_DIR, cover_path, find_cover

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


def get_post_entries_v2(feed_url: str) -> list[dict]:
    r = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    entries = []
    for item in feed.entries:
        link = item.get("link")
        if not link:
            continue
        image_url = None
        for media in item.get("media_content", []):
            if media.get("url"):
                image_url = media["url"]
                break
        if not image_url and item.get("links"):
            for link_obj in item["links"]:
                if link_obj.get("rel") == "enclosure" and link_obj.get("href"):
                    image_url = link_obj["href"]
                    break
        if not image_url:
            image_url = _extract_og_image(link)
        entries.append({"url": link, "image_url": image_url})
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


def _build_feed_item(entry: dict) -> dict:
    content, metadata = get_text_from_html(entry["url"])
    title = metadata["title"]
    return {
        "url": entry["url"],
        "title": title,
        "text": content,
        "cover_path": save_cover(entry["image_url"], title),
    }


def get_feeds(entries: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(_build_feed_item, entries))
