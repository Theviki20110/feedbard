import os
import re
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from substack_feed.ingestion.substack_fetcher import get_text_from_html

load_dotenv()

FEEDS_LIST_PATH = os.environ["FEEDS_LIST_PATH"]
IMAGES_DIR = os.environ["IMAGES_DIR"]

def read_feed_urls(path: str = FEEDS_LIST_PATH) -> list[str]:
    with open(path, "r") as f:
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
            for l in item["links"]:
                if l.get("rel") == "enclosure" and l.get("href"):
                    image_url = l["href"]
                    break
        if not image_url:
            image_url = _extract_og_image(link)
        entries.append({"url": link, "image_url": image_url})
    return entries

def save_image(image_url: str, dest_dir: str = IMAGES_DIR) -> str | None:
    if not image_url:
        return None
    os.makedirs(dest_dir, exist_ok=True)
    ext = os.path.splitext(urlparse(image_url).path)[1] or ".jpg"
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", os.path.basename(urlparse(image_url).path)) or "image"
    if not filename.endswith(ext):
        filename += ext
    dest_path = os.path.join(dest_dir, filename)

    r = requests.get(image_url, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path

def _build_feed_item(entry: dict) -> dict:
    content, metadata = get_text_from_html(entry["url"])
    return {
        "url": entry["url"],
        "title": metadata["title"],
        "text": content,
        "image_path": save_image(entry["image_url"]),
    }

def get_feeds(entries: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(_build_feed_item, entries))