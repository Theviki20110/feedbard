import os
import re
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from dotenv import load_dotenv

from extraction import get_text_from_html, html_to_text

load_dotenv()

FEEDS_LIST_PATH = os.environ["FEEDS_LIST_PATH"]
IMAGES_DIR = os.environ["IMAGES_DIR"]

def read_feed_urls(path: str = FEEDS_LIST_PATH) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def get_post_entries(feed_url: str) -> list[dict]:
    r = requests.get(feed_url, headers={"User-Agent": "substack-podcast/1.0"}, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    channel = root.find("channel")
    entries = []
    for item in channel.findall("item"):
        link = item.findtext("link")
        if not link:
            continue
        enclosure = item.find("enclosure")
        image_url = enclosure.get("url") if enclosure is not None else None
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

def get_feeds(entries: list[dict]) -> list[dict]:
    items = []
    for entry in entries:
        content, metadata = get_text_from_html(entry["url"])
        items.append({
            "url": entry["url"],
            "title": metadata,
            "text": content,
            "image_path": save_image(entry["image_url"]),
        })
    return items

if __name__ == "__main__":
    all_entries = [e for url in read_feed_urls() for e in get_post_entries(url)]
    feeds = get_feeds(all_entries)
    for feed in feeds:
        print(f"URL: {feed['url']}")
        print(f"Title: {feed['title']}")
        print(f"Text: {feed['text'][:100]}...")  # Print first 100 characters of text
        print("-" * 40)