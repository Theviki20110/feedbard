"""Feed entry -> article HTML + normalised metadata.

Everything downstream of this module works on an HTML fragment and a flat
metadata dict, so a feed is only ever as special as the way its content has
to be obtained. That difference lives here, as an ordered list of strategies:

1. `_from_substack` -- Substack's own API. It is the only source that hands
   back the subtitle, the post's own cover image, and the byline as a name
   rather than an email, so it wins for the hosts it covers.
2. `_from_feed` -- the entry's `content:encoded`, when the publisher puts the
   whole post in the feed (Ghost, WordPress, most static-site generators).
   Costs no extra request.
3. `_from_page` -- fetch the page and run readability over it. The universal
   fallback: works on a truncated feed, knows nothing about the publisher.
   The fetched HTML is cached to `raw_html/` (see `feedbard.paths`) before
   readability ever sees it, and concurrent requests to one host are capped
   and retried on 429 -- see `_get`/`_get_page_html` below.

The first strategy that both applies and returns a usable body wins; a
strategy that raises is logged and skipped, so one publisher changing its
markup degrades that feed to the next strategy instead of failing the run.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument

from feedbard.logger import logger
from feedbard.paths import raw_html_path

USER_AGENT = "Mozilla/5.0 (compatible; feedbard/1.0; +https://github.com/)"
TIMEOUT = 30

# Hosts served by Substack itself. Publications on a custom domain are not
# matched by name, so they are detected from the feed's <generator> instead.
SUBSTACK_HOSTS = (".substack.com",)

# A truncated feed still carries a teaser in content:encoded, so length is
# what separates "the whole post" from "the first paragraph and a link".
# Two short paragraphs of text; anything under that goes to the page fetch.
MIN_FULL_TEXT = 1200

# `get_feeds` fetches a feed's entries on a shared thread pool, and every
# entry in one feed shares a host: without a cap, a burst of new posts opens
# that many concurrent connections to one publisher, which is what a WAF's
# rate limit tends to key on. Kept low and configurable rather than derived
# from the pool size, since the failure mode (429) is per-host, not global.
PER_HOST_CONCURRENCY = int(os.getenv("PER_HOST_CONCURRENCY", "2"))

# A 429 during a burst of new posts is not a reason to fail the article: a
# short wait is cheap next to re-running the LLM stages that follow.
MAX_429_RETRIES = 3

_host_semaphores: dict[str, threading.Semaphore] = defaultdict(
    lambda: threading.Semaphore(PER_HOST_CONCURRENCY)
)
_host_semaphores_lock = threading.Lock()


def _host_semaphore(url: str) -> threading.Semaphore:
    host = urlparse(url).hostname or ""
    with _host_semaphores_lock:
        return _host_semaphores[host]


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    value = response.headers.get("Retry-After")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return 2**attempt


@dataclass
class Article:
    """What every strategy has to produce, whatever it had to start from."""

    html: str
    title: str = ""
    author: str = ""
    subtitle: str = ""
    published_at: str = ""
    cover_image: str = ""


def _get(url: str) -> requests.Response:
    with _host_semaphore(url):
        for attempt in range(MAX_429_RETRIES + 1):
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            if r.status_code != 429 or attempt == MAX_429_RETRIES:
                r.raise_for_status()
                return r
            wait = _retry_after_seconds(r, attempt)
            logger.warning(
                "429 from %s, retrying in %.0fs (attempt %d/%d)",
                urlparse(url).hostname,
                wait,
                attempt + 1,
                MAX_429_RETRIES,
            )
            time.sleep(wait)


def _get_page_html(url: str) -> str:
    """Cached ahead of `_get`: a page fetched once is never requested again,
    so a 429 from an earlier crash never blocks a resumed run from the pages
    it had already fetched."""
    cache_path = raw_html_path(url)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    page = _get(url).text
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(page, encoding="utf-8")
    return page


def _text_length(html: str) -> int:
    return len(BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True))


def _host_author(url: str) -> str:
    """Last-resort byline: the publication host. Keeps the library grouped
    under something a listener recognises instead of an empty folder."""
    host = urlparse(url or "").hostname or ""
    host = host.removeprefix("www.")
    return host.split(".")[0].replace("-", " ").title() if host else "Unknown"


def is_substack(entry: dict) -> bool:
    host = (urlparse(entry.get("url") or "").hostname or "").lower()
    if host.endswith(SUBSTACK_HOSTS):
        return True
    return "substack" in (entry.get("generator") or "").lower()


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def _from_substack(entry: dict) -> Article | None:
    if not is_substack(entry):
        return None

    # Imported lazily: a non-Substack deployment should not pay for the
    # dependency's import, and a broken release of it should not stop the run.
    from substack_api import Post

    post = Post(entry["url"])
    metadata = post.get_metadata()
    content = post.get_content()
    if not content:
        return None

    return Article(
        html=content,
        title=(metadata.get("title") or entry.get("title") or "").strip(),
        author=_substack_author(metadata) or entry.get("author") or _host_author(entry["url"]),
        subtitle=(metadata.get("subtitle") or "").strip(),
        published_at=metadata.get("post_date") or entry.get("published_at") or "",
        cover_image=metadata.get("cover_image") or entry.get("image_url") or "",
    )


def _substack_author(metadata: dict) -> str:
    """Substack keeps the author in a bylines list; the post-level `author`
    key is always null."""
    for byline in metadata.get("publishedBylines") or []:
        name = (byline.get("name") or "").strip()
        if name:
            return name
    return ""


def _from_feed(entry: dict) -> Article | None:
    html = entry.get("content_html") or ""
    if _text_length(html) < MIN_FULL_TEXT:
        return None
    return _article_from_entry(entry, html)


def _from_page(entry: dict) -> Article | None:
    page = _get_page_html(entry["url"])

    body = ReadabilityDocument(page).summary(html_partial=True)
    if _text_length(body) < 200:
        # Readability found nothing worth reading: better to fall through to
        # whatever the feed had than to narrate a cookie banner.
        return None

    meta = _page_metadata(page)
    return _article_from_entry(entry, body, page_meta=meta)


def _page_metadata(page: str) -> dict:
    """Open Graph first, then the plain HTML tags. Nothing here is required:
    the feed entry already supplies a title, a date and usually an author."""
    soup = BeautifulSoup(page, "html.parser")

    def prop(*names: str) -> str:
        for name in names:
            tag = soup.find("meta", property=name) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return ""

    title = prop("og:title", "twitter:title")
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    return {
        "title": title,
        "author": prop("author", "article:author", "twitter:creator"),
        "subtitle": prop("og:description", "description"),
        "published_at": prop("article:published_time", "og:article:published_time"),
        "cover_image": prop("og:image", "twitter:image"),
    }


def _article_from_entry(entry: dict, html: str, page_meta: dict | None = None) -> Article:
    """Merge the three sources of truth, most trustworthy first: the feed
    entry (authored by the publisher, per post), then the page's own meta
    tags, then a value derived from the URL."""
    meta = page_meta or {}
    author = entry.get("author") or meta.get("author") or ""
    # og:article:author is often a profile URL, which is not a name.
    if author.startswith("http"):
        author = ""
    return Article(
        html=html,
        title=(entry.get("title") or meta.get("title") or "").strip(),
        author=author.strip() or _host_author(entry["url"]),
        subtitle=(meta.get("subtitle") or "").strip(),
        published_at=entry.get("published_at") or meta.get("published_at") or "",
        cover_image=entry.get("image_url") or meta.get("cover_image") or "",
    )


STRATEGIES = (_from_substack, _from_feed, _from_page)


def fetch_article(entry: dict) -> Article:
    """Run the strategies in order and return the first usable article.

    Raises only when every strategy has been exhausted: at that point the post
    has no body to narrate, and skipping it silently would leave it marked as
    processed with nothing to show.
    """
    for strategy in STRATEGIES:
        try:
            article = strategy(entry)
        except Exception:  # noqa: BLE001 - a failing strategy is a fallback, not a crash
            logger.warning(
                "%s failed for %s, trying next strategy",
                strategy.__name__,
                entry.get("url"),
                exc_info=True,
            )
            continue
        if article is not None:
            logger.info("%s: fetched via %s", entry.get("url"), strategy.__name__)
            return article

    raise RuntimeError(f"no strategy could extract an article body from {entry.get('url')}")
