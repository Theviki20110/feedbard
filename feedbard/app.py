from dotenv import load_dotenv

from feedbard.feeds.reader import get_feeds, get_post_entries_v2, read_feed_urls
from feedbard.feeds.store import filter_new_posts, init_db, mark_post_seen
from feedbard.logger import logger
from feedbard.pipeline.orchestrator import process_feeds

load_dotenv()


def check_and_run() -> None:
    init_db()
    for feed_url in read_feed_urls():
        entries = get_post_entries_v2(feed_url)
        new_urls = set(filter_new_posts(feed_url, [e["url"] for e in entries]))
        new_entries = [e for e in entries if e["url"] in new_urls]

        if not new_entries:
            logger.info("[%s] nessun nuovo post", feed_url)
            continue

        logger.info("[%s] %d nuovo/i post trovato/i, avvio pipeline", feed_url, len(new_entries))
        feeds = get_feeds(new_entries)
        process_feeds(feeds)

        for entry in new_entries:
            mark_post_seen(feed_url, entry["url"])


if __name__ == "__main__":
    check_and_run()
