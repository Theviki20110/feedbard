import os
from dotenv import load_dotenv
from substack_feed.feeds.reader import read_feed_urls, get_post_entries_v2, get_feeds
from substack_feed.feeds.store import init_db, filter_new_posts, mark_post_seen
from substack_feed.logger import logger
from substack_feed.pipeline.orchestrator import process_feeds

load_dotenv()

ONLY_FIRST_FEED = os.environ["ONLY_FIRST_FEED"].lower() == "true"  # TEMP: process only 1st feed. Set to false to restore full run.

def check_and_run() -> None:
    init_db()
    feed_urls = read_feed_urls()
    if ONLY_FIRST_FEED:
        feed_urls = feed_urls[:1]
    for feed_url in feed_urls:
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
