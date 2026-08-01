from feed_reader import read_feed_urls, get_post_entries, get_feeds
from post_store import init_db, filter_new_posts, mark_post_seen
from processing import process_feeds


ONLY_FIRST_FEED = True  # TEMP: process only 1st feed. Remove this flag + filter below to restore full run.


def check_and_run() -> None:
    init_db()
    feed_urls = read_feed_urls()
    if ONLY_FIRST_FEED:
        feed_urls = feed_urls[:1]
    for feed_url in feed_urls:
        entries = get_post_entries(feed_url)
        new_urls = set(filter_new_posts(feed_url, [e["url"] for e in entries]))
        new_entries = [e for e in entries if e["url"] in new_urls]

        if not new_entries:
            print(f"[{feed_url}] nessun nuovo post")
            continue

        print(f"[{feed_url}] {len(new_entries)} nuovo/i post trovato/i, avvio pipeline")
        feeds = get_feeds(new_entries)
        process_feeds(feeds)

        for entry in new_entries:
            mark_post_seen(feed_url, entry["url"])


if __name__ == "__main__":
    check_and_run()
