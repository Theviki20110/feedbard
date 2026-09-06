from dotenv import load_dotenv

from feedbard.feeds.reader import get_feeds, get_post_entries_v2, read_feed_urls
from feedbard.feeds.store import filter_new_posts, init_db, mark_post_seen
from feedbard.logger import logger
from feedbard.pipeline.orchestrator import process_feeds

load_dotenv()

# One run processes at most this many new posts per feed, so a feed that
# dumps a dozen new entries at once doesn't turn a single cron tick into an
# hours-long LLM/TTS backlog. Posts left over are still new next run, since
# only the ones actually processed get marked seen.
MAX_POSTS_PER_RUN = 5


def check_and_run() -> None:
    init_db()
    for feed_url in read_feed_urls():
        try:
            _check_and_run_feed(feed_url)
        except Exception:
            # One broken feed (unreachable, or a post that jams the pipeline)
            # must not stop every feed listed after it in the same run.
            logger.error("[%s] feed run failed, skipping to next feed", feed_url, exc_info=True)


def _check_and_run_feed(feed_url: str) -> None:
    entries = get_post_entries_v2(feed_url)
    new_urls = set(filter_new_posts(feed_url, [e["url"] for e in entries]))
    new_entries = [e for e in entries if e["url"] in new_urls][:MAX_POSTS_PER_RUN]

    if not new_entries:
        logger.info("[%s] nessun nuovo post", feed_url)
        return

    logger.info("[%s] %d nuovo/i post trovato/i, avvio pipeline", feed_url, len(new_entries))
    feeds = get_feeds(new_entries)
    # get_feeds marks a post whose fetch failed with None; it never reaches
    # the pipeline and is left unseen, so it is retried next run.
    fetched = [(e, f) for e, f in zip(new_entries, feeds, strict=True) if f is not None]
    if not fetched:
        return
    fetched_entries, fetched_feeds = zip(*fetched, strict=True)

    results = process_feeds(list(fetched_feeds))
    for entry, dest in zip(fetched_entries, results, strict=True):
        # Only a post that made it all the way through gets marked seen;
        # one that failed mid-pipeline is retried next run instead of being
        # dropped on the floor.
        if dest is not None:
            mark_post_seen(feed_url, entry["url"])


if __name__ == "__main__":
    check_and_run()
