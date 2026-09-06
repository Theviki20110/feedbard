"""check_and_run wires feeds -> new-post filtering -> the pipeline -> the
seen-posts store. Every dependency is faked, so this only tests that wiring:
what gets skipped, what gets processed, and in what order state is recorded.
"""

from feedbard import app


def _entries(*urls):
    return [{"url": u, "title": u} for u in urls]


def test_feed_with_no_new_posts_is_skipped(monkeypatch):
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(app, "read_feed_urls", lambda: ["https://feed.example/rss"])
    monkeypatch.setattr(app, "get_post_entries_v2", lambda url: _entries("https://a.example/1"))
    monkeypatch.setattr(app, "filter_new_posts", lambda feed_url, urls: [])
    monkeypatch.setattr(app, "get_feeds", _unreachable)
    monkeypatch.setattr(app, "process_feeds", _unreachable)
    monkeypatch.setattr(app, "mark_post_seen", _unreachable)

    app.check_and_run()  # must not raise: nothing downstream should be called


def _unreachable(*a, **kw):
    raise AssertionError("should not have been called: no new posts")


def test_only_the_filtered_new_urls_reach_the_pipeline(monkeypatch):
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(app, "read_feed_urls", lambda: ["https://feed.example/rss"])
    monkeypatch.setattr(
        app,
        "get_post_entries_v2",
        lambda url: _entries("https://a.example/1", "https://a.example/2"),
    )
    monkeypatch.setattr(app, "filter_new_posts", lambda feed_url, urls: ["https://a.example/2"])

    seen_by_get_feeds = []

    def get_feeds(entries):
        seen_by_get_feeds.extend(entries)
        return entries

    monkeypatch.setattr(app, "get_feeds", get_feeds)
    monkeypatch.setattr(app, "process_feeds", lambda feeds: ["dest.mp3"] * len(feeds))
    marked = []
    monkeypatch.setattr(app, "mark_post_seen", lambda feed_url, url: marked.append(url))

    app.check_and_run()

    assert [e["url"] for e in seen_by_get_feeds] == ["https://a.example/2"]
    assert marked == ["https://a.example/2"]


def test_every_new_post_is_marked_seen_after_processing(monkeypatch):
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(app, "read_feed_urls", lambda: ["https://feed.example/rss"])
    monkeypatch.setattr(
        app,
        "get_post_entries_v2",
        lambda url: _entries("https://a.example/1", "https://a.example/2"),
    )
    monkeypatch.setattr(
        app,
        "filter_new_posts",
        lambda feed_url, urls: ["https://a.example/1", "https://a.example/2"],
    )
    monkeypatch.setattr(app, "get_feeds", lambda entries: entries)
    monkeypatch.setattr(app, "process_feeds", lambda feeds: ["dest.mp3"] * len(feeds))
    marked = []
    monkeypatch.setattr(app, "mark_post_seen", lambda feed_url, url: marked.append(url))

    app.check_and_run()

    assert marked == ["https://a.example/1", "https://a.example/2"]


def test_processing_runs_before_marking_posts_seen(monkeypatch):
    # Marking a post seen before it is actually processed would drop it on
    # the floor forever if the run were interrupted in between.
    order = []
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(app, "read_feed_urls", lambda: ["https://feed.example/rss"])
    monkeypatch.setattr(app, "get_post_entries_v2", lambda url: _entries("https://a.example/1"))
    monkeypatch.setattr(app, "filter_new_posts", lambda feed_url, urls: ["https://a.example/1"])
    monkeypatch.setattr(app, "get_feeds", lambda entries: entries)

    def process_feeds(feeds):
        order.append("process_feeds")
        return ["dest.mp3"] * len(feeds)

    monkeypatch.setattr(app, "process_feeds", process_feeds)
    monkeypatch.setattr(
        app, "mark_post_seen", lambda feed_url, url: order.append("mark_post_seen")
    )

    app.check_and_run()

    assert order == ["process_feeds", "mark_post_seen"]


def test_every_configured_feed_is_visited(monkeypatch):
    visited = []
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(
        app, "read_feed_urls", lambda: ["https://a.example/rss", "https://b.example/rss"]
    )

    def get_post_entries_v2(url):
        visited.append(url)
        return []

    monkeypatch.setattr(app, "get_post_entries_v2", get_post_entries_v2)
    monkeypatch.setattr(app, "filter_new_posts", lambda feed_url, urls: [])
    monkeypatch.setattr(app, "get_feeds", _unreachable)
    monkeypatch.setattr(app, "process_feeds", _unreachable)
    monkeypatch.setattr(app, "mark_post_seen", _unreachable)

    app.check_and_run()

    assert visited == ["https://a.example/rss", "https://b.example/rss"]


def test_init_db_runs_before_any_feed_is_read(monkeypatch):
    order = []

    def read_feed_urls():
        order.append("read_feed_urls")
        return []

    monkeypatch.setattr(app, "init_db", lambda: order.append("init_db"))
    monkeypatch.setattr(app, "read_feed_urls", read_feed_urls)

    app.check_and_run()

    assert order == ["init_db", "read_feed_urls"]


def test_a_failed_fetch_does_not_block_the_rest_of_the_batch(monkeypatch):
    # get_feeds marks a post whose fetch failed (every strategy exhausted)
    # with None; it must not stop the other posts in the same feed from
    # being processed and marked seen.
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(app, "read_feed_urls", lambda: ["https://feed.example/rss"])
    monkeypatch.setattr(
        app,
        "get_post_entries_v2",
        lambda url: _entries("https://a.example/1", "https://a.example/2"),
    )
    monkeypatch.setattr(
        app,
        "filter_new_posts",
        lambda feed_url, urls: ["https://a.example/1", "https://a.example/2"],
    )
    monkeypatch.setattr(app, "get_feeds", lambda entries: [None, entries[1]])
    monkeypatch.setattr(app, "process_feeds", lambda feeds: ["dest.mp3"] * len(feeds))
    marked = []
    monkeypatch.setattr(app, "mark_post_seen", lambda feed_url, url: marked.append(url))

    app.check_and_run()

    assert marked == ["https://a.example/2"]


def test_a_failed_pipeline_item_is_not_marked_seen(monkeypatch):
    # process_feeds marks a post that failed mid-pipeline with None; only the
    # posts that made it all the way through get marked seen.
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(app, "read_feed_urls", lambda: ["https://feed.example/rss"])
    monkeypatch.setattr(
        app,
        "get_post_entries_v2",
        lambda url: _entries("https://a.example/1", "https://a.example/2"),
    )
    monkeypatch.setattr(
        app,
        "filter_new_posts",
        lambda feed_url, urls: ["https://a.example/1", "https://a.example/2"],
    )
    monkeypatch.setattr(app, "get_feeds", lambda entries: entries)
    monkeypatch.setattr(app, "process_feeds", lambda feeds: [None, "dest.mp3"])
    marked = []
    monkeypatch.setattr(app, "mark_post_seen", lambda feed_url, url: marked.append(url))

    app.check_and_run()

    assert marked == ["https://a.example/2"]


def test_a_broken_feed_does_not_stop_the_next_feed(monkeypatch):
    monkeypatch.setattr(app, "init_db", lambda: None)
    monkeypatch.setattr(
        app, "read_feed_urls", lambda: ["https://a.example/rss", "https://b.example/rss"]
    )

    def get_post_entries_v2(url):
        if url == "https://a.example/rss":
            raise RuntimeError("feed unreachable")
        return _entries("https://b.example/1")

    monkeypatch.setattr(app, "get_post_entries_v2", get_post_entries_v2)
    monkeypatch.setattr(app, "filter_new_posts", lambda feed_url, urls: urls)
    monkeypatch.setattr(app, "get_feeds", lambda entries: entries)
    monkeypatch.setattr(app, "process_feeds", lambda feeds: ["dest.mp3"] * len(feeds))
    marked = []
    monkeypatch.setattr(app, "mark_post_seen", lambda feed_url, url: marked.append(url))

    app.check_and_run()

    assert marked == ["https://b.example/1"]
