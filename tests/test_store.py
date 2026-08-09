from substack_feed.feeds.store import filter_new_posts, init_db, mark_post_seen


def test_filter_new_posts_returns_all_when_none_seen(tmp_path):
    db_path = str(tmp_path / "store.sqlite3")
    init_db(db_path)

    new = filter_new_posts("https://feed.example/rss", ["a", "b"], db_path)

    assert new == ["a", "b"]


def test_mark_post_seen_excludes_it_from_future_filtering(tmp_path):
    db_path = str(tmp_path / "store.sqlite3")
    init_db(db_path)
    mark_post_seen("https://feed.example/rss", "a", db_path)

    new = filter_new_posts("https://feed.example/rss", ["a", "b"], db_path)

    assert new == ["b"]


def test_seen_posts_are_scoped_per_feed(tmp_path):
    db_path = str(tmp_path / "store.sqlite3")
    init_db(db_path)
    mark_post_seen("https://feed-a.example/rss", "a", db_path)

    new = filter_new_posts("https://feed-b.example/rss", ["a"], db_path)

    assert new == ["a"]


def test_mark_post_seen_is_idempotent(tmp_path):
    db_path = str(tmp_path / "store.sqlite3")
    init_db(db_path)
    mark_post_seen("https://feed.example/rss", "a", db_path)
    mark_post_seen("https://feed.example/rss", "a", db_path)

    new = filter_new_posts("https://feed.example/rss", ["a"], db_path)

    assert new == []
