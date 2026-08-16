from feedbard.feeds.reader import read_feed_urls


def test_read_feed_urls_skips_blank_lines_and_comments(tmp_path):
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text(
        "https://a.example/rss\n\n# https://commented-out.example/rss\nhttps://b.example/rss\n"
    )

    urls = read_feed_urls(str(feeds_file))

    assert urls == ["https://a.example/rss", "https://b.example/rss"]


def test_read_feed_urls_strips_whitespace(tmp_path):
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text("  https://a.example/rss  \n")

    urls = read_feed_urls(str(feeds_file))

    assert urls == ["https://a.example/rss"]
