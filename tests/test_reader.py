from feedbard.feeds.reader import _entry_author, _entry_content, _entry_image, read_feed_urls


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


def test_entry_content_prefers_the_full_body_over_the_teaser():
    item = {"content": [{"value": "<p>full</p>"}], "summary": "<p>teaser</p>"}

    assert _entry_content(item) == "<p>full</p>"


def test_entry_content_falls_back_to_the_summary():
    assert _entry_content({"summary": "<p>teaser</p>"}) == "<p>teaser</p>"


def test_entry_author_prefers_the_post_byline_over_the_feed():
    item = {"author_detail": {"name": "Jane Roe"}}
    feed = {"author_detail": {"name": "The Blog"}}

    assert _entry_author(item, feed) == "Jane Roe"


def test_entry_author_ignores_bare_email_addresses():
    # feedparser puts the address in `name` when the field carries no name,
    # and an email read aloud is worse than no byline at all.
    item = {"author_detail": {"name": "noreply@example.com"}, "author": "noreply@example.com"}

    assert _entry_author(item, {}) == ""


def test_entry_image_prefers_media_content_then_enclosure():
    item = {
        "media_content": [{"url": "https://img.example/a.jpg"}],
        "links": [{"rel": "enclosure", "href": "https://img.example/b.jpg"}],
    }

    assert _entry_image(item) == "https://img.example/a.jpg"
    assert _entry_image({"links": item["links"]}) == "https://img.example/b.jpg"
