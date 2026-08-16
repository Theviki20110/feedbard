import pytest

from feedbard.ingestion import fetcher
from feedbard.ingestion.fetcher import Article, fetch_article, is_substack


def entry(**overrides) -> dict:
    base = {
        "url": "https://blog.example.com/post",
        "image_url": "",
        "title": "A Post",
        "author": "",
        "published_at": "",
        "content_html": "",
        "generator": "",
        "feed_title": "The Blog",
    }
    return base | overrides


def body(paragraphs: int) -> str:
    return "<p>" + ("word " * 40 + "</p><p>") * paragraphs + "</p>"


def test_is_substack_matches_hosted_domain():
    assert is_substack(entry(url="https://someone.substack.com/p/x"))


def test_is_substack_matches_custom_domain_via_generator():
    # A publication on its own domain is only identifiable from the feed.
    assert is_substack(entry(url="https://newsletter.example.com/p/x", generator="Substack"))


def test_is_substack_false_for_plain_blog():
    assert not is_substack(entry(generator="Ghost 5.0"))


def test_feed_strategy_used_when_the_feed_carries_the_whole_post():
    article = fetch_article(entry(content_html=body(10), author="Jane Roe"))

    assert "word" in article.html
    assert article.title == "A Post"
    assert article.author == "Jane Roe"


def test_truncated_feed_falls_through_to_the_page(monkeypatch):
    page = "<html><head><meta property='og:image' content='https://img.example/c.jpg'>"
    page += f"</head><body><article>{body(10)}</article></body></html>"
    monkeypatch.setattr(fetcher, "_get", lambda url: type("R", (), {"text": page})())

    article = fetch_article(entry(content_html="<p>teaser</p>"))

    assert "word" in article.html
    assert article.cover_image == "https://img.example/c.jpg"


def test_author_falls_back_to_the_host_when_nobody_declares_one():
    article = fetch_article(entry(url="https://www.some-blog.org/p/1", content_html=body(10)))

    assert article.author == "Some Blog"


def test_profile_url_is_not_taken_as_an_author_name(monkeypatch):
    page = "<html><head><meta property='article:author' content='https://x.example/jane'>"
    page += f"</head><body><article>{body(10)}</article></body></html>"
    monkeypatch.setattr(fetcher, "_get", lambda url: type("R", (), {"text": page})())

    article = fetch_article(entry(url="https://blog.example.com/p/1"))

    assert article.author == "Blog"


def test_a_raising_strategy_is_skipped_not_fatal(monkeypatch):
    def boom(_entry):
        raise RuntimeError("publisher changed its markup")

    monkeypatch.setattr(fetcher, "STRATEGIES", (boom, fetcher._from_feed))

    article = fetch_article(entry(content_html=body(10)))

    assert isinstance(article, Article)


def test_exhausted_strategies_raise_rather_than_narrating_nothing(monkeypatch):
    monkeypatch.setattr(fetcher, "STRATEGIES", (lambda _entry: None,))

    with pytest.raises(RuntimeError):
        fetch_article(entry())
