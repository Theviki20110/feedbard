from substack_api import Post


def get_text_from_html(html: str) -> str:
    post = Post(html)
    metadata = post.get_metadata()
    content = post.get_content()
    return content, metadata
