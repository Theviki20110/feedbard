from bs4 import BeautifulSoup
from substack_api import Post


def get_text_from_html(html: str) -> str:
    # Initialize a post by its URL
    post = Post(html)

    # Get post metadata
    metadata = post.get_metadata()

    # Get the post's HTML content
    content = post.get_content()

    return content, metadata

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "noscript", "svg", "template"]):
        element.decompose()

    return soup.get_text(" ", strip=True)