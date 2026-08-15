import pytest

from substack_feed import paths
from substack_feed.pipeline import publisher

# A real MPEG-1 Layer III frame: mutagen refuses to tag a file it cannot
# parse, so a dummy byte string will not do. Header ff fb 90 c0 declares
# 128 kbps at 44.1 kHz, mono, which fixes the frame at
# 144 * 128000 / 44100 = 417 bytes. Getting that length wrong is what makes
# mutagen fail to sync, since it looks for the next header at exactly the
# offset the header itself implies.
SILENT_FRAME = bytes.fromhex("fffb90c0") + b"\x00" * 413


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Point every path helper at a scratch tree.

    The helpers close over module-level constants, so each one has to be
    repointed individually -- patching only LIBRARY_DIR would leave
    published_episode_path() writing to the real library.
    """
    monkeypatch.setattr(paths, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(paths, "EPISODES_DIR", tmp_path / "episodes")
    monkeypatch.setattr(paths, "COVERS_DIR", tmp_path / "covers")
    paths.EPISODES_DIR.mkdir(parents=True)
    paths.COVERS_DIR.mkdir(parents=True)
    return tmp_path


def _render(title: str) -> None:
    paths.episode_path(title).write_bytes(SILENT_FRAME * 40)


def test_publishes_into_author_title_tree(library):
    _render("Notes on Midtraining")
    dest = publisher.publish_item("Notes on Midtraining", "Sebastian Raschka")

    # This exact shape is what Audiobookshelf parses author and title from.
    assert dest.parent.name == "Notes on Midtraining"
    assert dest.parent.parent.name == "Sebastian Raschka"
    assert dest.name == "Notes on Midtraining.mp3"


def test_cover_lands_beside_the_audio(library):
    _render("Notes on Midtraining")
    paths.cover_path("Notes on Midtraining", ".png").write_bytes(b"\x89PNG\r\n\x1a\n")

    dest = publisher.publish_item("Notes on Midtraining", "Sebastian Raschka")
    assert (dest.parent / "cover.png").exists()


def test_missing_cover_still_publishes_audio(library):
    _render("Notes on Midtraining")
    dest = publisher.publish_item("Notes on Midtraining", "Sebastian Raschka")
    assert dest.exists()


def test_missing_episode_publishes_nothing(library):
    assert publisher.publish_item("Never Rendered", "Someone") is None


def test_existing_item_is_not_overwritten_by_default(library):
    _render("Notes on Midtraining")
    dest = publisher.publish_item("Notes on Midtraining", "Sebastian Raschka")
    dest.write_bytes(b"edited by the media server")

    publisher.publish_item("Notes on Midtraining", "Sebastian Raschka")
    assert dest.read_bytes() == b"edited by the media server"

    publisher.publish_item("Notes on Midtraining", "Sebastian Raschka", overwrite=True)
    assert dest.read_bytes() != b"edited by the media server"


def test_publish_leaves_the_source_in_place(library):
    # The library is a copy: the pipeline must keep its own state of record.
    _render("Notes on Midtraining")
    publisher.publish_item("Notes on Midtraining", "Sebastian Raschka")
    assert paths.episode_path("Notes on Midtraining").exists()


def test_tags_are_written(library):
    from mutagen.mp3 import MP3

    _render("Notes on Midtraining")
    dest = publisher.publish_item(
        "Notes on Midtraining", "Sebastian Raschka", "2026-08-15T10:00:00.000Z"
    )
    tags = MP3(dest).tags
    assert tags["TIT2"].text[0] == "Notes on Midtraining"
    assert tags["TPE1"].text[0] == "Sebastian Raschka"
    assert str(tags["TDRC"].text[0]) == "2026"


def test_library_names_keep_spaces_but_drop_separators():
    # Unlike slug(), these are display strings ABS parses back out.
    assert paths.library_name("KV Sharing: a/b") == "KV Sharing a b"
    assert paths.library_name("trailing dot.") == "trailing dot"
    assert paths.library_name("///") == "Unknown"


def test_publish_from_feed_item_without_title(library):
    assert publisher.publish({"author": "Someone"}) is None
