from feedbard import paths


def test_episode_and_cover_share_a_stem():
    # The whole point of slug-naming: the podcast container pairs an episode
    # with its artwork by filename, with no lookup table to keep in sync.
    title = "Controlling Reasoning Effort in LLMs"
    assert paths.episode_path(title).stem == paths.cover_path(title).stem


def test_shards_of_one_article_share_a_prefix():
    title = "Notes on Midtraining"
    stem = paths.slug(title)
    assert paths.text_shard_path(title, 7).name.startswith(stem)
    assert paths.speech_shard_path(title, 7).name.startswith(stem)
    assert paths.audio_shard_path(title, 7).name.startswith(stem)


def test_slug_strips_filesystem_hostile_characters():
    assert paths.slug("KV Sharing, mHC & Attention: a/b") == "KV_Sharing__mHC___Attention__a_b"


def test_slug_is_bounded():
    assert len(paths.slug("x" * 500)) == 150


def test_db_lives_under_the_data_root():
    # The dedup DB is the one unrebuildable piece of state, so it has to sit
    # inside the tree that gets mounted as a volume -- not next to the code.
    assert paths.DB_PATH.parent == paths.DATA_DIR


def test_deliverables_are_not_mixed_with_scratch():
    # An Audiobookshelf library points at episodes/ and covers/; a shard
    # landing in either would show up as a broken episode.
    assert paths.EPISODES_DIR.parent == paths.DATA_DIR
    assert paths.COVERS_DIR.parent == paths.DATA_DIR
    for shard_dir in (
        paths.TEXT_SHARDS_DIR,
        paths.SPEECH_SHARDS_DIR,
        paths.AUDIO_SHARDS_DIR,
        paths.VISUAL_SHARDS_DIR,
    ):
        assert shard_dir.parent == paths.SHARDS_DIR


def test_figures_and_visual_shards_are_keyed_on_content_hash():
    # Not on the title: that is what lets a figure reused across two articles
    # be fetched and described once.
    vid = "0123456789"
    assert paths.figure_path(vid, ".png").stem == vid
    assert paths.visual_shard_path(vid).stem == vid
