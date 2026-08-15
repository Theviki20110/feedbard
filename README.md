# substack-feed

Polls a list of Substack RSS feeds, turns new posts into narrated audio
(translated, with visuals described for TTS), and keeps track of what's
already been processed.

## Flow

```
app.check_and_run()
  ├─ feeds/reader.read_feed_urls()      assets/feeds_list.txt -> feed URLs
  ├─ feeds/reader.get_post_entries_v2() RSS -> entries (url, image_url)
  ├─ feeds/store.filter_new_posts()     drop entries already seen (sqlite)
  ├─ feeds/reader.get_feeds()           fetch full post HTML + cover art
  ├─ pipeline/orchestrator.process_feeds()
  │    for each item:
  │      ├─ ingestion/html_parser.extract()          HTML -> typed blocks + visuals
  │      ├─ pipeline/visual_describer.describe_visuals()  LLM vision -> classify/describe images
  │      ├─ pipeline/translator.translate_blocks()        LLM -> translated blocks
  │      ├─ pipeline/sanitizer.sanitize_blocks()          LLM + scrub -> speakable text
  │      └─ pipeline/audio_renderer.generate_audio_from_blocks()  TTS -> episode MP3
  └─ feeds/store.mark_post_seen()       record processed URLs
```

`llm_client.py` and `asr_client.py` are the only places that talk to external
model APIs (Anthropic / Bedrock / Ollama for LLM, a Whisper-compatible
server for transcription-based QA in the audio renderer).

## On-disk layout

Everything lives under `DATA_DIR` (default `data/`), split by how long the
artefacts are worth keeping:

```
data/
  episodes/     finished MP3, one per article    ─┐ the deliverable:
  covers/       episode artwork, one per article ─┘ point Audiobookshelf here
  figures/      article images, fetched for description (input cache)
  shards/
    text/       translated text, per block     ─┐ resumable scratch: deleting
    speech/     speech-ready text, per block    │ any of it costs money to
    audio/      synthesized audio, per block    │ rebuild but loses nothing
    visual/     image class + description       ─┘
```

Episodes, covers, and the text/speech/audio shards are all named from the
same `paths.slug()` of the article title, so an episode pairs with its cover
by name alone — no lookup table to keep in sync. Figures and visual shards
are keyed on the image content hash instead, which is what lets a figure
reused across two articles be fetched and described once.

`paths.py` is the single source of truth for these locations; no other module
builds a path by hand. `scripts/migrate_layout.py` moves a pre-refactor
`audio/` + `images/` tree onto this one.

