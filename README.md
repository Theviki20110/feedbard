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
  │      ├─ pipeline/audio_renderer.generate_audio_from_blocks()  TTS -> episode MP3
  │      └─ pipeline/publisher.publish()                  -> Audiobookshelf library
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

## Publishing to Audiobookshelf

Publishing copies each finished episode into `LIBRARY_DIR` (default
`library/`), laid out the way an Audiobookshelf **book** library expects:

```
library/
  Sebastian Raschka/
    Understanding Reasoning LLMs/
      Understanding Reasoning LLMs.mp3
      cover.jpg
```

Articles are published as books rather than as podcast episodes because a
podcast library allows exactly one cover for the entire feed, while a book
library gives every item its own — which is the point of keeping per-article
artwork. ABS parses the author and title straight out of the folder names, so
those keep their spaces and capitals; only characters that would break a path
are stripped. The MP3 is also ID3-tagged (title, artist, album, year, embedded
cover), since the concatenated TTS output carries no tags at all.

The library is a copy: `DATA_DIR` stays the pipeline's state of record, so a
republish never depends on what the media server did to its own files.

In Audiobookshelf: **Settings → Libraries → Add Library**, type **Book**,
folder = your `LIBRARY_DIR`, then **Scan**.

```
uv run python scripts/publish.py --list          # what is rendered, and does it have a cover
uv run python scripts/publish.py --all           # backfill everything
uv run python scripts/refetch_covers.py          # re-download missing cover art
```

`publish.py` looks the author and date up from the feeds, since the rendered
files carry neither; `--author` skips that lookup for posts that have fallen
off the feed.

## Setup

```
uv sync
cp .env.example .env   # fill in the values
```

Required env vars are documented in `.env.example`.

## Running

```
uv run python cron_job.py     # single run (reads assets/feeds_list.txt)
```

`entrypoint.sh` runs the same thing in a loop, on `CRON_INTERVAL_SECONDS`
(used by the Docker image, see `Dockerfile` / `docker-compose.yml`).

## Development

```
uv sync --group dev
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # tests
```
