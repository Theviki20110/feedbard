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
  ├─ feeds/reader.get_feeds()           fetch full post HTML + hero image
  ├─ pipeline/orchestrator.process_feeds()
  │    for each item:
  │      ├─ ingestion/html_parser.extract()          HTML -> typed blocks + visuals
  │      ├─ pipeline/visual_describer.describe_visuals()  LLM vision -> classify/describe images
  │      ├─ pipeline/translator.translate_blocks()        LLM -> translated blocks
  │      └─ pipeline/audio_renderer.generate_audio_from_blocks()  TTS -> audio file
  └─ feeds/store.mark_post_seen()       record processed URLs
```

`llm_client.py` and `asr_client.py` are the only places that talk to external
model APIs (Anthropic / Bedrock / Ollama for LLM, a Whisper-compatible
server for transcription-based QA in the audio renderer).

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
