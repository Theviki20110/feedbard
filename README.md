# feedbard

Polls a list of RSS/Atom feeds, turns new posts into narrated audio
(translated, with visuals described for TTS), and keeps track of what's
already been processed.

## Intended use

A personal listening tool: it makes a private, spoken copy of articles *you*
are already entitled to read, for your own use.

Everything it produces is a derivative work of someone else's writing —
translated, narrated, re-hosted. So:

- Only add feeds whose terms allow it. `assets/feeds_list.txt` ships empty for
  that reason; nothing is processed until you put URLs in it.
- Paid or subscriber-only posts stay off the list unless the publisher's terms
  say otherwise.
- Don't redistribute the output. That includes exposing the Audiobookshelf
  library holding it to anyone but yourself.
- The fetcher's first strategy calls Substack's undocumented internal API. It
  is convenient, not sanctioned; using it may breach Substack's terms.

Respecting all of this is the operator's responsibility, not the tool's.

## Flow

```
app.check_and_run()
  ├─ feeds/reader.read_feed_urls()      assets/feeds_list.txt -> feed URLs
  ├─ feeds/reader.get_post_entries_v2() feed -> entries (url, title, author,
  │                                     date, body if the feed carries one)
  ├─ feeds/store.filter_new_posts()     drop entries already seen (sqlite)
  ├─ feeds/reader.get_feeds()           ingestion/fetcher -> article HTML + cover
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

## Getting the article

Any feed works, because how a post's body is obtained is the only thing that
differs between publishers, and that difference is confined to
`ingestion/fetcher.py`. It tries three strategies in order and takes the first
that returns a usable body:

1. **Substack API** -- for `*.substack.com` hosts and for custom domains whose
   feed says `<generator>Substack</generator>`. It is the only source that
   returns the subtitle, the post's own cover image, and a byline as a name.
2. **The feed itself** -- `content:encoded`, when the publisher puts the whole
   post in the feed (Ghost, WordPress, most static-site generators). Costs no
   request beyond the poll. A body under ~1200 characters of text is read as a
   teaser, not an article, and falls through.
3. **Readability over the page** -- the universal fallback for a truncated
   feed. Metadata comes from the entry first, then the page's Open Graph tags.

A strategy that raises is logged and skipped, so a publisher changing its
markup degrades that feed to the next strategy instead of failing the run.
Metadata is normalised on the way out (dates to ISO-8601, an absent byline to
the publication host), so nothing downstream knows where an article came from.

Adding a publisher-specific strategy means writing one function returning
`Article | None` and putting it in `STRATEGIES` before the generic ones.

## On-disk layout

Everything lives under `DATA_DIR` (default `data/`), split by how long the
artefacts are worth keeping:

```
data/
  post_store.sqlite3  posts already processed, per feed (the only
                      state here that cannot be rebuilt)
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
  Author Name/
    Article Title/
      Article Title.mp3
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

Required env vars are documented in `.env.example`. Then put one feed URL per
line in `assets/feeds_list.txt` (empty by default — see **Intended use**); a
run with no feeds does nothing.

`.env` holds live credentials (model provider API keys, AWS keys). It is
gitignored and excluded from the image build context — keep it that way. Scope
the AWS key to the Bedrock and Polly actions the pipeline needs and nothing
else: on a container host, anything that can talk to the Docker daemon can
read the environment of a running container.

## Narration language

`TARGET_LANGUAGE` (default `Italian`) sets the language articles are
translated into and narrated in. It reaches the LLM prompts directly, and the
deterministic scrub in `pipeline/sanitizer.py` reads its spoken forms for
symbols (`σ` -> "sigma", `<=` -> "minore o uguale a") from
`assets/lexicons/<language>.json`.

Supporting a new language means adding one JSON file there -- `symbols` for
single glyphs, `sequences` for ordered multi-character replacements, longest
first -- and pointing `TARGET_LANGUAGE` and the TTS voice at it. A language
with no lexicon file still runs: the scrub logs a warning and falls back to
the English spoken forms, so a symbol is narrated in the wrong language rather
than silently dropped from a claim.

## Links

A read-aloud URL is a minute of spelled-out path segments a listener cannot
use, so no link reaches the speech engine. Removal happens in three steps,
each covering the one before it:

1. **Before the LLM pass**, every URL is replaced by a `⟪link: host⟫` marker.
   The model never sees an address, so it cannot leave one in, and the host
   tells it what was being pointed at -- "available on GitHub" instead of a
   sentence that trails off.
2. **The LLM pass** consumes the marker as part of its normal rewrite,
   phrasing the sentence around the named source. No extra call, no extra cost.
3. **The scrub** deletes any marker or raw URL that survived and turns the
   connector it stranded into a full stop.

Step 3 is deterministic, so it cannot mend prose: `"cloned the scripts from,
we can run"` is grammatical nonsense in any language a regex could patch. When
a block reaches it having skipped or failed the LLM pass *and* it contained a
link, one small repair call fixes the connective tissue
(`assets/repair_prompt.txt`). That call is accepted only if the result keeps
every `⟦VIS:⟧` placeholder, introduces no link, is not a refusal, and stays
within 15% of the original length -- otherwise the unrepaired text is kept.
The repair pass can improve a block, never replace it.

## Running

```
uv run python cron_job.py     # single run (reads assets/feeds_list.txt)
                              # empty feed list -> nothing to do
```

`entrypoint.sh` runs the same thing in a loop, on `CRON_INTERVAL_SECONDS`
(used by the Docker image, see `Dockerfile` / `docker-compose.yml`).

## Container

`ghcr.io/theviki20110/feedbard:latest`, built for `linux/amd64` and
`linux/arm64` by `.github/workflows/docker-publish.yml` on every push to
`main`. Two volumes, matching the split above: `/app/data` for the pipeline's
state, `/app/library` for the Audiobookshelf book library.

```
docker run -d --name feedbard \
  -e LLM_PROVIDER=bedrock -e MODEL_ID=... \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_REGION=us-east-1 \
  -e TTS_PROVIDER=polly -e TARGET_LANGUAGE=Italian \
  -v /mnt/user/appdata/feedbard:/app/data \
  -v /mnt/user/media/audiobooks/feedbard:/app/library \
  ghcr.io/theviki20110/feedbard:latest
```

The feed list is configuration, so it lives on the data volume rather than in
the image: on first start the entrypoint copies `assets/feeds_list.txt` to
`$DATA_DIR/feeds_list.txt` and never touches it again, so editing it from the
host survives an image update.

Credentials come from env vars only — `AWS_PROFILE` names a profile in a
`~/.aws` the container does not have, so Bedrock and Polly need
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` instead.

Files are written as `PUID:PGID` (default `99:100`, the Unraid convention)
under `UMASK`, so the media server can read the library without running as
root. A failing run logs and is retried on the next tick rather than killing
the container: a feed or model error is usually transient.

### Unraid

The Community Applications template lives in its own templates repository, not
here. What it has to set: the two paths above (Library pointing at the same
share added in Audiobookshelf as a **Book** library), `LLM_PROVIDER` +
`MODEL_ID`, the provider credentials, `TTS_PROVIDER`, `TARGET_LANGUAGE`, and
optionally `CRON_INTERVAL_SECONDS`, `PUID`, `PGID`, `UMASK`. Everything else
has a working default in the image.

## Development

```
uv sync --group dev
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # tests
```

## License

MIT — see `LICENSE`. It covers this code only, not anything the pipeline
fetches or produces from third-party content.
