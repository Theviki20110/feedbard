# feedbard

Polls a list of RSS/Atom feeds, turns new posts into narrated audio
(translated, with visuals described for TTS), and keeps track of what's
already been processed.

## Intended use

A personal listening tool: it makes a private, spoken copy of articles *you*
are already entitled to read, for your own use.

Everything it produces is a derivative work of someone else's writing —
translated, narrated, re-hosted. So:

- Only add feeds whose terms allow it. The feed list (`FEEDS_LIST_PATH`,
  default `data/feeds_list.txt`) ships empty for that reason; nothing is
  processed until you put URLs in it.
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
  ├─ feeds/reader.read_feed_urls()      FEEDS_LIST_PATH -> feed URLs
  ├─ feeds/reader.get_post_entries_v2() feed -> entries (url, title, author,
  │                                     date, body if the feed carries one)
  ├─ feeds/store.filter_new_posts()     drop entries already seen (sqlite)
  ├─ (cap new entries to MAX_POSTS_PER_RUN)  keep one cron tick from turning
  │                                     a feed's backlog into an hours-long run
  ├─ feeds/reader.get_feeds()           ingestion/fetcher -> article HTML + cover
  ├─ pipeline/orchestrator.process_feeds()
  │    for each item:
  │      ├─ ingestion/html_parser.extract()          HTML -> typed blocks + visuals
  │      ├─ pipeline/triage.triage()                 LLM -> where the article ends,
  │      │                                           what carries nothing, what moves
  │      ├─ pipeline/visual_describer.describe_visuals()  LLM vision -> classify/describe images
  │      ├─ pipeline/table_describer.describe_tables()    LLM -> a table read as prose
  │      ├─ pipeline/code_describer.describe_code_blocks() LLM -> code/diagram read as prose
  │      ├─ pipeline/narration.attach_descriptions()      descriptions -> the block stream
  │      ├─ pipeline/translator.translate_blocks()        LLM -> translated blocks
  │      ├─ pipeline/sanitizer.sanitize_blocks()          allowlist + LLM -> speakable text
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

Requests to `huggingface.co` carry an `Authorization: Bearer` header when
`HF_TOKEN` is set — that host rate-limits unauthenticated requests hard
enough to trip the 429 retry loop otherwise.

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
    visual/     image class + description       │
    table/      spoken description, per table    │
    code/       spoken description, per code block│
    triage/     the editorial verdict, per article┘
```

Episodes, covers, and the text/speech/audio shards are all named from the
same `paths.slug()` of the article title, so an episode pairs with its cover
by name alone — no lookup table to keep in sync. Figures, visual shards and
table shards are keyed on the content hash of what they describe instead,
which is what lets a figure or a table reused across two articles be fetched
and described once.

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
line in the file `FEEDS_LIST_PATH` points at (default `data/feeds_list.txt`,
empty by default — see **Intended use**); a run with no feeds does nothing.

`.env` holds live credentials (model provider API keys, AWS keys). It is
gitignored and excluded from the image build context — keep it that way. Scope
the AWS key to the Bedrock and Polly actions the pipeline needs and nothing
else: on a container host, anything that can talk to the Docker daemon can
read the environment of a running container.

## Narration language

`TARGET_LANGUAGE` (default `Italian`) sets the language articles are
translated into and narrated in, and it is the only place the language is
written. It accepts whatever an operator would naturally type -- an English
language name (`Italian`), a bare code (`it`), or a full tag (`pt-BR`) --
and `feedbard/language.py` derives the rest:

| Derived | Used by | Example |
| --- | --- | --- |
| name | every LLM prompt | `Italian` |
| short code | HTTP TTS server, ASR QA round-trip | `it` |
| BCP-47 tag | Polly | `it-IT` |

- `TTS_VOICE_ID` / `TTS_LANGUAGE_CODE` -- voice and short code for
  `TTS_PROVIDER=http` (also used by the ASR round-trip check).
- `POLLY_VOICE_ID` / `POLLY_LANGUAGE_CODE` -- voice and BCP-47 code for
  `TTS_PROVIDER=polly`.
- `VOXCPM_MODEL` / `VOXCPM_REF_AUDIO` -- model name and reference clip for
  `TTS_PROVIDER=voxcpm` (VoxCPM2's OpenAI-compatible `/v1/audio/speech`).
  The voice comes from `ref_audio`, a short wav used for zero-shot cloning,
  not from a `voice` name: that field exists in the request schema but is
  ignored unless it names a preset already registered on the server, so
  without `VOXCPM_REF_AUDIO` every call invents a different speaker.

`TTS_LANGUAGE_CODE` and `POLLY_LANGUAGE_CODE` remain as overrides for what
derivation cannot know -- a regional variant the tag does not carry, or a
provider that spells a code its own way -- but they no longer have to be kept
in sync by hand, which is what used to let a half-edited `.env` narrate one
language with another's codes.

**Voices are not derived.** Nothing maps "which language" to "which voice", so
`TTS_VOICE_ID` / `POLLY_VOICE_ID` stay explicit. Their defaults are Italian; if
`TARGET_LANGUAGE` names another language and the voice is still the default,
the run logs a warning rather than reading the right words in the wrong accent.

**There is nothing else to configure.** No word list, no symbol table, no
per-language file. Adding a language means setting `TARGET_LANGUAGE` and
picking a voice. The prompts stay in English because they are instructions to
a model, not text anyone hears.

That is a deliberate constraint, and it is what the rest of this section is
about: every decision that depends on a language is either derived from the
Unicode database or made by the model, never written down in the code.

### What is speakable, and how it is decided

`pipeline/speakable.py` answers this by exclusion. A passage made only of
letters, combining marks, digits, spaces and punctuation is already speech and
is sent to the voice untouched. Anything else -- a symbol, a URL, a fragment
of code, an emoji, a rule drawn out of box characters -- is notation, and only
the model can say what it should sound like *here*.

Nothing in that rule names a language. Letters, marks and digits come from
Unicode general categories. The punctuation set is derived rather than listed:
a character counts as punctuation if the Unicode database says its name is one
of thirteen concepts (`FULL STOP`, `COMMA`, `QUESTION MARK`, `PARENTHESIS`,
`DANDA`, ...), so `,` `、` `،` all arrive as "comma" and every script's
version comes along without anyone adding it. Listing the characters instead
would have quietly meant "Latin".

Three things pass the character test and still are not prose, so they are
checked separately:

- **A lone letter from another script.** `σ` in Italian prose is a variable
  and has to be spelled out; the same `σ` in Greek prose is a word. What
  matters is that it stands alone *and* is foreign to this passage, never that
  it is Greek — a rule based on script alone would have flagged every Japanese
  article, since Japanese mixes three scripts in a sentence.
- **A run of two or more dashes.** `--model` is a flag, `—` is punctuation.
- **A name followed by a dotted version number.** `DeepSeek 4.5` must be read
  as a version; read as a decimal it is a different number, and where the dot
  separates thousands, a very different one.

On a corpus of real translated blocks this sends about one block in five to
the model, so the common case costs nothing.

### What happens to the rest

`pipeline/sanitizer.py` checks, asks, asks again, then deletes:

1. Already speakable -> returned as-is, no call.
2. One call, told what tripped the check. If the answer passes, done.
3. One more call, told what the previous answer left behind. If it passes, done.
4. `strip_unspeakable` drops what is left -- whole tokens, not characters, so
   a URL leaves a gap rather than a run of letters. Reached only after the
   model has failed twice, and logged as the failure it is.

Tables, code blocks and figures never take this path: they are described as
prose by their own stages before they ever reach it. Those descriptions still
go through the sanitizer, since a described formula arrives full of symbols.

### Declining

Every prompt in `assets/` tells the model to answer with **nothing** rather
than explain why it will not do the task. A sentence about the task is
indistinguishable from article prose downstream: it gets saved to a shard and
read aloud as if the author had written it. An empty answer is a value the
pipeline can act on -- the table falls back to reading its rows, the code block
is dropped, the paragraph is asked for once more and then reported as missing.

### Editorial judgements

`pipeline/triage.py` makes one call per article, cached in `shards/triage/`,
and decides three things the code used to decide with English regexes: where
the closing material starts (bibliography, author bio, further reading), which
blocks carry nothing a listener needs (subscription prompts, shop links,
source-only captions), and which paragraph refers back to the figure above it
and should be narrated before it.

It returns indices, never text, so it cannot rewrite the article. Nothing is
removed from the block list -- a dropped block is emptied in place, because
every per-block shard on disk is keyed by index. And it fails open: any error,
bad JSON, or out-of-range index leaves the article exactly as parsed, because
narrating a bibliography is a much smaller problem than losing half a post.

## What a listener hears that they cannot see

Four kinds of content in an article are not prose, and each would be a silent
gap in the episode if it were simply skipped:

- **Figures.** The vision pass (`assets/visual_prompt.txt`) classifies each
  image as `decorative`, `illustrative`, or `essential` and, for the last
  two, writes a short paragraph saying what it shows -- the takeaway of a
  chart, a formula read out in words. `decorative` covers banners, logos, and
  header photos: those stay silent, because narrating them interrupts the
  prose without adding anything. SVG images (shields.io badges, arXiv/build
  status) are classified `decorativo` without a vision call: they are vector,
  not raster, so there is nothing for a vision model to look at.
- **Tables.** A grid read cell by cell is unlistenable and a grid dropped
  takes its numbers with it, so an LLM restates it
  (`assets/table_prompt.txt`): small tables are read out in full, larger ones
  become the comparison they make plus the values that matter. If that call
  fails, the rows are read out flatly rather than lost.
- **Code blocks.** Read character by character, code is unlistenable and
  carries no information a listener can use, so an LLM describes what it does
  or shows instead (`assets/code_prompt.txt`) -- this also catches a `<pre>`
  block that is actually ASCII art (boxes, arrows) rather than runnable code.
  If that call fails, the block is dropped from narration rather than read
  raw.
- **Footnotes.** The `[3]` marker is stripped mid-sentence -- a number there
  wrecks the prosody -- and the note's body is appended to the paragraph that
  cites it, so it is heard where the author put it.

Descriptions are written directly in `TARGET_LANGUAGE`, so nothing translates
them afterwards; the sanitizer still runs over them, since a described formula
arrives full of symbols. `pipeline/narration.py` is what writes each
description back into the block the parser emitted for it, which is what fixes
its position in the episode. Both kinds of description are cached in
`shards/` by content hash, so an interrupted run never pays for the same
figure or table twice.

## Links

A read-aloud URL is a minute of spelled-out path segments a listener cannot
use, so no link reaches the speech engine.

Nothing special is needed to catch one. A URL contains characters that are not
letters, digits, spaces or punctuation, so it fails the allowlist like any
other notation, and the block goes to the model with the rest of its context.
The model rewrites the sentence around the address -- "the code is available
on GitHub" rather than a sentence that trails off -- because the domain is
still there for it to read.

If the model hands back an address anyway, the answer fails the same check
that sent it, and it is asked once more. If that fails too, the last-resort
scrub drops the whole token rather than the offending character: deleting only
the slashes out of `https://github.com/rasbt/evals` would leave
`httpsgithubcomrasbtevals` for the voice to attempt.

## Running

```
uv run python cron_job.py     # single run (reads FEEDS_LIST_PATH)
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
