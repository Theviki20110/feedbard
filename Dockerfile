FROM python:3.12-slim

# ffmpeg: audio concat/encode step in the pipeline needs it at runtime.
# gosu: the entrypoint starts as root to fix volume ownership, then drops to
# PUID/PGID -- the Unraid convention for keeping the media library readable.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gosu \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Deps first so code-only changes don't invalidate this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen && chmod +x /app/entrypoint.sh

RUN mkdir -p /app/data /app/library

# Two volumes: /app/data is the pipeline's private state, /app/library is the
# published tree the media server reads. Keeping them apart means the media
# server never sees scratch files, and its library can be mounted read-only
# elsewhere without exposing the rest.
#
# Set after the build steps on purpose: dependency bytecode still gets
# compiled into the image, while at runtime the container writes nothing
# outside the two volumes.
#
# FEEDS_LIST_PATH points into the data volume, not at the copy baked into the
# image: the list is configuration, so it has to survive an image update and
# be editable from the host. The entrypoint seeds it on first start.
ENV DATA_DIR=/app/data \
    LIBRARY_DIR=/app/library \
    FEEDS_LIST_PATH=/app/data/feeds_list.txt \
    CRON_INTERVAL_SECONDS=10800 \
    LOG_LEVEL=INFO \
    PUID=99 \
    PGID=100 \
    UMASK=022 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

VOLUME ["/app/data", "/app/library"]

# tini reaps the ffmpeg children the renderer spawns and forwards the stop
# signal, so `docker stop` ends a run instead of waiting out the timeout.
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]

LABEL org.opencontainers.image.title="feedbard" \
      org.opencontainers.image.description="Turns RSS/Atom posts into narrated audiobooks for Audiobookshelf" \
      org.opencontainers.image.source="https://github.com/Theviki20110/feedbard" \
      org.opencontainers.image.licenses="MIT"
