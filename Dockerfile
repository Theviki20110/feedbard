FROM python:3.12-slim

# ffmpeg: audio concat/encode step in the pipeline needs it at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Deps first so code-only changes don't invalidate this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

RUN mkdir -p /app/audio /app/images /app/data

ENV DB_PATH=/app/data/post_store.sqlite3 \
    AUDIO_DIR=/app/audio \
    IMAGES_DIR=/app/images \
    CRON_INTERVAL_SECONDS=10800

VOLUME ["/app/data", "/app/audio", "/app/images"]

ENTRYPOINT ["/app/entrypoint.sh"]
