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

RUN mkdir -p /app/data /app/library

# Two volumes: /app/data is the pipeline's private state, /app/library is the
# published tree the media server reads. Keeping them apart means the media
# server never sees scratch files, and its library can be mounted read-only
# elsewhere without exposing the rest.
#
# Set after the build steps on purpose: dependency bytecode still gets
# compiled into the image, while at runtime the container writes nothing
# outside the two volumes.
ENV DATA_DIR=/app/data \
    LIBRARY_DIR=/app/library \
    CRON_INTERVAL_SECONDS=10800 \
    PYTHONDONTWRITEBYTECODE=1

VOLUME ["/app/data", "/app/library"]

ENTRYPOINT ["/app/entrypoint.sh"]
