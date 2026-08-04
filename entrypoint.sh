#!/bin/sh
set -eu

echo "substack-feed container start, interval=${CRON_INTERVAL_SECONDS}s"

while true; do
    echo "[$(date -Iseconds)] run start"
    uv run --frozen python cron_job.py || echo "[$(date -Iseconds)] run failed, continuing"
    echo "[$(date -Iseconds)] run end, sleeping ${CRON_INTERVAL_SECONDS}s"
    sleep "${CRON_INTERVAL_SECONDS}"
done
