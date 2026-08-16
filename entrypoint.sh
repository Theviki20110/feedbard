#!/bin/sh
# Container loop: seed configuration onto the volumes, drop to the unprivileged
# user Unraid expects, then run the pipeline on an interval.
set -eu

PUID="${PUID:-99}"
PGID="${PGID:-100}"
umask "${UMASK:-022}"

mkdir -p "$DATA_DIR" "$LIBRARY_DIR"

# The feed list is configuration on the data volume; the copy in the image is
# only a starting point, so it is never allowed to overwrite an edited file.
if [ ! -f "$FEEDS_LIST_PATH" ]; then
    cp /app/assets/feeds_list.txt "$FEEDS_LIST_PATH"
    echo "seeded feed list at $FEEDS_LIST_PATH"
fi

# Only the volume roots and the files the container owns: a recursive chown of
# a media library is minutes of work on every start, and Audiobookshelf only
# needs to read what was written under the umask above.
if [ "$(id -u)" = "0" ]; then
    chown "$PUID:$PGID" "$DATA_DIR" "$LIBRARY_DIR" "$FEEDS_LIST_PATH" 2>/dev/null || true
    RUN="gosu $PUID:$PGID"
else
    RUN=""
fi

echo "feedbard container start, uid=$PUID gid=$PGID interval=${CRON_INTERVAL_SECONDS}s"

while true; do
    echo "[$(date -Iseconds)] run start"
    # A failed run must not kill the container: the next tick retries, and a
    # transient feed or model error is the common case.
    if $RUN python /app/cron_job.py; then
        echo "[$(date -Iseconds)] run ok"
    else
        echo "[$(date -Iseconds)] run failed, retrying next tick"
    fi
    echo "[$(date -Iseconds)] sleeping ${CRON_INTERVAL_SECONDS}s"
    sleep "${CRON_INTERVAL_SECONDS}"
done
