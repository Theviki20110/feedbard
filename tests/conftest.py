import os

# Set before any feedbard import: modules read their configuration at import
# time, so tests need values that point nowhere real in place first.
os.environ.setdefault("FEEDS_LIST_PATH", "unused_feeds.txt")
os.environ.setdefault("DATA_DIR", "unused_data")
os.environ.setdefault("ASR_BASE_URL", "http://unused")
os.environ.setdefault("TTS_BASE_URL", "http://unused")
os.environ.setdefault("MODEL_ID", "unused-model")
