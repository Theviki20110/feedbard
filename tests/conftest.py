import os

# Set before any feedbard import: several modules read required env vars
# at import time (os.environ[...]), so tests need dummy values in place first.
os.environ.setdefault("FEEDS_LIST_PATH", "unused_feeds.txt")
os.environ.setdefault("DATA_DIR", "unused_data")
os.environ.setdefault("ASR_BASE_URL", "http://unused")
os.environ.setdefault("TTS_BASE_URL", "http://unused")
os.environ.setdefault("MODEL_ID", "unused-model")
os.environ.setdefault("ONLY_FIRST_FEED", "false")
