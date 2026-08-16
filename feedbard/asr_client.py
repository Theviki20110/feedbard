import os

import requests
from dotenv import load_dotenv

load_dotenv()

# Read, not required: transcription is a QA step some deployments never
# reach, and an import-time KeyError would take the whole run down with it.
ASR_BASE_URL = os.getenv("ASR_BASE_URL", "")


def generate_transcription(audio_bytes: bytes, lang: str = "it") -> str:
    if not ASR_BASE_URL:
        raise RuntimeError("ASR_BASE_URL is not set; point it at a Whisper-compatible server")
    resp = requests.post(
        f"{ASR_BASE_URL}/asr",
        params={"task": "transcribe", "language": lang, "output": "json", "encode": "true"},
        files={"audio_file": ("block.mp3", audio_bytes)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]
