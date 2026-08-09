import os

import requests
from dotenv import load_dotenv

load_dotenv()

ASR_BASE_URL = os.environ["ASR_BASE_URL"]


def generate_transcription(audio_bytes: bytes, lang: str = "it") -> str:
    resp = requests.post(
        f"{ASR_BASE_URL}/asr",
        params={"task": "transcribe", "language": lang, "output": "json", "encode": "true"},
        files={"audio_file": ("block.mp3", audio_bytes)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]
