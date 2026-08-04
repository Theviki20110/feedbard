import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

WHISPER_BASE_URL = os.environ["WHISPER_BASE_URL"]

def generate_transcription(audio_bytes: bytes, lang: str = "it") -> str:
    resp = requests.post(
        f"{WHISPER_BASE_URL}/asr",
        params={"task": "transcribe", "language": lang, "output": "json", "encode": "true"},
        files={"audio_file": ("block.mp3", audio_bytes)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]