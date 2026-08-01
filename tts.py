import os

import requests

KOKORO_BASE_URL = "https://tts.home.vinzlab.com"
AUDIO_DIR = "audio"


def text_to_speech(text: str, voice: str = "if_sara") -> bytes:
    response = requests.post(
        f"{KOKORO_BASE_URL}/v1/audio/speech",
        json={
            "model": "kokoro",
            "voice": voice,
            "input": text,
            "response_format": "mp3",
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.content


def generate_audio_from_text(text: str, title: str, dest_dir: str = AUDIO_DIR) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    filename = "".join(c if c.isalnum() or c in "-_" else "_" for c in title) + ".mp3"
    dest_path = os.path.join(dest_dir, filename)

    audio_bytes = text_to_speech(text)
    with open(dest_path, "wb") as f:
        f.write(audio_bytes)
    return dest_path
