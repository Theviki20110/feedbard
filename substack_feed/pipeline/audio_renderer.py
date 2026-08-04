import os
import jiwer
import requests
from dotenv import load_dotenv
from substack_feed.asr_client import generate_transcription

load_dotenv()

TTS_BASE_URL = os.environ["TTS_BASE_URL"]
AUDIO_DIR = os.environ["AUDIO_DIR"]
WER_THRESHOLD = 0.2  # 20% WER threshold for logging warnings

def call_tts(text: str, language: str = "it", voice_id: str = "Leonardo.wav") -> bytes:
    body = {
            "voice_mode": "predefined",
            "predefined_voice_id": voice_id,
            "output_format": "wav",
            "split_text": True,
            "text": text,
            "language": language,
        }
    response = requests.post(
        f"{TTS_BASE_URL}/tts",
        json=body,
        timeout=180,
    )
    response.raise_for_status()
    return response.content

def generate_speech(text: str, language: str = "it", voice_id: str = "Leonardo.wav") -> bytes:

    while True:
        audio_bytes = call_tts(text, language, voice_id)
        transcription = generate_transcription(audio_bytes, lang=language)

        wer = jiwer.wer(text, transcription)

        if wer > WER_THRESHOLD:  # If WER is greater than 20%, log a warning
            print(f"Warning: High WER ({wer:.2%}) for text: {text[:50]}...")
        else:
            break

    return audio_bytes

def generate_audio_from_blocks(documents, title: str, dest_dir: str = AUDIO_DIR) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:150]
    dest_path = os.path.join(dest_dir, safe_title + ".mp3")
    audio_bytes = bytearray()

    for index, document in enumerate(documents.blocks):
        if document.translated_text is None:
            continue
        audio_bytes.extend(generate_speech(document.translated_text))
        print(f"[{title}] render_for_tts: block {index}")

    with open(dest_path, "wb") as f:
        f.write(audio_bytes)
    return dest_path
