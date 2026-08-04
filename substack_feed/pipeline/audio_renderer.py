import os
import re
from sqlite3 import IntegrityError

import requests
from dotenv import load_dotenv

from substack_feed.ingestion.html_parser import VIS_RE, Document

load_dotenv()

KOKORO_BASE_URL = os.environ["KOKORO_BASE_URL"]
AUDIO_DIR = os.environ["AUDIO_DIR"]

def check_integrity(doc: Document, text: str) -> None:
    """Run this AFTER every stage that touches the text. If a model
    rewrote or swallowed a placeholder, fail explicitly: audio with
    silent gaps is worse than a pipeline that stops."""
    expected = doc.visual_order()
    found = VIS_RE.findall(text)
    if found != expected:
        missing = [v for v in expected if v not in found]
        extra = [v for v in found if v not in expected]
        raise IntegrityError(
            f"expected {len(expected)} placeholders, found {len(found)}; "
            f"missing={missing} extra={extra} "
            f"reordered={not missing and not extra}")


def render_for_tts(doc: Document, text: str, *, frame: str = "Nella figura: {d}") -> str:
    """Final substitution. No model involved: at this point the merge
    is a str.replace, because position was never lost."""
    check_integrity(doc, text)

    def sub(m: re.Match) -> str:
        v = doc.visuals[m.group(1)]
        if v.klass == "decorativo" or not v.description:
            return ""
        return frame.format(d=v.description.rstrip(". ") + ".")

    return re.sub(r"\n{3,}", "\n\n", VIS_RE.sub(sub, text)).strip()


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


def generate_audio_from_blocks(text: str, title: str, dest_dir: str = AUDIO_DIR) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:150]
    dest_path = os.path.join(dest_dir, safe_title + ".mp3")

    audio_bytes = text_to_speech(text)
    with open(dest_path, "wb") as f:
        f.write(audio_bytes)
    return dest_path
