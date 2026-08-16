import difflib
import os
import re

import boto3
import jiwer
import requests
from dotenv import load_dotenv

from feedbard.asr_client import generate_transcription
from feedbard.logger import logger
from feedbard.paths import (
    AUDIO_SHARDS_DIR,
    EPISODES_DIR,
    audio_shard_path,
    episode_path,
)

load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "http").lower()
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "")
WER_THRESHOLD = 0.2  # 20% WER threshold for logging warnings
MAX_TTS_ATTEMPTS = 3

WER_NORMALIZE = jiwer.Compose(
    [
        jiwer.SubstituteRegexes({r"[_=<>|`]": " ", r"\s+": " "}),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

# Shell commands, URLs, code fences, and config blocks are structurally
# unspeakable by TTS/ASR round-trip comparison - retrying never helps since
# the mismatch isn't random. Skip the WER gate for these instead of burning
# 3x TTS+ASR calls per block on a check that can never pass.
CODE_LIKE_RE = re.compile(
    r"(https?://|^\s*[$#>]|```|-{1,2}\w[\w-]*=|\b\w+@\w+|::|/[\w./-]+/|\.(py|json|toml|sh|js)\b)",
    re.MULTILINE,
)


def _looks_like_code(text: str) -> bool:
    return bool(CODE_LIKE_RE.search(text))


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
POLLY_VOICE_ID = os.getenv("POLLY_VOICE_ID", "Bianca")
POLLY_ENGINE = os.getenv("POLLY_ENGINE", "generative")

_polly_client = None


def _get_polly_client():
    global _polly_client
    if _polly_client is None:
        _polly_client = boto3.client("polly", region_name=AWS_REGION)
    return _polly_client


def call_tts_http(text: str, language: str = "it", voice_id: str = "Leonardo.wav") -> bytes:
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


def call_tts_polly(text: str, language: str = "it") -> bytes:
    response = _get_polly_client().synthesize_speech(
        Text=text,
        VoiceId=POLLY_VOICE_ID,
        OutputFormat="mp3",
        Engine=POLLY_ENGINE,
        LanguageCode="it-IT" if language == "it" else language,
    )
    return response["AudioStream"].read()


def call_tts(text: str, language: str = "it", voice_id: str = "Leonardo.wav") -> bytes:
    if TTS_PROVIDER == "polly":
        return call_tts_polly(text, language)
    if TTS_PROVIDER == "http":
        return call_tts_http(text, language, voice_id)
    raise ValueError(f"Unsupported TTS_PROVIDER={TTS_PROVIDER!r}; use 'http' or 'polly'")


def load_final_audio_shard(title: str, block_index: int) -> bytes | None:
    # This is the accepted take, whatever the loop in generate_speech settled
    # on. Its presence means the block never needs TTS+ASR again.
    path = audio_shard_path(title, block_index)
    if not path.exists():
        return None
    return path.read_bytes()


def save_final_audio_shard(audio_bytes: bytes, title: str, block_index: int) -> None:
    AUDIO_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    audio_shard_path(title, block_index).write_bytes(audio_bytes)


def _word_diff(text: str, transcription: str) -> str:
    ref_words = WER_NORMALIZE(text)[0]
    hyp_words = WER_NORMALIZE(transcription)[0]
    diff = difflib.ndiff(ref_words, hyp_words)
    return " ".join(token for token in diff if not token.startswith("?"))


def generate_speech(
    text: str,
    title: str,
    block_index: int,
    language: str = "it",
    voice_id: str = "Leonardo.wav",
) -> bytes:
    cached = load_final_audio_shard(title, block_index)
    if cached is not None:
        logger.info("[%s] block %d: resumed audio from shard", title, block_index)
        return cached

    if _looks_like_code(text):
        audio_bytes = call_tts(text, language, voice_id)
        save_final_audio_shard(audio_bytes, title, block_index)
        return audio_bytes

    attempt = 0
    while True:
        audio_bytes = call_tts(text, language, voice_id)
        transcription = generate_transcription(audio_bytes, lang=language)

        wer = jiwer.wer(
            text,
            transcription,
            reference_transform=WER_NORMALIZE,
            hypothesis_transform=WER_NORMALIZE,
        )

        attempt += 1
        if wer <= WER_THRESHOLD:
            break

        if attempt >= MAX_TTS_ATTEMPTS:
            logger.warning(
                "High WER (%.2f%%) after %d attempts, giving up. "
                "Reference: %r Transcription: %r Diff: %s",
                wer * 100,
                attempt,
                text,
                transcription,
                _word_diff(text, transcription),
            )
            break

        logger.warning(
            "High WER (%.2f%%) on attempt %d/%d, retrying. "
            "Reference: %r Transcription: %r Diff: %s",
            wer * 100,
            attempt,
            MAX_TTS_ATTEMPTS,
            text,
            transcription,
            _word_diff(text, transcription),
        )

    save_final_audio_shard(audio_bytes, title, block_index)
    return audio_bytes


def generate_audio_from_blocks(documents, title: str) -> str:
    """Concatenate every block's audio into the episode the podcast serves."""
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = episode_path(title)
    audio_bytes = bytearray()

    for index, document in enumerate(documents.blocks):
        # speech_text is what the sanitizer produced; translated_text is the
        # fallback for a document that skipped that stage. An empty
        # speech_text means the block was invisible-only and has nothing to say.
        text = (
            document.speech_text if document.speech_text is not None else document.translated_text
        )
        if not text:
            continue
        audio_bytes.extend(generate_speech(text, title, index))
        logger.info("[%s] render_for_tts: block %d", title, index)

    dest_path.write_bytes(audio_bytes)
    return str(dest_path)
