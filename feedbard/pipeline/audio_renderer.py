import difflib
import os

import boto3
import jiwer
import requests
from dotenv import load_dotenv

from feedbard.asr_client import generate_transcription
from feedbard.language import LANGUAGE_CODE, LANGUAGE_NAME, LANGUAGE_TAG
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

WER_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
POLLY_ENGINE = os.getenv("POLLY_ENGINE", "generative")

# Polly wants a BCP-47 tag ("it-IT"); the HTTP TTS server and the ASR QA step
# below both want the bare short code ("it"). Both are derived from
# TARGET_LANGUAGE by `feedbard.language`, so changing the narration language
# moves them with it. The overrides remain for the cases derivation cannot
# cover: a regional variant the tag does not carry, or a provider that names
# a code differently.
POLLY_LANGUAGE_CODE = os.getenv("POLLY_LANGUAGE_CODE") or LANGUAGE_TAG
TTS_LANGUAGE_CODE = os.getenv("TTS_LANGUAGE_CODE") or LANGUAGE_CODE

# A voice is an identity, not a language: nothing derives "which voice" from
# "which language", so these stay explicit. The defaults are Italian, which is
# the one thing here that cannot follow TARGET_LANGUAGE on its own -- so an
# operator who moved the language and forgot the voice gets told, rather than
# an episode read in the right words with the wrong accent.
_DEFAULT_VOICE_LANGUAGE = "Italian"
POLLY_VOICE_ID = os.getenv("POLLY_VOICE_ID", "Bianca")
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "Leonardo.wav")

if LANGUAGE_NAME != _DEFAULT_VOICE_LANGUAGE and not (
    os.getenv("POLLY_VOICE_ID") if TTS_PROVIDER == "polly" else os.getenv("TTS_VOICE_ID")
):
    logger.warning(
        "audio_renderer: narrating in %s but TTS_PROVIDER=%s is still using its default "
        "%s voice (%s). Set %s to a voice for %s.",
        LANGUAGE_NAME,
        TTS_PROVIDER,
        _DEFAULT_VOICE_LANGUAGE,
        POLLY_VOICE_ID if TTS_PROVIDER == "polly" else TTS_VOICE_ID,
        "POLLY_VOICE_ID" if TTS_PROVIDER == "polly" else "TTS_VOICE_ID",
        LANGUAGE_NAME,
    )

_polly_client = None


def _get_polly_client():
    global _polly_client
    if _polly_client is None:
        _polly_client = boto3.client("polly", region_name=AWS_REGION)
    return _polly_client


def call_tts_http(text: str, voice_id: str = TTS_VOICE_ID) -> bytes:
    body = {
        "voice_mode": "predefined",
        "predefined_voice_id": voice_id,
        "output_format": "wav",
        "split_text": True,
        "text": text,
        "language": TTS_LANGUAGE_CODE,
    }
    response = requests.post(
        f"{TTS_BASE_URL}/tts",
        json=body,
        timeout=180,
    )
    response.raise_for_status()
    return response.content


def call_tts_polly(text: str) -> bytes:
    response = _get_polly_client().synthesize_speech(
        Text=text,
        VoiceId=POLLY_VOICE_ID,
        OutputFormat="mp3",
        Engine=POLLY_ENGINE,
        LanguageCode=POLLY_LANGUAGE_CODE,
    )
    return response["AudioStream"].read()


def call_tts(text: str, voice_id: str = TTS_VOICE_ID) -> bytes:
    if TTS_PROVIDER == "polly":
        return call_tts_polly(text)
    if TTS_PROVIDER == "http":
        return call_tts_http(text, voice_id)
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
    voice_id: str = TTS_VOICE_ID,
) -> bytes:
    cached = load_final_audio_shard(title, block_index)
    if cached is not None:
        logger.info("[%s] block %d: resumed audio from shard", title, block_index)
        return cached

    audio_bytes = call_tts(text, voice_id)
    transcription = generate_transcription(audio_bytes, lang=TTS_LANGUAGE_CODE)

    wer = jiwer.wer(
        text,
        transcription,
        reference_transform=WER_NORMALIZE,
        hypothesis_transform=WER_NORMALIZE,
    )

    if wer > WER_THRESHOLD:
        logger.warning(
            "High WER (%.2f%%). Reference: %r Transcription: %r Diff: %s",
            wer * 100,
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
