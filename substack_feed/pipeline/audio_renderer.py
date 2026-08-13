import difflib
import os

import boto3
import jiwer
import requests
from dotenv import load_dotenv

from substack_feed.asr_client import generate_transcription
from substack_feed.logger import logger
from substack_feed.paths import safe_filename

load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "http").lower()
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "")
AUDIO_DIR = os.environ["AUDIO_DIR"]
AUDIO_SHARDS_DIR = os.path.join(AUDIO_DIR, "audio_shards")
WER_THRESHOLD = 0.2  # 20% WER threshold for logging warnings
MAX_TTS_ATTEMPTS = 3

WER_NORMALIZE = jiwer.Compose(
    [
        jiwer.SubstituteRegexes({r"\s+": " "}),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

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


def save_audio_shard(audio_bytes: bytes, safe_title: str, block_index: int, attempt: int) -> str:
    os.makedirs(AUDIO_SHARDS_DIR, exist_ok=True)
    shard_path = os.path.join(
        AUDIO_SHARDS_DIR, f"{safe_title}_block{block_index}_attempt{attempt}.wav"
    )
    with open(shard_path, "wb") as f:
        f.write(audio_bytes)
    return shard_path


def _final_shard_path(safe_title: str, block_index: int) -> str:
    # No "attemptN" suffix: this is the accepted take (whatever the loop in
    # generate_speech settled on), separate from the per-attempt debug shards
    # above. Its presence means the block never needs TTS+ASR again.
    return os.path.join(AUDIO_SHARDS_DIR, f"{safe_title}_block{block_index}.wav")


def load_final_audio_shard(safe_title: str, block_index: int) -> bytes | None:
    path = _final_shard_path(safe_title, block_index)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def save_final_audio_shard(audio_bytes: bytes, safe_title: str, block_index: int) -> None:
    os.makedirs(AUDIO_SHARDS_DIR, exist_ok=True)
    with open(_final_shard_path(safe_title, block_index), "wb") as f:
        f.write(audio_bytes)


def _word_diff(text: str, transcription: str) -> str:
    ref_words = WER_NORMALIZE(text)[0]
    hyp_words = WER_NORMALIZE(transcription)[0]
    diff = difflib.ndiff(ref_words, hyp_words)
    return " ".join(token for token in diff if not token.startswith("?"))


def generate_speech(
    text: str,
    safe_title: str,
    block_index: int,
    language: str = "it",
    voice_id: str = "Leonardo.wav",
) -> bytes:
    cached = load_final_audio_shard(safe_title, block_index)
    if cached is not None:
        logger.info("[%s] block %d: resumed audio from shard", safe_title, block_index)
        return cached

    attempt = 0
    while True:
        audio_bytes = call_tts(text, language, voice_id)
        save_audio_shard(audio_bytes, safe_title, block_index, attempt)
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

    save_final_audio_shard(audio_bytes, safe_title, block_index)
    return audio_bytes


def generate_audio_from_blocks(documents, title: str, dest_dir: str = AUDIO_DIR) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    safe_title = safe_filename(title)
    dest_path = os.path.join(dest_dir, safe_title + ".mp3")
    audio_bytes = bytearray()

    for index, document in enumerate(documents.blocks):
        if document.translated_text is None:
            continue
        audio_bytes.extend(generate_speech(document.translated_text, safe_title, index))
        logger.info("[%s] render_for_tts: block %d", title, index)

    with open(dest_path, "wb") as f:
        f.write(audio_bytes)
    return dest_path
