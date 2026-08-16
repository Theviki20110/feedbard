import logging
from types import SimpleNamespace

import pytest

from feedbard.pipeline import audio_renderer
from feedbard.pipeline.audio_renderer import (
    _looks_like_code,
    call_tts,
    call_tts_http,
    call_tts_polly,
    generate_audio_from_blocks,
    generate_speech,
)


def _unreachable(*a, **kw):
    raise AssertionError("should not have been called")


# --------------------------------------------------------------------------
# call_tts_http
# --------------------------------------------------------------------------


def test_call_tts_http_posts_the_configured_language_and_voice(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen["url"], seen["json"] = url, json
        return SimpleNamespace(raise_for_status=lambda: None, content=b"wav-bytes")

    monkeypatch.setattr(audio_renderer, "TTS_BASE_URL", "http://tts.local")
    monkeypatch.setattr(audio_renderer.requests, "post", fake_post)

    out = call_tts_http("ciao mondo")

    assert out == b"wav-bytes"
    assert seen["url"] == "http://tts.local/tts"
    assert seen["json"]["language"] == audio_renderer.TTS_LANGUAGE_CODE
    assert seen["json"]["predefined_voice_id"] == audio_renderer.TTS_VOICE_ID
    assert seen["json"]["text"] == "ciao mondo"


def test_call_tts_http_accepts_a_voice_override(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen["json"] = json
        return SimpleNamespace(raise_for_status=lambda: None, content=b"")

    monkeypatch.setattr(audio_renderer.requests, "post", fake_post)

    call_tts_http("testo", voice_id="Altra.wav")
    assert seen["json"]["predefined_voice_id"] == "Altra.wav"


# --------------------------------------------------------------------------
# call_tts_polly
# --------------------------------------------------------------------------


class _FakePollyClient:
    def __init__(self, audio_bytes):
        self._audio_bytes = audio_bytes
        self.kwargs = None

    def synthesize_speech(self, **kwargs):
        self.kwargs = kwargs
        return {"AudioStream": SimpleNamespace(read=lambda: self._audio_bytes)}


def test_call_tts_polly_uses_the_configured_voice_and_language(monkeypatch):
    client = _FakePollyClient(b"mp3-bytes")
    monkeypatch.setattr(audio_renderer, "_get_polly_client", lambda: client)

    out = call_tts_polly("ciao mondo")

    assert out == b"mp3-bytes"
    assert client.kwargs["VoiceId"] == audio_renderer.POLLY_VOICE_ID
    assert client.kwargs["LanguageCode"] == audio_renderer.POLLY_LANGUAGE_CODE
    assert client.kwargs["Engine"] == audio_renderer.POLLY_ENGINE
    assert client.kwargs["Text"] == "ciao mondo"


# --------------------------------------------------------------------------
# call_tts: provider dispatch
# --------------------------------------------------------------------------


def test_call_tts_dispatches_to_http(monkeypatch):
    monkeypatch.setattr(audio_renderer, "TTS_PROVIDER", "http")
    monkeypatch.setattr(audio_renderer, "call_tts_http", lambda text, voice_id: b"http")
    monkeypatch.setattr(audio_renderer, "call_tts_polly", _unreachable)
    assert call_tts("testo") == b"http"


def test_call_tts_dispatches_to_polly(monkeypatch):
    monkeypatch.setattr(audio_renderer, "TTS_PROVIDER", "polly")
    monkeypatch.setattr(audio_renderer, "call_tts_polly", lambda text: b"polly")
    monkeypatch.setattr(audio_renderer, "call_tts_http", _unreachable)
    assert call_tts("testo") == b"polly"


def test_call_tts_unsupported_provider_raises(monkeypatch):
    monkeypatch.setattr(audio_renderer, "TTS_PROVIDER", "sagemaker")
    with pytest.raises(ValueError, match="Unsupported TTS_PROVIDER"):
        call_tts("testo")


# --------------------------------------------------------------------------
# _looks_like_code
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "vedi https://example.com/docs",
        "$ uv run pytest",
        "```python\nprint(1)\n```",
        "--model=north-mini",
        "contattami a someone@example.com",
        "il file è in /usr/local/bin/feedbard",
        "modifica config.toml",
    ],
)
def test_looks_like_code_true_for_code_shaped_text(text):
    assert _looks_like_code(text)


def test_looks_like_code_false_for_plain_prose():
    assert not _looks_like_code("Il modello raggiunge il 92% di accuratezza sul benchmark.")


# --------------------------------------------------------------------------
# generate_speech
# --------------------------------------------------------------------------


@pytest.fixture
def shard(monkeypatch, tmp_path):
    def shard_path(title, index):
        return tmp_path / f"{title}_{index}.wav"

    monkeypatch.setattr(audio_renderer, "AUDIO_SHARDS_DIR", tmp_path)
    monkeypatch.setattr(audio_renderer, "audio_shard_path", shard_path)
    return tmp_path


def test_resumed_from_shard_skips_tts_entirely(monkeypatch, shard):
    (shard / "Articolo_0.wav").write_bytes(b"cached-audio")
    monkeypatch.setattr(audio_renderer, "call_tts", _unreachable)

    out = generate_speech("qualsiasi testo", "Articolo", 0)
    assert out == b"cached-audio"


def test_code_like_text_skips_the_wer_check(monkeypatch, shard):
    monkeypatch.setattr(audio_renderer, "call_tts", lambda text, voice_id: b"audio")
    monkeypatch.setattr(audio_renderer, "generate_transcription", _unreachable)

    out = generate_speech("uv run pytest --model=x", "Articolo", 0)
    assert out == b"audio"
    assert (shard / "Articolo_0.wav").read_bytes() == b"audio"


def test_matching_transcription_logs_no_warning(monkeypatch, shard, caplog):
    text = "Il modello raggiunge una buona accuratezza sul benchmark di riferimento."
    monkeypatch.setattr(audio_renderer, "call_tts", lambda t, voice_id: b"audio")
    monkeypatch.setattr(audio_renderer, "generate_transcription", lambda audio, lang: text)

    with caplog.at_level(logging.WARNING):
        generate_speech(text, "Articolo", 0)
    assert "High WER" not in caplog.text


def test_mismatched_transcription_logs_a_high_wer_warning(monkeypatch, shard, caplog):
    text = "Il modello raggiunge una buona accuratezza sul benchmark di riferimento."
    monkeypatch.setattr(audio_renderer, "call_tts", lambda t, voice_id: b"audio")
    monkeypatch.setattr(
        audio_renderer, "generate_transcription", lambda audio, lang: "parole completamente diverse"
    )

    with caplog.at_level(logging.WARNING):
        generate_speech(text, "Articolo", 0)
    assert "High WER" in caplog.text


def test_audio_is_still_saved_even_on_a_high_wer(monkeypatch, shard):
    text = "Il modello raggiunge una buona accuratezza sul benchmark di riferimento."
    monkeypatch.setattr(audio_renderer, "call_tts", lambda t, voice_id: b"audio-bytes")
    monkeypatch.setattr(
        audio_renderer, "generate_transcription", lambda audio, lang: "tutt'altra cosa"
    )

    generate_speech(text, "Articolo", 0)
    assert (shard / "Articolo_0.wav").read_bytes() == b"audio-bytes"


def test_asr_language_matches_the_configured_tts_language_code(monkeypatch, shard):
    seen = {}

    def generate_transcription(audio, lang):
        seen["lang"] = lang
        return "testo"

    monkeypatch.setattr(audio_renderer, "call_tts", lambda t, voice_id: b"audio")
    monkeypatch.setattr(audio_renderer, "generate_transcription", generate_transcription)

    generate_speech("un testo qualunque abbastanza lungo da non sembrare codice", "Articolo", 0)
    assert seen["lang"] == audio_renderer.TTS_LANGUAGE_CODE


# --------------------------------------------------------------------------
# generate_audio_from_blocks
# --------------------------------------------------------------------------


@pytest.fixture
def episode(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_renderer, "EPISODES_DIR", tmp_path)
    monkeypatch.setattr(audio_renderer, "episode_path", lambda title: tmp_path / f"{title}.mp3")
    return tmp_path


def _block(speech_text=None, translated_text=None):
    return SimpleNamespace(speech_text=speech_text, translated_text=translated_text)


def test_blocks_are_concatenated_in_order(monkeypatch, episode):
    calls = []

    def generate_speech(text, title, index):
        calls.append(index)
        return f"[{index}]".encode()

    monkeypatch.setattr(audio_renderer, "generate_speech", generate_speech)
    blocks = [_block(speech_text="a"), _block(speech_text="b"), _block(speech_text="c")]
    doc = SimpleNamespace(blocks=blocks)

    dest = generate_audio_from_blocks(doc, "Articolo")

    assert calls == [0, 1, 2]
    assert open(dest, "rb").read() == b"[0][1][2]"


def test_prefers_speech_text_over_translated_text(monkeypatch, episode):
    seen = []

    def generate_speech(text, title, index):
        seen.append(text)
        return b"x"

    monkeypatch.setattr(audio_renderer, "generate_speech", generate_speech)
    doc = SimpleNamespace(blocks=[_block(speech_text="finale", translated_text="grezzo")])

    generate_audio_from_blocks(doc, "Articolo")
    assert seen == ["finale"]


def test_falls_back_to_translated_text_when_sanitizer_was_skipped(monkeypatch, episode):
    seen = []

    def generate_speech(text, title, index):
        seen.append(text)
        return b"x"

    monkeypatch.setattr(audio_renderer, "generate_speech", generate_speech)
    doc = SimpleNamespace(blocks=[_block(speech_text=None, translated_text="grezzo")])

    generate_audio_from_blocks(doc, "Articolo")
    assert seen == ["grezzo"]


def test_blocks_with_nothing_to_say_are_skipped(monkeypatch, episode):
    calls = []

    def generate_speech(text, title, index):
        calls.append(index)
        return b"x"

    monkeypatch.setattr(audio_renderer, "generate_speech", generate_speech)
    blocks = [
        _block(speech_text=""),  # invisible-only, nothing to say
        _block(speech_text=None, translated_text=None),
        _block(speech_text="testo"),
    ]
    doc = SimpleNamespace(blocks=blocks)

    generate_audio_from_blocks(doc, "Articolo")
    assert calls == [2]
