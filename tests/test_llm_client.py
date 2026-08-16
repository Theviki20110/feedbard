"""Provider dispatch for generate_response / generate_vision_response.

None of these hit a real network or SDK: each provider branch is exercised by
faking just enough of its client's shape (Anthropic's `messages.stream`
context manager, Bedrock's `converse` dict, Ollama's HTTP JSON) to prove the
dispatch, the truncation guard, and the max_tokens budget are wired correctly.
"""

from types import SimpleNamespace

import pytest

from feedbard import llm_client


@pytest.fixture(autouse=True)
def _model_id(monkeypatch):
    # Every path is gated on MODEL_ID being set; give each test a real one and
    # let the one test that cares about the missing case override it.
    monkeypatch.setattr(llm_client, "MODEL_ID", "test-model")


# --------------------------------------------------------------------------
# _require_model_id
# --------------------------------------------------------------------------


def test_missing_model_id_raises_before_any_call(monkeypatch):
    monkeypatch.setattr(llm_client, "MODEL_ID", "")
    with pytest.raises(RuntimeError, match="MODEL_ID"):
        llm_client.generate_response("prompt")


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


def test_ollama_generate_response_posts_the_prompt(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client, "OLLAMA_BASE_URL", "http://ollama.local")
    seen = {}

    def fake_post(url, json, timeout):
        seen["url"], seen["json"] = url, json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "ciao", "done_reason": "stop"},
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    text, elapsed = llm_client.generate_response("scrivi ciao")

    assert text == "ciao"
    assert elapsed >= 0
    assert seen["url"] == "http://ollama.local/api/generate"
    assert seen["json"]["prompt"] == "scrivi ciao"
    assert seen["json"]["model"] == "test-model"
    assert "images" not in seen["json"]


def test_ollama_truncation_raises(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *a, **kw: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "tronc", "done_reason": "length"},
        ),
    )
    with pytest.raises(RuntimeError, match="truncated"):
        llm_client.generate_response("prompt")


def test_ollama_vision_encodes_the_image(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "ollama")
    seen = {}

    def fake_post(url, json, timeout):
        seen["json"] = json
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": "una figura", "done_reason": "stop"},
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    text, _ = llm_client.generate_vision_response("descrivi", b"\x89PNG\r\n", "image/png")

    assert text == "una figura"
    assert len(seen["json"]["images"]) == 1
    assert isinstance(seen["json"]["images"][0], str)  # base64, not raw bytes


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, message):
        self._message = message
        self.stream_kwargs = None
        self.create_kwargs = None

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        return _FakeStream(self._message)

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return self._message


class _FakeAnthropicClient:
    def __init__(self, message):
        self.messages = _FakeMessages(message)


def _text_message(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_anthropic_generate_response_streams_and_extracts_text(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "anthropic")
    client = _FakeAnthropicClient(_text_message("risposta"))
    monkeypatch.setattr(llm_client, "_get_anthropic_client", lambda: client)

    text, _ = llm_client.generate_response("domanda")

    assert text == "risposta"
    assert client.messages.stream_kwargs["model"] == "test-model"
    assert client.messages.stream_kwargs["max_tokens"] == 64000
    assert client.messages.stream_kwargs["messages"] == [{"role": "user", "content": "domanda"}]


def test_anthropic_truncated_response_raises(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "anthropic")
    client = _FakeAnthropicClient(_text_message("tronc", stop_reason="max_tokens"))
    monkeypatch.setattr(llm_client, "_get_anthropic_client", lambda: client)

    with pytest.raises(RuntimeError, match="truncated"):
        llm_client.generate_response("domanda")


def test_anthropic_vision_sends_image_and_text_blocks(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "anthropic")
    client = _FakeAnthropicClient(_text_message("una figura"))
    monkeypatch.setattr(llm_client, "_get_anthropic_client", lambda: client)

    text, _ = llm_client.generate_vision_response("descrivi", b"rawbytes", "image/jpeg")

    assert text == "una figura"
    kwargs = client.messages.create_kwargs
    assert kwargs["max_tokens"] == 1024
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1] == {"type": "text", "text": "descrivi"}


# --------------------------------------------------------------------------
# Bedrock
# --------------------------------------------------------------------------


class _FakeBedrockClient:
    def __init__(self, response):
        self._response = response
        self.converse_kwargs = None

    def converse(self, **kwargs):
        self.converse_kwargs = kwargs
        return self._response


def _bedrock_response(text, stop_reason="end_turn"):
    return {
        "stopReason": stop_reason,
        "output": {"message": {"content": [{"text": text}]}},
    }


def test_bedrock_generate_response_extracts_text(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "bedrock")
    client = _FakeBedrockClient(_bedrock_response("risposta"))
    monkeypatch.setattr(llm_client, "_get_bedrock_client", lambda: client)

    text, _ = llm_client.generate_response("domanda")

    assert text == "risposta"
    assert client.converse_kwargs["modelId"] == "test-model"
    assert client.converse_kwargs["inferenceConfig"]["maxTokens"] == 64000
    assert client.converse_kwargs["messages"] == [
        {"role": "user", "content": [{"text": "domanda"}]}
    ]


def test_bedrock_truncated_response_raises(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "bedrock")
    client = _FakeBedrockClient(_bedrock_response("tronc", stop_reason="max_tokens"))
    monkeypatch.setattr(llm_client, "_get_bedrock_client", lambda: client)

    with pytest.raises(RuntimeError, match="truncated"):
        llm_client.generate_response("domanda")


def test_bedrock_vision_sends_image_bytes_with_format_from_media_type(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "bedrock")
    client = _FakeBedrockClient(_bedrock_response("una figura"))
    monkeypatch.setattr(llm_client, "_get_bedrock_client", lambda: client)

    text, _ = llm_client.generate_vision_response("descrivi", b"rawbytes", "image/png")

    assert text == "una figura"
    content = client.converse_kwargs["messages"][0]["content"]
    assert content[0]["image"]["format"] == "png"
    assert content[0]["image"]["source"]["bytes"] == b"rawbytes"
    assert client.converse_kwargs["inferenceConfig"]["maxTokens"] == 1024


# --------------------------------------------------------------------------
# Unsupported provider
# --------------------------------------------------------------------------


def test_unsupported_provider_raises_on_text(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        llm_client.generate_response("prompt")


def test_unsupported_provider_raises_on_vision(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        llm_client.generate_vision_response("prompt", b"x", "image/png")
