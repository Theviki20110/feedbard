import base64
import os
import time

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.environ["MODEL_ID"]
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

_anthropic_client: anthropic.Anthropic | None = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(timeout=180.0)
    return _anthropic_client


def _extract_text(response: anthropic.types.Message) -> str:
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "Anthropic response truncated: hit max_tokens before finishing. "
            "Raise max_tokens or split the input."
        )
    return next(block.text for block in response.content if block.type == "text")


def _generate_with_ollama(prompt: str, image_bytes: bytes | None = None) -> str:
    payload: dict[str, object] = {
        "model": MODEL_ID,
        "prompt": prompt,
        "stream": False,
    }
    if image_bytes is not None:
        payload["images"] = [base64.b64encode(image_bytes).decode("ascii")]

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("done_reason") == "length":
        raise RuntimeError(
            "Ollama response truncated: the model reached its generation limit."
        )
    return result["response"]


def generate_response(prompt: str) -> tuple[str, float]:
    start = time.time()
    if LLM_PROVIDER == "ollama":
        text = _generate_with_ollama(prompt)
    elif LLM_PROVIDER == "anthropic":
        with _get_anthropic_client().messages.stream(
            model=MODEL_ID,
            max_tokens=64000,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            text = _extract_text(stream.get_final_message())
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; use 'anthropic' or 'ollama'"
        )
    elapsed = time.time() - start

    return text, elapsed


def generate_vision_response(prompt: str, image_bytes: bytes, media_type: str) -> tuple[str, float]:
    start = time.time()
    if LLM_PROVIDER == "ollama":
        text = _generate_with_ollama(prompt, image_bytes)
    elif LLM_PROVIDER == "anthropic":
        response = _get_anthropic_client().messages.create(
            model=MODEL_ID,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = _extract_text(response)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; use 'anthropic' or 'ollama'"
        )
    elapsed = time.time() - start

    return text, elapsed
