import base64
import os
import time

import anthropic
import boto3
import requests
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_RETRY_CONFIG = Config(retries={"max_attempts": 10, "mode": "adaptive"})

_anthropic_client: anthropic.Anthropic | None = None
_bedrock_client = None


def _require_model_id() -> None:
    """Checked at call time rather than import time so a misconfigured
    container fails with an actionable message instead of a KeyError in a
    module nothing has asked to use yet."""
    if not MODEL_ID:
        raise RuntimeError(
            f"MODEL_ID is not set; it must name a model for LLM_PROVIDER={LLM_PROVIDER!r} "
            "(Bedrock inference profile ARN, Anthropic model name, or Ollama tag)"
        )


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(timeout=180.0)
    return _anthropic_client


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime", region_name=AWS_REGION, config=BEDROCK_RETRY_CONFIG
        )
    return _bedrock_client


def _extract_text(response: anthropic.types.Message) -> str:
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "Anthropic response truncated: hit max_tokens before finishing. "
            "Raise max_tokens or split the input."
        )
    return next(block.text for block in response.content if block.type == "text")


def _extract_text_bedrock(response: dict) -> str:
    if response.get("stopReason") == "max_tokens":
        raise RuntimeError(
            "Bedrock response truncated: hit max_tokens before finishing. "
            "Raise max_tokens or split the input."
        )
    return next(
        block["text"] for block in response["output"]["message"]["content"] if "text" in block
    )


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
        raise RuntimeError("Ollama response truncated: the model reached its generation limit.")
    return result["response"]


def generate_response(prompt: str) -> tuple[str, float]:
    _require_model_id()
    start = time.time()
    if LLM_PROVIDER == "ollama":
        text = _generate_with_ollama(prompt)
    elif LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        with client.messages.stream(
            model=MODEL_ID,
            max_tokens=64000,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            text = _extract_text(stream.get_final_message())
    elif LLM_PROVIDER == "bedrock":
        client = _get_bedrock_client()
        response = client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 64000},
        )
        text = _extract_text_bedrock(response)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; use 'anthropic', 'bedrock', or 'ollama'"
        )
    elapsed = time.time() - start

    return text, elapsed


def generate_vision_response(prompt: str, image_bytes: bytes, media_type: str) -> tuple[str, float]:
    _require_model_id()
    start = time.time()
    if LLM_PROVIDER == "ollama":
        text = _generate_with_ollama(prompt, image_bytes)
    elif LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=1024,
            messages=[
                {
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
                }
            ],
        )
        text = _extract_text(response)
    elif LLM_PROVIDER == "bedrock":
        client = _get_bedrock_client()
        image_format = media_type.split("/")[-1]
        response = client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 1024},
        )
        text = _extract_text_bedrock(response)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; use 'anthropic', 'bedrock', or 'ollama'"
        )
    elapsed = time.time() - start

    return text, elapsed
