import base64
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.environ["MODEL_ID"]

_client = anthropic.Anthropic(timeout=180.0)


def _extract_text(response: anthropic.types.Message) -> str:
    return next(block.text for block in response.content if block.type == "text")


def generate_response(prompt: str) -> tuple[str, float]:
    start = time.time()
    response = _client.messages.create(
        model=MODEL_ID,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start

    return _extract_text(response), elapsed


def generate_vision_response(prompt: str, image_bytes: bytes, media_type: str) -> tuple[str, float]:
    start = time.time()
    response = _client.messages.create(
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
    elapsed = time.time() - start

    return _extract_text(response), elapsed
