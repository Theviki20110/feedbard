import json
import time

import boto3
from botocore.config import Config

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

_config = Config(connect_timeout=5, read_timeout=180)
_client = boto3.client("bedrock-runtime", config=_config, region_name="us-east-1")


def generate_response(prompt: str) -> tuple[str, float]:
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16384,
        "messages": [{"role": "user", "content": prompt}],
    }

    start = time.time()
    response = _client.invoke_model(modelId=MODEL_ID, body=json.dumps(request_body))
    elapsed = time.time() - start

    response_body = json.loads(response.get("body").read())
    return response_body["content"][0]["text"], elapsed
