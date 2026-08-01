from pathlib import Path
from jinja2 import Template
from bedrock_client import generate_response

TRANSLATOR_PROMPT_PATH = Path(__file__).parent / "assets" / "translator_prompt.txt"

def translate_text(text: str, target_language: str = "Italian") -> str:
    prompt = Template(TRANSLATOR_PROMPT_PATH.read_text()).render(SOURCE_TEXT=text, TARGET_LANGUAGE=target_language)
    translated_text, elapsed_time = generate_response(prompt)
    print(f"Translation to {target_language} completed in {elapsed_time:.2f} seconds")
    return translated_text
