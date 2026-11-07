"""Check that your Gemini key and model name work. Uses one request from the free quota.

    python scripts/check_gemini.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from services.llm.provider import GeminiProvider  # noqa: E402


def main() -> None:
    if not config.GEMINI_API_KEY or not config.GEMINI_MODEL:
        print("GEMINI_API_KEY and GEMINI_MODEL must both be set in your .env file.")
        raise SystemExit(1)
    result = GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL, config.LLM_TIMEOUT_SECONDS).generate_json(
        'Reply with JSON only: {"ok": true}', "ping", max_output_tokens=20)
    if result.ok:
        print(f"OK: model {config.GEMINI_MODEL} answered. Reply: {result.data}")
    else:
        print(f"FAILED: {result.error}")
        print("HTTP 400/404: check the model name. HTTP 403/401: check the key. HTTP 429: the free quota is used up for now.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
