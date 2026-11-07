"""Check that your Gemini key and model name work. Uses one request from the free quota.

    python scripts/check_gemini.py
"""
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from services.llm.provider import GeminiProvider  # noqa: E402


def main() -> None:
    if not config.GEMINI_API_KEY or not config.GEMINI_MODEL:
        print("GEMINI_API_KEY and GEMINI_MODEL must both be set in your .env file.")
        raise SystemExit(1)
    provider = GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL, config.LLM_TIMEOUT_SECONDS)
    started = time.perf_counter()
    result = provider.generate_json('Reply with JSON only: {"ok": true}', "ping", max_output_tokens=20)
    ping_seconds = time.perf_counter() - started
    if not result.ok:
        print(f"FAILED: {result.error}")
        print("HTTP 400/404: check the model name. HTTP 403/401: check the key. HTTP 429: the free quota is used up for now.")
        raise SystemExit(1)
    print(f"OK: model {config.GEMINI_MODEL} answered in {ping_seconds:.1f} s. Reply: {result.data}")

    # The same kind of request the chatbot makes for every question in PLANNER_MODE=llm_first (uses one more request).
    from services.llm.quota import QuotaGuard
    from services.planner import QuestionPlanner

    planner = QuestionPlanner(provider, QuotaGuard(daily_limit=5, session_limit=5, minute_limit=5))
    started = time.perf_counter()
    outcome = planner.plan("did negative reviews go up or down last month", ["Desk Lamp"], date.today(), "check")
    plan_seconds = time.perf_counter() - started
    print(f"A real question-understanding request took {plan_seconds:.1f} s (status: {outcome.status}).")
    print(f"PLANNER_MODE={config.PLANNER_MODE}: " + (
        "every question that is not an action waits about that long. If it feels slow, use PLANNER_MODE=hybrid "
        "(rules answer the clear questions instantly, the AI only the unsure ones)." if config.PLANNER_MODE == "llm_first" else
        "only questions the rules are unsure about wait for the AI."))
    print(f"LLM_TIMEOUT_SECONDS={config.LLM_TIMEOUT_SECONDS:g}: a request slower than this is given up on and the rules answer instead.")


if __name__ == "__main__":
    main()
