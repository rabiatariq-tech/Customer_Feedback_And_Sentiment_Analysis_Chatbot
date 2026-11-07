"""Score a question-understanding planner against the evaluation sets in this folder.

    python evaluation/run_eval.py                          # rule-based planner (no network)
    python evaluation/run_eval.py --show                   # also print every failing case and who answered it
    python evaluation/run_eval.py --set heldout --show     # harder, untuned phrasings
    python evaluation/run_eval.py --set third              # the newest untuned set: run it once, report the number
    python evaluation/run_eval.py --set heldout --planner hybrid --max-calls 60 --show

With --planner hybrid|gemini the real Gemini key from .env is used. Calls are spaced out to stay under the free
tier's per-minute limit, a per-minute 429 is waited out and retried once, and answers are cached on disk (the
cache is thrown away when the prompt or model changes). Cases Gemini could not answer for API reasons (rate limit,
daily quota, network, call limit) are NOT counted as mistakes: they are reported as "not scored".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import fresh_cases, heldout_cases, planner_cases as pc, third_cases  # noqa: E402
from services import chat_parser  # noqa: E402

RATE_LIMIT_WAIT = 65          # seconds to wait after a per-minute 429 (the window is one minute)
SECONDS_BETWEEN_CALLS = 5     # 15 requests/minute is the free-tier limit; 5 s keeps a safe margin
SERVER_RETRY_WAIT = 10        # seconds to wait before retrying once after a 5xx ("model overloaded")
CACHE_META_KEY = "__prompt_hash__"
ACTION_INTENTS = ("help", "add_review", "save_review", "delete_review", "refuse", "review_prompt", "analyze_text", "import_help")
CASE_SETS = {"main": pc.CASES, "heldout": heldout_cases.CASES, "fresh": fresh_cases.CASES, "third": third_cases.CASES}


def query_to_plan(q) -> dict:
    return {
        "intent": q.intent, "sentiment": q.sentiment, "topic": q.topic, "product": q.product,
        "search": [t.lower() for t in q.search_terms],
        "period": (q.period.date_from, q.period.date_to) if q.period else None,
        "prev_period": (q.previous_period.date_from, q.previous_period.date_to) if q.previous_period else None,
        "granularity": q.granularity, "limit": q.limit, "review_id": q.review_id,
    }


def rule_plan(question: str) -> dict:
    """The rule-based planner's answer for one question, in the evaluation's plan format."""
    plan = query_to_plan(chat_parser.parse(question, pc.PRODUCTS, date.fromisoformat(pc.TODAY)))
    plan["source"] = "rules"
    return plan


def _prompt_hash(model: str) -> str:
    from services import planner as planner_mod  # read at call time so a changed prompt is noticed
    return hashlib.sha256(f"{model}\n{planner_mod.SYSTEM_PROMPT}".encode("utf-8")).hexdigest()[:16]


def ai_planner(mode: str, max_calls: int, cache_path: Path):
    """A planner backed by the real Gemini key (`mode` is "hybrid" or "llm_first"), with the same routing rule as
    the app: action intents never go to the AI, and in hybrid mode neither do questions the rules are sure about."""
    import config
    from services.llm.provider import GeminiProvider
    from services.llm.quota import QuotaGuard, is_daily_quota_error
    from services.planner import QuestionPlanner

    if not config.GEMINI_API_KEY or not config.GEMINI_MODEL:
        raise SystemExit("Set GEMINI_API_KEY and GEMINI_MODEL in .env first.")
    # The script paces itself, so the app's guards must not interfere: no per-minute cap, no daily cap.
    guard = QuotaGuard(daily_limit=10_000, session_limit=10_000, minute_limit=10_000)
    planner = QuestionPlanner(GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL, config.LLM_TIMEOUT_SECONDS), guard)

    fingerprint = _prompt_hash(config.GEMINI_MODEL)
    saved: dict = {}
    if cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            loaded = {}
        if loaded.get(CACHE_META_KEY) == fingerprint:  # answers given for a different prompt/model are not reused
            saved = loaded
    state = {"calls": 0, "cached": 0, "skipped": 0, "api_problems": 0, "daily_exhausted": False}
    today = date.fromisoformat(pc.TODAY)

    def save() -> None:
        saved[CACHE_META_KEY] = fingerprint
        cache_path.write_text(json.dumps(saved, indent=1), encoding="utf-8")

    def not_run(why: str) -> dict:
        return {"intent": f"(not run: {why})", "source": "none"}

    def ask(question: str):
        outcome = planner.plan(question, pc.PRODUCTS, today, "eval")
        detail = outcome.detail or ""
        if outcome.status == "error" and "429" in detail and not is_daily_quota_error(detail):
            time.sleep(RATE_LIMIT_WAIT)          # per-minute limit: wait for the window to reset, then retry once
            guard.clear_provider_pause()
            outcome = planner.plan(question, pc.PRODUCTS, today, "eval")
        elif outcome.status == "error" and "HTTP 5" in detail:
            time.sleep(SERVER_RETRY_WAIT)        # Google overloaded: retry once
            outcome = planner.plan(question, pc.PRODUCTS, today, "eval")
        return outcome

    def plan(question: str) -> dict:
        rules = chat_parser.parse(question, pc.PRODUCTS, today)
        if rules.intent in ACTION_INTENTS or (mode == "hybrid" and rules.confident):
            result = query_to_plan(rules)
            result["source"] = "rules"
            return result
        if question in saved and question != CACHE_META_KEY:
            state["cached"] += 1
            return {**saved[question], "source": "gemini (cached)"}
        if state["daily_exhausted"]:
            state["skipped"] += 1
            return not_run("daily Gemini quota used up")
        if state["calls"] >= max_calls:
            state["skipped"] += 1
            return not_run("call limit reached")
        if state["calls"]:
            time.sleep(SECONDS_BETWEEN_CALLS)
        state["calls"] += 1
        outcome = ask(question)
        if outcome.query is not None:
            result = query_to_plan(outcome.query)
            saved[question] = result
            save()
            return {**result, "source": "gemini"}
        detail = outcome.detail or ""
        if outcome.status == "invalid":           # Gemini answered, but with an unusable plan: a real miss
            return {"intent": f"(unusable plan: {detail})", "source": "gemini"}
        if is_daily_quota_error(detail):
            state["daily_exhausted"] = True
        state["api_problems"] += 1
        return not_run(f"{outcome.status} {detail}".strip())

    plan.state = state
    return plan


def _norm(value):
    """Lists and tuples compare equal (a period may come back from JSON as a list)."""
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    return value


def check(plan: dict, expected: dict) -> list[str]:
    """Fields that differ (empty list = pass)."""
    return [f"{k}: expected {v!r}, got {plan.get(k)!r}" for k, v in expected.items() if _norm(plan.get(k)) != _norm(v)]


def evaluate(planner=rule_plan, cases=None, info: dict | None = None) -> tuple[list[tuple[str, dict, list[str]]], dict]:
    """(failures, per-intent [passed, total]). Cases the planner could not run (source "none") are left out of
    both, and listed in info["unscored"]; info["sources"] maps each question to who answered it."""
    cases = cases if cases is not None else pc.CASES
    failures, per_intent = [], defaultdict(lambda: [0, 0])
    sources, unscored = {}, []
    for question, expected in cases:
        plan = planner(question)
        sources[question] = plan.get("source", "?")
        if plan.get("source") == "none":
            unscored.append(question)
            continue
        diffs = check(plan, expected)
        per_intent[expected["intent"]][1] += 1
        if diffs:
            failures.append((question, expected, diffs))
        else:
            per_intent[expected["intent"]][0] += 1
    if info is not None:
        info["sources"], info["unscored"] = sources, unscored
    return failures, dict(per_intent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print each failing case and who answered it")
    ap.add_argument("--set", choices=sorted(CASE_SETS), default="main", help="which evaluation set to score")
    ap.add_argument("--planner", choices=["rules", "hybrid", "gemini"], default="rules",
                    help="rules (default, no network), hybrid (rules + AI when unsure) or gemini (AI for everything except actions)")
    ap.add_argument("--max-calls", type=int, default=15, help="most Gemini calls to make in this run (free quota is small)")
    args = ap.parse_args()
    planner = rule_plan
    if args.planner != "rules":
        planner = ai_planner("hybrid" if args.planner == "hybrid" else "llm_first", args.max_calls,
                             Path(__file__).resolve().parent / ".llm_cache.json")
    cases = CASE_SETS[args.set]
    info: dict = {}
    failures, per_intent = evaluate(planner, cases=cases, info=info)
    scored = len(cases) - len(info["unscored"])
    passed = scored - len(failures)
    print(f"Planner accuracy ({args.planner}, {args.set} set): {passed}/{scored} ({passed / scored:.0%})" if scored else "Nothing was scored.")
    if info["unscored"]:
        print(f"NOT SCORED: {len(info['unscored'])} of {len(cases)} cases could not be answered by Gemini "
              "(rate limit, daily quota, network or the --max-calls limit). They are left out of the percentage.")
    if args.planner != "rules":
        st = planner.state
        print(f"Gemini calls this run: {st['calls']}; answers reused from the cache: {st['cached']}; "
              f"API problems: {st['api_problems']}; skipped: {st['skipped']}")
        if st["daily_exhausted"]:
            print("Google's DAILY quota is used up. Run again tomorrow: finished answers are cached.")
    print()
    print(f"{'intent':<14}{'passed':>7}{'total':>7}")
    for intent, (ok, n) in sorted(per_intent.items()):
        print(f"{intent:<14}{ok:>7}{n:>7}")
    if args.show:
        print("\nFailures:")
        for question, _, diffs in failures:
            print(f"- {question}   [{info['sources'].get(question, '?')}]")
            for d in diffs:
                print(f"    {d}")
        if info["unscored"]:
            print("\nNot scored:")
            for question in info["unscored"]:
                print(f"- {question}")


if __name__ == "__main__":
    main()
