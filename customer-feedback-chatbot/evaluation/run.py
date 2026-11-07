"""Score a question-understanding function against the gold set.

    python -m evaluation.run            # rule-based parser
Prints accuracy overall and per intent, and every miss, so rule changes can be judged by numbers.
A planner (for example the Gemini one) is scored the same way by passing any function question -> plan dict.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from typing import Callable

from evaluation.questions import CASES, HARD, PRODUCTS, TODAY
from services import chat_parser

FIELDS = ("sentiment", "topic", "product", "limit")


def rule_plan(question: str) -> dict:
    q = chat_parser.parse(question, PRODUCTS, TODAY)
    return {
        "intent": q.intent, "sentiment": q.sentiment, "topic": q.topic, "product": q.product, "limit": q.limit,
        "period": (q.period.date_from, q.period.date_to) if q.period else None,
        "granularity": q.granularity, "group_by": q.group_by, "terms": list(q.search_terms),
    }


def check(plan: dict, expected: dict) -> list[str]:
    """Reasons the plan differs from the gold answer (empty list = correct)."""
    problems = []
    if plan.get("intent") != expected["intent"]:
        problems.append(f"intent {plan.get('intent')!r} != {expected['intent']!r}")
    if expected["intent"] in ("help", "add_review", "analyze_text", "delete_review", "refuse", "unknown"):
        return problems  # no filters to judge
    for f in FIELDS:
        if plan.get(f) != expected.get(f):
            problems.append(f"{f} {plan.get(f)!r} != {expected.get(f)!r}")
    if plan.get("period") != expected.get("period"):
        problems.append(f"period {plan.get('period')} != {expected.get('period')}")
    if sorted(plan.get("terms") or []) != sorted(expected.get("terms") or []):
        problems.append(f"terms {plan.get('terms')} != {expected.get('terms') or []}")
    for f in ("granularity", "group_by"):
        if f in expected and plan.get(f) != expected[f]:
            problems.append(f"{f} {plan.get(f)!r} != {expected[f]!r}")
    return problems


def evaluate(plan_fn: Callable[[str], dict], cases=None) -> dict:
    cases = CASES if cases is None else cases
    misses, per_intent = [], defaultdict(lambda: [0, 0])
    for question, expected in cases:
        problems = check(plan_fn(question), expected)
        per_intent[expected["intent"]][1] += 1
        if problems:
            misses.append((question, problems))
        else:
            per_intent[expected["intent"]][0] += 1
    total = len(cases)
    return {"correct": total - len(misses), "total": total, "misses": misses, "per_intent": dict(per_intent)}


def main() -> None:
    for name, cases in (("core", CASES), ("hard", HARD)):
        r = evaluate(rule_plan, cases)
        print(f"\n== {name}: {r['correct']}/{r['total']} = {r['correct'] / r['total']:.0%}")
        if name == "core":
            for intent, (ok, n) in sorted(r["per_intent"].items()):
                print(f"   {intent:14s} {ok}/{n}")
        for question, problems in r["misses"]:
            print(f"  MISS  {question}\n        " + "; ".join(problems))


if __name__ == "__main__":
    sys.exit(main())
