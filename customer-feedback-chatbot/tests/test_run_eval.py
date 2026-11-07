"""The Gemini branch of evaluation/run_eval.py, tested with a fake provider (no network, no sleeping)."""
import json

import pytest

import config
from evaluation import planner_cases as pc, run_eval
from services.llm.provider import LLMResult


class Fake:
    """Answers 429 once if asked to, then a fixed plan."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls, self.fail_first = 0, Fake.fail_first
        Fake.instances.append(self)

    def generate_json(self, system, user, max_output_tokens=400):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return LLMResult(False, error="HTTP 429: GenerateRequestsPerMinutePerProjectPerModel-FreeTier", quota_exhausted=True)
        return LLMResult(True, data={"intent": "count", "sentiment": "negative"})


@pytest.fixture()
def fake_gemini(monkeypatch):
    sleeps = []
    Fake.instances, Fake.fail_first = [], False
    monkeypatch.setattr(config, "GEMINI_API_KEY", "key")
    monkeypatch.setattr(config, "GEMINI_MODEL", "model-a")
    monkeypatch.setattr("services.llm.provider.GeminiProvider", Fake)
    monkeypatch.setattr(run_eval.time, "sleep", sleeps.append)
    return sleeps


UNSURE = "tell me about stuff"       # rules are not confident, so hybrid asks Gemini


def test_gemini_answers_are_cached_and_spaced_out(fake_gemini, tmp_path):
    cache = tmp_path / "cache.json"
    plan = run_eval.ai_planner("hybrid", 5, cache)
    first = plan(UNSURE)
    assert first["intent"] == "count" and first["source"] == "gemini"
    assert plan(UNSURE)["source"] == "gemini (cached)" and plan.state["calls"] == 1
    assert json.loads(cache.read_text())["tell me about stuff"]["intent"] == "count"


def test_a_per_minute_429_is_waited_out_then_retried_once(fake_gemini, tmp_path):
    Fake.fail_first = True
    plan = run_eval.ai_planner("hybrid", 5, tmp_path / "cache.json")
    result = plan(UNSURE)
    assert result["intent"] == "count"                                   # the retry worked
    assert run_eval.RATE_LIMIT_WAIT in fake_gemini and plan.state["api_problems"] == 0


def test_call_limit_and_action_intents(fake_gemini, tmp_path):
    plan = run_eval.ai_planner("llm_first", 1, tmp_path / "cache.json")
    assert plan("delete review 5")["source"] == "rules"                  # actions never go to the AI, as in the app
    assert plan(UNSURE)["source"] == "gemini"
    assert plan("another odd question")["source"] == "none" and plan.state["skipped"] == 1


def test_cache_is_ignored_after_the_prompt_changes(fake_gemini, tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    run_eval.ai_planner("hybrid", 5, cache)(UNSURE)
    monkeypatch.setattr("services.planner.SYSTEM_PROMPT", "a different prompt")
    again = run_eval.ai_planner("hybrid", 5, cache)
    assert again(UNSURE)["source"] == "gemini"                           # asked again, not replayed from the old cache


def test_list_and_tuple_periods_compare_equal():
    assert run_eval.check({"period": ["2026-10-01", "2026-10-21"]}, {"period": pc.THIS_MONTH}) == []
