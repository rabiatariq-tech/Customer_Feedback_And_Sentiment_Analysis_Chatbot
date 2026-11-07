"""Quota guard (per-minute cap, rate-limit pause, daily block) and the provider's reading of Google's error body."""
import io
import json
import urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from services.llm.provider import _http_error_text
from services.llm.quota import QuotaGuard, is_daily_quota_error

LA = ZoneInfo("America/Los_Angeles")


def guard(**kw):
    now = [datetime(2026, 10, 21, 12, tzinfo=LA)]
    return QuotaGuard(daily_limit=kw.pop("daily", 100), session_limit=kw.pop("session", 100), clock=lambda: now[0], **kw), now


def http_error(code, body):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError("https://example.test", code, "x", {}, io.BytesIO(raw))


# ---- provider: Google's explanation must reach the guard ---------------------------------------------
def test_http_error_text_includes_googles_message_and_quota_id():
    body = {"error": {"message": "You exceeded your current quota.", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}
    text = _http_error_text(http_error(429, body))
    assert text.startswith("HTTP 429: You exceeded") and "PerMinute" in text
    assert not is_daily_quota_error(text)
    body["error"]["details"][0]["violations"][0]["quotaId"] = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert is_daily_quota_error(_http_error_text(http_error(429, body)))


def test_http_error_text_survives_unreadable_bodies_and_never_shows_a_key():
    assert _http_error_text(http_error(503, b"<html>not json</html>")) == "HTTP 503"
    leaked = {"error": {"message": "bad key AIzaSyA1234567890abcdefghijklmnopqrstuv"}}
    assert "AIza" not in _http_error_text(http_error(400, leaked))
    assert len(_http_error_text(http_error(500, {"error": {"message": "x" * 5000}}))) < 320


# ---- quota guard ------------------------------------------------------------------------------------
def test_per_minute_cap_frees_up_after_a_minute():
    g, now = guard(minute_limit=2)
    assert g.try_acquire("a")[0] and g.try_acquire("b")[0]
    assert g.try_acquire("c") == (False, "minute_limit")
    now[0] += timedelta(seconds=61)
    assert g.try_acquire("c")[0]


def test_refund_also_frees_a_minute_slot():
    g, _ = guard(minute_limit=1)
    assert g.try_acquire("a")[0]
    g.refund("a")
    assert g.try_acquire("a")[0]


def test_per_minute_429_pauses_only_for_about_a_minute():
    g, now = guard()
    g.provider_said_quota_gone("HTTP 429: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    assert g.try_acquire("a") == (False, "provider_rate_limit")
    now[0] += timedelta(seconds=66)
    assert g.try_acquire("a")[0]            # not blocked for the day


def test_daily_429_blocks_until_the_next_pacific_day():
    g, now = guard()
    g.provider_said_quota_gone("HTTP 429: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    now[0] += timedelta(hours=2)
    assert g.try_acquire("a") == (False, "daily_provider_quota")
    now[0] = datetime(2026, 10, 22, 0, 5, tzinfo=LA)
    assert g.try_acquire("a")[0]


def test_three_unexplained_429s_in_a_row_block_the_day_but_a_success_resets_the_count():
    g, now = guard()
    for _ in range(2):
        g.provider_said_quota_gone("HTTP 429")
        now[0] += timedelta(seconds=70)
    g.provider_call_succeeded()
    g.provider_said_quota_gone("HTTP 429")
    now[0] += timedelta(seconds=70)
    assert g.try_acquire("a")[0]            # only one 429 since the last success
    for _ in range(3):
        g.provider_said_quota_gone("HTTP 429")
    assert g.try_acquire("a") == (False, "daily_provider_quota")


def test_clear_provider_pause_lets_a_script_retry_after_waiting():
    g, _ = guard()
    g.provider_said_quota_gone("HTTP 429")
    g.clear_provider_pause()
    assert g.try_acquire("a")[0]
