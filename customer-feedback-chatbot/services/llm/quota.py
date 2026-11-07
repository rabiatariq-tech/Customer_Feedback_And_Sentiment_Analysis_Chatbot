"""Quota guard for a free-tier key that every visitor shares.

Protections: a small cap per visitor session, a per-minute cap for the whole app (Google's free tier also limits
requests per minute), a daily cap for the whole app (set below the real quota so a burst never hits Google's
limit), and a cache so the same question never costs a second call. The daily counter resets at midnight US
Pacific time, like Google's free-tier quota.

When Google answers 429 the guard looks at *which* limit was hit. A per-minute limit only pauses the AI for about
a minute; a per-day limit (or three 429s in a row) blocks it until the next Pacific midnight.
"""
from __future__ import annotations

import threading
from collections import OrderedDict, deque
from datetime import datetime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
MINUTE_PAUSE_SECONDS = 65      # a little over one minute, so the per-minute window has really reset
BLOCK_AFTER_CONSECUTIVE_429 = 3


def is_daily_quota_error(error: str | None) -> bool:
    """True when Google's 429 message says the DAILY quota is used up (it names it "PerDay"/"per day"/"daily")."""
    text = (error or "").lower().replace(" ", "").replace("_", "").replace("-", "")
    return any(word in text for word in ("perday", "daily"))


class QuotaGuard:
    def __init__(self, daily_limit: int, session_limit: int, clock: Callable[[], datetime] | None = None, cache_size: int = 200,
                 minute_limit: int = 10):
        self.daily_limit, self.session_limit, self.minute_limit = daily_limit, session_limit, minute_limit
        self._clock = clock or (lambda: datetime.now(PACIFIC))
        self._lock = threading.Lock()
        self._day = self._clock().date()
        self._used_today = 0
        self._sessions: dict[str, int] = {}
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = cache_size
        self.blocked_until_tomorrow = False  # set when Google itself says the daily quota is gone
        self._recent: deque[datetime] = deque()   # start times of calls in the last minute
        self._paused_until: datetime | None = None  # short pause after a per-minute 429
        self._consecutive_429 = 0

    def _roll_day(self) -> None:
        today = self._clock().date()
        if today != self._day:
            self._day, self._used_today, self._sessions = today, 0, {}
            self.blocked_until_tomorrow = False
            self._paused_until, self._consecutive_429 = None, 0

    def try_acquire(self, session_id: str) -> tuple[bool, str | None]:
        """Reserve one call. Returns (allowed, reason_if_not)."""
        with self._lock:
            self._roll_day()
            now = self._clock()
            if self.blocked_until_tomorrow:
                return False, "daily_provider_quota"
            if self._paused_until is not None:
                if now < self._paused_until:
                    return False, "provider_rate_limit"
                self._paused_until = None
            if self._used_today >= self.daily_limit:
                return False, "daily_limit"
            if self._sessions.get(session_id, 0) >= self.session_limit:
                return False, "session_limit"
            while self._recent and now - self._recent[0] >= timedelta(seconds=60):
                self._recent.popleft()
            if len(self._recent) >= self.minute_limit:
                return False, "minute_limit"
            self._recent.append(now)
            self._used_today += 1
            self._sessions[session_id] = self._sessions.get(session_id, 0) + 1
            return True, None

    def refund(self, session_id: str) -> None:
        """Give a reservation back when no request was actually made."""
        with self._lock:
            self._used_today = max(0, self._used_today - 1)
            self._sessions[session_id] = max(0, self._sessions.get(session_id, 0) - 1)
            if self._recent:
                self._recent.pop()

    def provider_said_quota_gone(self, error: str | None = None) -> None:
        """Google answered 429. With no message (old behaviour) assume the daily quota is gone. With a message,
        block for the day only if it says so (or after repeated 429s); otherwise pause for about a minute."""
        with self._lock:
            self._consecutive_429 += 1
            if error is None or is_daily_quota_error(error) or self._consecutive_429 >= BLOCK_AFTER_CONSECUTIVE_429:
                self.blocked_until_tomorrow = True
            else:
                self._paused_until = self._clock() + timedelta(seconds=MINUTE_PAUSE_SECONDS)

    def provider_call_succeeded(self) -> None:
        """A call went through: the earlier 429s were a passing rate limit, not a used-up quota."""
        with self._lock:
            self._consecutive_429 = 0

    def clear_provider_pause(self) -> None:
        """Forget a short rate-limit pause (the evaluation script waits it out itself, then retries)."""
        with self._lock:
            self._paused_until = None

    def cache_get(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def cache_put(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def status(self) -> dict[str, int]:
        with self._lock:
            self._roll_day()
            return {"used_today": self._used_today, "daily_limit": self.daily_limit}
