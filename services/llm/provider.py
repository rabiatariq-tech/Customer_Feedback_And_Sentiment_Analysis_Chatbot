"""Provider adapter. The rest of the app only sees `LLMProvider.generate_json`, so a different provider
(or a local model) can be swapped in by writing one small class."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class LLMResult:
    ok: bool
    data: Any = None          # parsed JSON when ok
    error: str | None = None  # short, safe message (never includes the API key)
    quota_exhausted: bool = False
    unreachable: bool = False  # timeout or no network: the request never got an answer from Google


class LLMProvider(Protocol):
    name: str

    def generate_json(self, system: str, user: str, max_output_tokens: int = 400) -> LLMResult: ...


_KEY_LIKE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def _http_error_text(exc: urllib.error.HTTPError) -> str:
    """"HTTP 429: <Google's explanation>". Google's body says *which* limit was hit (per minute or per day), and
    the quota guard needs that to decide how long to pause. The text is shortened and anything that looks like an
    API key is removed before it can reach a log or the screen."""
    detail = ""
    try:
        body = json.loads(exc.read().decode("utf-8", errors="replace"))
        error = body.get("error", {}) if isinstance(body, dict) else {}
        parts = [str(error.get("message") or "")]
        for item in error.get("details") or []:
            for violation in (item.get("violations") or []) if isinstance(item, dict) else []:
                if isinstance(violation, dict) and violation.get("quotaId"):
                    parts.append(str(violation["quotaId"]))
        detail = " ".join(" ".join(p.split()) for p in parts if p)
    except (ValueError, OSError, AttributeError, TypeError):
        pass  # no readable body: keep just the status code
    detail = _KEY_LIKE.sub("[key removed]", detail)[:300]
    return f"HTTP {exc.code}: {detail}" if detail else f"HTTP {exc.code}"


class GeminiProvider:
    """Google Gemini through its REST API (no extra package needed)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float = 15.0):
        self._key, self._model, self._timeout = api_key, model, timeout

    def generate_json(self, system: str, user: str, max_output_tokens: int = 400) -> LLMResult:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        request = urllib.request.Request(
            GEMINI_URL.format(model=self._model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310 - fixed https URL
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return LLMResult(False, error=_http_error_text(exc), quota_exhausted=exc.code == 429)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return LLMResult(False, error=type(exc).__name__, unreachable=True)
        except ValueError:
            return LLMResult(False, error="bad response")
        return _parse_payload(payload)


def _parse_payload(payload: Any) -> LLMResult:
    """The JSON answer inside Google's reply, or a short reason why there is none. The reasons matter: "the answer
    was cut off" (raise the token limit) and "blocked" (reword) need different fixes than a broken reply."""
    try:
        candidate = payload["candidates"][0]
    except (KeyError, IndexError, TypeError):
        blocked = (payload.get("promptFeedback") or {}).get("blockReason") if isinstance(payload, dict) else None
        return LLMResult(False, error=f"no answer (blocked: {blocked})" if blocked else "unexpected response shape")
    finish = str(candidate.get("finishReason") or "") if isinstance(candidate, dict) else ""
    try:
        parts = candidate["content"]["parts"]
        # a model that "thinks" may send its reasoning as separate parts: only the answer text is wanted
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    except (KeyError, TypeError, AttributeError):
        text = ""
    if not text.strip():
        return LLMResult(False, error=f"empty answer (finish: {finish or 'unknown'})")
    try:
        return LLMResult(True, data=json.loads(text))
    except ValueError:
        cut = finish == "MAX_TOKENS"
        return LLMResult(False, error="answer was cut off (token limit)" if cut else f"answer was not valid JSON (finish: {finish or 'unknown'})")
