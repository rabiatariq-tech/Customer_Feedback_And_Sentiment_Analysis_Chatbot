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
            return LLMResult(False, error=type(exc).__name__)
        except ValueError:
            return LLMResult(False, error="bad response")
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return LLMResult(True, data=json.loads(text))
        except (KeyError, IndexError, TypeError, ValueError):
            return LLMResult(False, error="unexpected response shape")
