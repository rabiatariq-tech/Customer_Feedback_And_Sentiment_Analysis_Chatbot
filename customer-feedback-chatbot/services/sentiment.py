"""Sentiment analysis with a pretrained Hugging Face three-class model.

The model is loaded lazily (so importing this module never needs torch) and only
once per analyzer. Every public method returns a ``SentimentResult`` and never
raises for bad input or a missing model: problems are reported in ``error``.

Scores are the model's softmax outputs. Treat them as confidence-like numbers,
not guaranteed probabilities that the label is correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import config

LABELS = ("positive", "neutral", "negative")

# Label names used by common three-class models, mapped to our canonical labels.
_LABEL_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "label_2": "positive",
    "neutral": "neutral",
    "neu": "neutral",
    "label_1": "neutral",
    "negative": "negative",
    "neg": "negative",
    "label_0": "negative",
}


@dataclass
class SentimentResult:
    label: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    model: str = ""
    truncated: bool = False  # True when the text exceeded the model's input limit
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.label is not None

    @property
    def confidence(self) -> float | None:
        return self.scores.get(self.label) if self.label else None

    @property
    def flags(self) -> list[str]:
        """Quality flags: 'low_confidence' (top score below the threshold) and 'truncated'."""
        found = []
        if self.confidence is not None and self.confidence < config.SENTIMENT_LOW_CONFIDENCE:
            found.append("low_confidence")
        if self.truncated:
            found.append("truncated")
        return found

    def to_record_fields(self) -> dict[str, Any]:
        """Fields to store in the feedback table (only meaningful when ``ok``)."""
        return {
            "sentiment_label": self.label,
            "score_positive": self.scores.get("positive"),
            "score_neutral": self.scores.get("neutral"),
            "score_negative": self.scores.get("negative"),
            "model_version": self.model,
            "flags": ",".join(self.flags) or None,
        }


def parse_pipeline_output(raw: Any) -> tuple[str | None, dict[str, float], str | None]:
    """Turn one item of text-classification output into (label, scores, error).

    Accepts a list of {label, score} dicts (``top_k=None``), a nested list with one
    such list inside, or a single {label, score} dict.
    """
    if isinstance(raw, dict):
        raw = [raw]
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    if not raw:
        return None, {}, "the model returned no scores"

    scores: dict[str, float] = {}
    for item in raw:
        try:
            name = _LABEL_ALIASES[str(item["label"]).strip().lower()]
            scores[name] = float(item["score"])
        except (KeyError, TypeError, ValueError):
            return None, {}, f"unrecognised model output: {item!r}"

    if not scores:
        return None, {}, "the model returned no usable scores"
    for name in LABELS:
        scores.setdefault(name, 0.0)
    label = max(LABELS, key=lambda name: scores[name])
    return label, scores, None


class SentimentAnalyzer:
    """Wraps a text-classification pipeline. Pass ``pipeline`` to inject a fake in tests."""

    def __init__(
        self,
        model_name: str | None = None,
        max_tokens: int | None = None,
        pipeline: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name or config.SENTIMENT_MODEL
        self.max_tokens = max_tokens or config.SENTIMENT_MAX_TOKENS
        self._pipe = pipeline
        self.load_error: str | None = None
        self.quantized = False
        self.quantize_note: str | None = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def ensure_loaded(self) -> tuple[bool, str | None]:
        """Load the model if needed. Returns (ok, error_message). Retries after a failure."""
        if self._pipe is not None:
            return True, None
        try:
            from transformers import pipeline  # imported lazily: heavy, needs torch

            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=self.max_tokens,
            )
            self.load_error = None
            self._maybe_quantize()
            return True, None
        except Exception as exc:  # noqa: BLE001 - report any load problem to the UI
            self._pipe = None
            self.load_error = f"Could not load model '{self.model_name}': {type(exc).__name__}: {exc}"
            return False, self.load_error

    def _maybe_quantize(self) -> None:
        """Experimental: 8-bit dynamic quantisation of the Linear layers (CPU). Never raises.

        If it fails, the normal full-precision model keeps working and ``quantize_note`` says why.
        """
        if not config.SENTIMENT_QUANTIZE or self._pipe is None:
            return
        try:
            import torch

            self._pipe.model = torch.ao.quantization.quantize_dynamic(
                self._pipe.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.quantized = True
            self.quantize_note = "8-bit dynamic quantisation is on (experimental)."
        except Exception as exc:  # noqa: BLE001 - keep the working model
            self.quantized = False
            self.quantize_note = f"Quantisation was requested but failed, using the normal model: {type(exc).__name__}: {exc}"

    def _is_truncated(self, text: str) -> bool:
        tokenizer = getattr(self._pipe, "tokenizer", None)
        if tokenizer is None:
            return False
        try:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"]) > self.max_tokens
        except Exception:  # noqa: BLE001 - truncation info is best-effort
            return False

    def analyze(self, text: str) -> SentimentResult:
        return self.analyze_many([text])[0]

    def analyze_many(self, texts: Sequence[str], batch_size: int | None = None) -> list[SentimentResult]:
        """Analyze texts; the output list always matches the input order and length."""
        results = [SentimentResult(model=self.model_name) for _ in texts]
        valid = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        for i in range(len(texts)):
            if i not in valid:
                results[i].error = "empty text"
        if not valid:
            return results

        ok, error = self.ensure_loaded()
        if not ok:
            for i in valid:
                results[i].error = error
            return results

        try:
            raw = self._pipe([texts[i] for i in valid], batch_size=batch_size or config.SENTIMENT_BATCH_SIZE)
        except Exception as exc:  # noqa: BLE001 - e.g. out-of-memory on a huge batch
            for i in valid:
                results[i].error = f"analysis failed: {type(exc).__name__}: {exc}"
            return results

        if len(raw) != len(valid):
            for i in valid:
                results[i].error = "the model returned an unexpected number of results"
            return results

        for i, item in zip(valid, raw):
            label, scores, parse_error = parse_pipeline_output(item)
            results[i].label, results[i].scores, results[i].error = label, scores, parse_error
            if parse_error is None:
                results[i].truncated = self._is_truncated(texts[i])
        return results
