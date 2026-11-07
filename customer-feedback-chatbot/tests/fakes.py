"""Test doubles so sentiment logic can be tested without downloading a model."""
from __future__ import annotations

from services.sentiment import SentimentAnalyzer

_POSITIVE = {"positive": 0.90, "neutral": 0.07, "negative": 0.03}
_NEUTRAL = {"positive": 0.10, "neutral": 0.80, "negative": 0.10}
_NEGATIVE = {"positive": 0.02, "neutral": 0.08, "negative": 0.90}


def _as_output(scores: dict[str, float]) -> list[dict]:
    return [{"label": label, "score": score} for label, score in scores.items()]


class FakePipeline:
    """Keyword-based stand-in for a Hugging Face text-classification pipeline."""

    def __init__(self, tokenizer=None):
        self.calls: list[list[str]] = []
        self.tokenizer = tokenizer

    def __call__(self, texts, batch_size=None, **kwargs):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            low = text.lower()
            if "weird" in low:
                out.append([{"label": "LABEL_9", "score": 1.0}])  # unrecognised label
            elif any(w in low for w in ("love", "great", "fantastic")):
                out.append(_as_output(_POSITIVE))
            elif any(w in low for w in ("terrible", "broke", "awful")):
                out.append(_as_output(_NEGATIVE))
            else:
                out.append(_as_output(_NEUTRAL))
        return out


class ExplodingPipeline:
    def __call__(self, texts, **kwargs):
        raise RuntimeError("out of memory")


class WrongLengthPipeline:
    def __call__(self, texts, **kwargs):
        return []


class WordTokenizer:
    """Counts whitespace-separated words as tokens (plus two special tokens)."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split()) + 2))}


class UnloadableAnalyzer(SentimentAnalyzer):
    """Analyzer whose model can never be loaded (simulates no internet / missing torch)."""

    def ensure_loaded(self):
        self.load_error = f"Could not load model '{self.model_name}': OSError: no connection"
        return False, self.load_error


def make_analyzer(pipeline=None, model_name="fake/model", max_tokens=512) -> SentimentAnalyzer:
    return SentimentAnalyzer(model_name=model_name, max_tokens=max_tokens, pipeline=pipeline or FakePipeline())
