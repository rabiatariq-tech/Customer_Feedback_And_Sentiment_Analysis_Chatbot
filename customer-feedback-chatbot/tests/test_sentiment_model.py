"""Optional tests that run the REAL model. Skipped unless RUN_MODEL_TESTS=1.

    PowerShell:  $env:RUN_MODEL_TESTS = "1"; pytest tests/test_sentiment_model.py -v

The first run downloads the model (about 500 MB), so these are excluded from the normal test run.
"""
import os

import pytest

if os.getenv("RUN_MODEL_TESTS") != "1":
    pytest.skip("set RUN_MODEL_TESTS=1 to run tests against the real model", allow_module_level=True)

from services.sentiment import SentimentAnalyzer  # noqa: E402


@pytest.fixture(scope="module")
def analyzer():
    instance = SentimentAnalyzer()
    ok, error = instance.ensure_loaded()
    assert ok, error
    return instance


def test_clear_positive_and_negative(analyzer):
    positive, negative = analyzer.analyze_many(
        ["I absolutely love this, it works perfectly!", "Terrible. It broke after two days and I want a refund."]
    )
    assert positive.label == "positive"
    assert negative.label == "negative"
    assert abs(sum(positive.scores.values()) - 1.0) < 1e-3


def test_neutral_statement(analyzer):
    assert analyzer.analyze("The package arrived on Tuesday.").label == "neutral"


def test_very_long_text_is_truncated_not_rejected(analyzer):
    result = analyzer.analyze("The delivery was slow and the box was damaged. " * 200)
    assert result.ok and result.truncated


def test_short_and_empty_inputs(analyzer):
    short, empty = analyzer.analyze_many(["ok", ""])
    assert short.ok
    assert empty.error == "empty text"
