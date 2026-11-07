import pytest

from services import sentiment
from tests.fakes import (
    ExplodingPipeline,
    FakePipeline,
    UnloadableAnalyzer,
    WordTokenizer,
    WrongLengthPipeline,
    make_analyzer,
)


# --- parse_pipeline_output -------------------------------------------------

def test_parse_flat_list_of_scores():
    raw = [{"label": "negative", "score": 0.1}, {"label": "neutral", "score": 0.2}, {"label": "positive", "score": 0.7}]
    label, scores, error = sentiment.parse_pipeline_output(raw)
    assert (label, error) == ("positive", None)
    assert scores == {"negative": 0.1, "neutral": 0.2, "positive": 0.7}


def test_parse_nested_list_and_single_dict():
    nested = [[{"label": "Negative", "score": 0.6}, {"label": "Positive", "score": 0.4}]]
    label, scores, _ = sentiment.parse_pipeline_output(nested)
    assert label == "negative"
    assert scores["neutral"] == 0.0  # missing classes default to zero
    assert sentiment.parse_pipeline_output({"label": "neutral", "score": 0.9})[0] == "neutral"


def test_parse_generic_label_names():
    raw = [{"label": "LABEL_0", "score": 0.1}, {"label": "LABEL_1", "score": 0.1}, {"label": "LABEL_2", "score": 0.8}]
    assert sentiment.parse_pipeline_output(raw)[0] == "positive"


def test_parse_rejects_unknown_labels_and_empty_output():
    assert sentiment.parse_pipeline_output([{"label": "joy", "score": 1.0}])[2] is not None
    assert sentiment.parse_pipeline_output([])[2] is not None
    assert sentiment.parse_pipeline_output([{"score": 1.0}])[2] is not None


# --- SentimentAnalyzer -----------------------------------------------------

def test_analyze_returns_structured_result():
    result = make_analyzer().analyze("I love this lamp")
    assert result.ok
    assert result.label == "positive"
    assert result.model == "fake/model"
    assert abs(sum(result.scores.values()) - 1.0) < 1e-9
    assert result.confidence == 0.90
    fields = result.to_record_fields()
    assert fields["sentiment_label"] == "positive"
    assert fields["score_negative"] == 0.03
    assert fields["model_version"] == "fake/model"


def test_all_three_classes_are_reachable():
    labels = [r.label for r in make_analyzer().analyze_many(["great", "terrible", "it arrived"])]
    assert labels == ["positive", "negative", "neutral"]


def test_empty_and_blank_texts_get_errors_and_order_is_preserved():
    pipe = FakePipeline()
    results = make_analyzer(pipe).analyze_many(["love it", "", "   ", "awful"])
    assert [r.label for r in results] == ["positive", None, None, "negative"]
    assert results[1].error == "empty text"
    assert pipe.calls == [["love it", "awful"]]  # blank texts never reach the model


def test_all_blank_input_does_not_load_the_model():
    analyzer = UnloadableAnalyzer(model_name="fake/model")
    results = analyzer.analyze_many(["", " "])
    assert all(r.error == "empty text" for r in results)


def test_unparseable_model_output_is_reported_per_item():
    results = make_analyzer().analyze_many(["weird text", "great"])
    assert not results[0].ok and "unrecognised" in results[0].error
    assert results[1].ok


def test_model_load_failure_is_reported_not_raised():
    result = UnloadableAnalyzer(model_name="fake/model").analyze("great")
    assert not result.ok
    assert result.error.startswith("Could not load model")


def test_pipeline_exception_is_reported_not_raised():
    results = make_analyzer(ExplodingPipeline()).analyze_many(["a", "b"])
    assert all("analysis failed" in r.error for r in results)


def test_unexpected_result_count_is_an_error():
    results = make_analyzer(WrongLengthPipeline()).analyze_many(["a"])
    assert "unexpected number" in results[0].error


def test_truncation_flag_uses_tokenizer_limit():
    analyzer = make_analyzer(FakePipeline(tokenizer=WordTokenizer()), max_tokens=10)
    short, long = analyzer.analyze_many(["great stuff", "great " * 50])
    assert short.truncated is False
    assert long.truncated is True and long.ok


def test_no_tokenizer_means_not_truncated():
    assert make_analyzer().analyze("great " * 500).truncated is False


def test_very_short_inputs_return_a_result():
    results = make_analyzer().analyze_many(["ok", "Good", "👍"])
    assert all(r.ok for r in results)
