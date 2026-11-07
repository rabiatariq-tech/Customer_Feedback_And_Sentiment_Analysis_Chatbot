"""Question-understanding evaluation as a regression test (see evaluation/run_eval.py for the report)."""
from evaluation import heldout_cases, planner_cases
from evaluation.run_eval import evaluate

# Rule-based accuracy on the held-out set when this test was written. An LLM planner has to beat it; the rules must not
# get worse. Raise it when a change genuinely improves the held-out score.
HELDOUT_BASELINE = 0.80


def test_main_evaluation_set_passes_completely():
    failures, _ = evaluate(cases=planner_cases.CASES)
    assert not failures, [f[0] for f in failures]


def test_heldout_accuracy_does_not_regress():
    failures, _ = evaluate(cases=heldout_cases.CASES)
    accuracy = 1 - len(failures) / len(heldout_cases.CASES)
    assert accuracy >= HELDOUT_BASELINE, f"held-out accuracy fell to {accuracy:.0%}"
