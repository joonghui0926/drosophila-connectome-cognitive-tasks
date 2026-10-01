import numpy as np

from flystudy.metrics import addition_stratum, classification_metrics, constant_answer_baseline, sample_mix


def test_multiplication_metrics_do_not_hide_zero_and_one_collapse():
    rows = [
        {"a": 8, "b": 7, "op": "*", "y": 56},
        {"a": 0, "b": 88, "op": "*", "y": 0},
        {"a": 1, "b": 61, "op": "*", "y": 61},
        {"a": 10, "b": 5, "op": "+", "y": 15},
    ]
    metrics = classification_metrics(rows, np.asarray([0, 0, 61, 15]))
    assert metrics["nontrivial"]["accuracy"] == 0.0
    assert metrics["times_zero"]["accuracy"] == 1.0
    assert metrics["times_one"]["accuracy"] == 1.0


def test_sample_mix_reports_multiplication_fractions():
    rows = (
        [{"a": 2, "b": 3, "op": "*", "y": 6}] * 6
        + [{"a": 0, "b": 4, "op": "*", "y": 0}] * 2
        + [{"a": 1, "b": 4, "op": "*", "y": 4}] * 2
    )
    mix = sample_mix(rows)
    assert mix["multiplicationFractions"] == {
        "nontrivial": 0.6,
        "times_zero": 0.2,
        "times_one": 0.2,
    }


def test_addition_metrics_separate_carry_and_100_boundary():
    rows = [
        {"a": 12, "b": 3, "op": "+", "y": 15},
        {"a": 18, "b": 7, "op": "+", "y": 25},
        {"a": 48, "b": 52, "op": "+", "y": 100},
    ]
    assert [addition_stratum(row) for row in rows] == ["no_ones_carry", "ones_carry", "sum_100"]
    metrics = classification_metrics(rows, np.asarray([15, 24, 100]))
    assert metrics["no_ones_carry"]["accuracy"] == 1.0
    assert metrics["ones_carry"]["accuracy"] == 0.0
    assert metrics["sum_100"]["accuracy"] == 1.0


def test_constant_answer_baseline_uses_most_frequent_target():
    rows = [{"y": value} for value in [4, 7, 7, 9]]
    baseline = constant_answer_baseline(rows)
    assert baseline["answer"] == 7
    assert baseline["accuracy"] == 0.5
    fixed = constant_answer_baseline(rows, answer=4)
    assert fixed["answer"] == 4
    assert fixed["accuracy"] == 0.25
