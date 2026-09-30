"""Tests for predictive parity and the plain-text report renderer."""

import math

import pytest

from fairness_metrics import (
    CONSTRAINT_MEASURES,
    fairness_report,
    find_fair_thresholds,
    per_group_metrics,
    predictive_parity,
    threshold_scan,
    to_markdown,
    to_text,
)


def ppv_data():
    # Group 'a': precision 1.0 (2/2 predicted positive are true).
    # Group 'b': precision 0.5 (1/2 predicted positive are true).
    y_true = [1, 1, 0, 0, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 1, 0, 0]
    groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    return y_true, y_pred, groups


def test_predictive_parity_values():
    gm = per_group_metrics(*ppv_data())
    pp = predictive_parity(gm)
    assert pp["precision_by_group"]["a"] == pytest.approx(1.0)
    assert pp["precision_by_group"]["b"] == pytest.approx(0.5)
    assert pp["max_difference"] == pytest.approx(0.5)
    assert pp["min_ratio"] == pytest.approx(0.5)


def test_predictive_parity_zero_on_equal_data():
    y_true = [0, 1, 1, 0, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 1, 1, 0]
    groups = ["a", "b", "a", "b", "a", "b", "a", "b"]
    pp = predictive_parity(per_group_metrics(y_true, y_pred, groups))
    assert pp["max_difference"] == pytest.approx(0.0)
    assert pp["min_ratio"] == pytest.approx(1.0)


def test_predictive_parity_ignores_nan_groups():
    # Group 'b' is never predicted positive -> precision NaN; the other
    # group's value still compares against itself, giving zero difference.
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    groups = ["a", "a", "b", "b"]
    pp = predictive_parity(per_group_metrics(y_true, y_pred, groups))
    assert math.isnan(pp["precision_by_group"]["b"])
    assert pp["max_difference"] == pytest.approx(0.0)


def test_fairness_report_includes_predictive_parity():
    report = fairness_report(*ppv_data())
    assert "predictive_parity" in report
    assert report["predictive_parity"]["max_difference"] == pytest.approx(0.5)


def test_predictive_parity_gap_is_a_scan_constraint():
    assert "predictive_parity_gap" in CONSTRAINT_MEASURES


def test_find_fair_thresholds_with_predictive_parity_constraint():
    y_true = [0, 1, 1, 0, 0, 1, 1, 0] * 5
    scores = [0.1, 0.9, 0.8, 0.2, 0.6, 0.7, 0.85, 0.3] * 5
    groups = ["a", "b"] * 20
    result = find_fair_thresholds(
        y_true, scores, groups,
        constraint="predictive_parity_gap", max_gap=0.5,
    )
    assert set(result["thresholds"]) == {"a", "b"}
    assert result["achieved_gap"] <= 0.5


def test_threshold_scan_includes_predictive_parity_row_key():
    y_true = [0, 1, 1, 0, 0, 1, 1, 0]
    scores = [0.1, 0.9, 0.8, 0.2, 0.6, 0.7, 0.85, 0.3]
    groups = ["a", "b", "a", "b", "a", "b", "a", "b"]
    scan = threshold_scan(y_true, scores, groups, thresholds=[0.5])
    row = scan["scan"][0]
    assert "predictive_parity_max_difference" in row
    assert 0.0 <= row["predictive_parity_max_difference"] <= 1.0


def test_to_text_renders_report():
    report = fairness_report(*ppv_data(), scores=[0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.2, 0.1])
    text = to_text(report)
    for section in (
        "Overall performance",
        "Per-group breakdown",
        "Demographic parity",
        "Equalized odds",
        "Equal opportunity",
        "Predictive parity",
        "Disparate impact",
        "Calibration by group",
    ):
        assert section in text


def test_to_text_is_ascii_only():
    report = fairness_report(*ppv_data())
    text = to_text(report)
    text.encode("ascii")  # raises if any non-ASCII slipped in


def test_to_text_shows_group_rows():
    report = fairness_report(*ppv_data())
    text = to_text(report)
    assert "a" in text and "b" in text
    assert "0.5000" in text  # group b precision


def test_to_markdown_includes_predictive_parity_section():
    report = fairness_report(*ppv_data())
    markdown = to_markdown(report)
    assert "## Predictive parity" in markdown
