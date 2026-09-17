"""Tests for fairness_metrics.thresholds and fairness_metrics.report."""

import json

import pytest

from fairness_metrics import (
    CONSTRAINT_MEASURES,
    demographic_parity,
    fairness_report,
    find_fair_thresholds,
    per_group_metrics,
    threshold_scan,
    to_json,
    to_markdown,
)


def scored_data():
    # Group 'x' gets systematically lower scores -> single threshold is unfair.
    rng = __import__("numpy").random.default_rng(42)
    n = 400
    scores_x = rng.beta(2, 5, n).tolist()
    scores_y = rng.beta(5, 2, n).tolist()
    scores = scores_x + scores_y
    y_true = [1 if s > 0.5 else 0 for s in scores]
    # Add label noise so it is not trivially separable.
    y_true = [1 - y if (i % 11 == 0) else y for i, y in enumerate(y_true)]
    groups = ["x"] * n + ["y"] * n
    return y_true, scores, groups


def test_threshold_scan_structure():
    y_true, scores, groups = scored_data()
    out = threshold_scan(y_true, scores, groups, thresholds=[0.3, 0.5, 0.7])
    assert out["thresholds"] == [0.3, 0.5, 0.7]
    assert len(out["scan"]) == 3
    row = out["scan"][0]
    assert "demographic_parity_max_difference" in row
    assert "disparate_impact_min_ratio" in row
    assert set(row["groups"]) == {"x", "y"}


def test_threshold_scan_rejects_bad_grid():
    y_true, scores, groups = scored_data()
    with pytest.raises(ValueError):
        threshold_scan(y_true, scores, groups, thresholds=[0.0, 1.5])


def test_find_fair_thresholds_reduces_gap():
    y_true, scores, groups = scored_data()
    baseline = threshold_scan(y_true, scores, groups, thresholds=[0.5])
    baseline_gap = baseline["scan"][0]["demographic_parity_max_difference"]
    assert baseline_gap > 0.1  # sanity: the single threshold really is unfair

    result = find_fair_thresholds(
        y_true, scores, groups, constraint="demographic_parity", max_gap=0.1
    )
    assert result["constraint_satisfied"] is True
    assert result["achieved_gap"] <= 0.1
    assert set(result["thresholds"]) == {"x", "y"}
    # Per-group thresholds differ because the score distributions differ.
    assert result["thresholds"]["x"] != result["thresholds"]["y"]


def test_find_fair_thresholds_validates_inputs():
    y_true, scores, groups = scored_data()
    with pytest.raises(ValueError):
        find_fair_thresholds(y_true, scores, groups, constraint="not_a_measure")
    with pytest.raises(ValueError):
        find_fair_thresholds(y_true, scores, groups, objective="not_an_objective")
    with pytest.raises(ValueError):
        find_fair_thresholds(y_true, scores, groups, max_gap=1.5)
    assert "demographic_parity" in CONSTRAINT_MEASURES


def test_report_json_roundtrip():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    groups = ["a", "a", "b", "b"]
    report = fairness_report(y_true, y_pred, groups)
    parsed = json.loads(to_json(report))
    assert parsed["n_groups"] == 2
    assert parsed["groups"]["a"]["n"] == 2


def test_report_markdown_contents():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    groups = ["a", "a", "b", "b"]
    report = fairness_report(y_true, y_pred, groups)
    md = to_markdown(report)
    assert "# Fairness Audit Report" in md
    assert "Demographic parity" in md
    assert "Disparate impact" in md
    assert "| a |" in md and "| b |" in md
    assert "Four-fifths rule" in md
