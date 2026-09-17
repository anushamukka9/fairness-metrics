"""Tests for fairness_metrics.metrics."""

import math

import pytest

from fairness_metrics import (
    calibration_by_group,
    demographic_parity,
    disparate_impact,
    equal_opportunity,
    equalized_odds,
    fairness_report,
    overall_metrics,
    per_group_metrics,
)


def parity_data():
    # Two groups, identical outcome mix -> perfect parity everywhere.
    y_true = [0, 1, 1, 0, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 1, 1, 0]
    groups = ["a", "b", "a", "b", "a", "b", "a", "b"]
    return y_true, y_pred, groups


def biased_data():
    # Group 'x' selected at 100%, group 'y' at 0% -> max disparity.
    y_true = [1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = [1, 1, 1, 1, 0, 0, 0, 0]
    groups = ["x", "x", "x", "x", "y", "y", "y", "y"]
    return y_true, y_pred, groups


def test_per_group_metrics_counts():
    y_true, y_pred, groups = parity_data()
    gm = per_group_metrics(y_true, y_pred, groups)
    assert set(gm) == {"a", "b"}
    assert gm["a"].n == 4
    assert gm["a"].positives == 2
    assert gm["a"].negatives == 2
    assert gm["a"].selection_rate == pytest.approx(0.5)
    assert gm["a"].accuracy == pytest.approx(1.0)


def test_demographic_parity_zero_on_equal_data():
    y_true, y_pred, groups = parity_data()
    dp = demographic_parity(per_group_metrics(y_true, y_pred, groups))
    assert dp["max_difference"] == pytest.approx(0.0)
    assert dp["min_ratio"] == pytest.approx(1.0)


def test_demographic_parity_detects_bias():
    y_true, y_pred, groups = biased_data()
    dp = demographic_parity(per_group_metrics(y_true, y_pred, groups))
    assert dp["max_difference"] == pytest.approx(1.0)
    assert dp["min_ratio"] == pytest.approx(0.0)


def test_disparate_impact_four_fifths_rule():
    y_true, y_pred, groups = biased_data()
    di = disparate_impact(per_group_metrics(y_true, y_pred, groups))
    assert di["min_ratio"] == pytest.approx(0.0)
    assert di["four_fifths_pass"] is False

    y_true2, y_pred2, groups2 = parity_data()
    di2 = disparate_impact(per_group_metrics(y_true2, y_pred2, groups2))
    assert di2["four_fifths_pass"] is True


def test_equalized_odds_and_opportunity():
    y_true, y_pred, groups = biased_data()
    grp = per_group_metrics(y_true, y_pred, groups)
    eo = equalized_odds(grp)
    assert eo["tpr_max_difference"] == pytest.approx(1.0)
    eopp = equal_opportunity(grp)
    assert eopp["tpr_max_difference"] == pytest.approx(1.0)


def test_overall_metrics():
    y_true, y_pred, _ = parity_data()
    om = overall_metrics(y_true, y_pred)
    assert om.n == 8
    assert om.accuracy == pytest.approx(1.0)
    assert om.balanced_accuracy == pytest.approx(1.0)


def test_calibration_by_group_well_calibrated():
    rng = __import__("numpy").random.default_rng(7)
    n = 2000
    scores = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < scores).astype(int).tolist()
    groups = ["g1"] * (n // 2) + ["g2"] * (n - n // 2)
    cal = calibration_by_group(y_true, scores.tolist(), groups, n_bins=10)
    for g in ("g1", "g2"):
        assert cal[g]["expected_calibration_error"] < 0.05
        assert cal[g]["max_bin_gap"] < 0.15


def test_fairness_report_bundle():
    y_true, y_pred, groups = biased_data()
    report = fairness_report(y_true, y_pred, groups, scores=[0.9] * 8)
    assert report["n_samples"] == 8
    assert report["n_groups"] == 2
    for key in (
        "demographic_parity",
        "equalized_odds",
        "equal_opportunity",
        "disparate_impact",
        "calibration_by_group",
    ):
        assert key in report
    assert "x" in report["groups"] and "y" in report["groups"]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        per_group_metrics([0, 1], [0, 1, 0], ["a", "b", "c"])
    with pytest.raises(ValueError):
        per_group_metrics([0, 2], [0, 1], ["a", "b"])
    with pytest.raises(ValueError):
        calibration_by_group([0, 1], [0.5, 1.5], ["a", "b"])
    with pytest.raises(ValueError):
        per_group_metrics([], [], [])
