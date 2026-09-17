"""fairness-metrics library: group fairness diagnostics for binary classifiers."""

from .metrics import (
    GroupMetrics,
    calibration_by_group,
    demographic_parity,
    disparate_impact,
    equal_opportunity,
    equalized_odds,
    fairness_report,
    overall_metrics,
    per_group_metrics,
)
from .report import to_json, to_markdown
from .thresholds import CONSTRAINT_MEASURES, find_fair_thresholds, threshold_scan

__all__ = [
    "GroupMetrics",
    "per_group_metrics",
    "overall_metrics",
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "disparate_impact",
    "calibration_by_group",
    "fairness_report",
    "threshold_scan",
    "find_fair_thresholds",
    "CONSTRAINT_MEASURES",
    "to_json",
    "to_markdown",
]

__version__ = "0.1.0"
__author__ = "Anusha Mukka"
__homepage__ = "https://anushamukka.com"
