"""Core fairness metrics computed from predictions, labels, and a sensitive attribute.

All metrics operate on binary classification outcomes and support an arbitrary
number of protected groups. Nothing here requires more than numpy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence

import numpy as np

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
]


def _binary_array(values: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence")
    unique = np.unique(arr)
    if not set(unique.tolist()) <= {0, 1}:
        raise ValueError(f"{name} must be binary (only 0/1 values), got {unique.tolist()}")
    return arr.astype(int)


def _group_array(values: Sequence[Hashable], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence")
    return arr


def _check_aligned(*arrays: np.ndarray) -> None:
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"all inputs must have the same length, got {sorted(lengths)}")
    if next(iter(lengths)) == 0:
        raise ValueError("inputs must be non-empty")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


@dataclass(frozen=True)
class GroupMetrics:
    """Confusion-derived metrics for a single group."""

    group: Hashable
    n: int
    positives: int  # true positives count (y=1)
    negatives: int  # true negatives count (y=0)
    selection_rate: float  # P(yhat=1) — fraction predicted positive
    true_positive_rate: float  # TPR = recall = P(yhat=1 | y=1)
    false_positive_rate: float  # FPR = P(yhat=1 | y=0)
    true_negative_rate: float  # TNR = specificity = P(yhat=0 | y=0)
    precision: float  # P(y=1 | yhat=1)
    accuracy: float
    balanced_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["group"] = str(self.group)
        return d


def _confusion_for(y_true: np.ndarray, y_pred: np.ndarray) -> GroupMetrics:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = len(y_true)
    return GroupMetrics(
        group="overall",
        n=n,
        positives=tp + fn,
        negatives=tn + fp,
        selection_rate=_safe_div(tp + fp, n),
        true_positive_rate=_safe_div(tp, tp + fn),
        false_positive_rate=_safe_div(fp, fp + tn),
        true_negative_rate=_safe_div(tn, tn + fp),
        precision=_safe_div(tp, tp + fp),
        accuracy=_safe_div(tp + tn, n),
        balanced_accuracy=(
            _safe_div(tp, tp + fn) + _safe_div(tn, tn + fp)
        )
        / 2,
    )


def per_group_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Sequence[Hashable],
) -> Dict[str, GroupMetrics]:
    """Compute :class:`GroupMetrics` for every distinct group label.

    Returns a dict keyed by the string form of the group label, preserving
    sorted order of the labels for deterministic output.
    """
    yt = _binary_array(y_true, "y_true")
    yp = _binary_array(y_pred, "y_pred")
    gr = _group_array(groups, "groups")
    _check_aligned(yt, yp, gr)

    result: Dict[str, GroupMetrics] = {}
    for label in sorted(set(gr.tolist()), key=str):
        mask = gr == label
        gm = _confusion_for(yt[mask], yp[mask])
        result[str(label)] = GroupMetrics(
            group=label,
            n=gm.n,
            positives=gm.positives,
            negatives=gm.negatives,
            selection_rate=gm.selection_rate,
            true_positive_rate=gm.true_positive_rate,
            false_positive_rate=gm.false_positive_rate,
            true_negative_rate=gm.true_negative_rate,
            precision=gm.precision,
            accuracy=gm.accuracy,
            balanced_accuracy=gm.balanced_accuracy,
        )
    return result


def overall_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> GroupMetrics:
    """Aggregate confusion metrics over the whole sample."""
    yt = _binary_array(y_true, "y_true")
    yp = _binary_array(y_pred, "y_pred")
    _check_aligned(yt, yp)
    return _confusion_for(yt, yp)


def _rates_by_group(
    group_metrics: Mapping[str, GroupMetrics], field: str
) -> Dict[str, float]:
    return {g: getattr(gm, field) for g, gm in group_metrics.items()}


def _max_pairwise_difference(rates: Mapping[str, float]) -> float:
    vals = [v for v in rates.values() if not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    return float(max(vals) - min(vals))


def _min_pairwise_ratio(rates: Mapping[str, float]) -> float:
    vals = [v for v in rates.values() if not np.isnan(v)]
    if len(vals) < 2:
        return 1.0
    lo, hi = min(vals), max(vals)
    if hi == 0:
        return 1.0 if lo == 0 else 0.0
    return float(lo / hi)


def demographic_parity(group_metrics: Mapping[str, GroupMetrics]) -> Dict[str, Any]:
    """Demographic (statistical) parity: equal P(yhat=1) across groups.

    Reports both the maximum pairwise selection-rate difference and the
    minimum selection-rate ratio between any two groups.
    """
    rates = _rates_by_group(group_metrics, "selection_rate")
    return {
        "selection_rates": rates,
        "max_difference": _max_pairwise_difference(rates),
        "min_ratio": _min_pairwise_ratio(rates),
    }


def equalized_odds(group_metrics: Mapping[str, GroupMetrics]) -> Dict[str, Any]:
    """Equalized odds: equal TPR *and* equal FPR across groups.

    Reports the maximum pairwise differences of both rates.
    """
    tpr = _rates_by_group(group_metrics, "true_positive_rate")
    fpr = _rates_by_group(group_metrics, "false_positive_rate")
    return {
        "tpr_by_group": tpr,
        "fpr_by_group": fpr,
        "tpr_max_difference": _max_pairwise_difference(tpr),
        "fpr_max_difference": _max_pairwise_difference(fpr),
        "max_difference": max(_max_pairwise_difference(tpr), _max_pairwise_difference(fpr)),
    }


def equal_opportunity(group_metrics: Mapping[str, GroupMetrics]) -> Dict[str, Any]:
    """Equal opportunity: equal TPR (recall on the positive class) across groups."""
    tpr = _rates_by_group(group_metrics, "true_positive_rate")
    return {
        "tpr_by_group": tpr,
        "tpr_max_difference": _max_pairwise_difference(tpr),
    }


def disparate_impact(
    group_metrics: Mapping[str, GroupMetrics], threshold: float = 0.8
) -> Dict[str, Any]:
    """Disparate impact: minimum selection-rate ratio between any two groups.

    The classic "four-fifths rule" flags ratios below ``threshold`` (default
    0.8) as evidence of adverse impact.
    """
    rates = _rates_by_group(group_metrics, "selection_rate")
    ratio = _min_pairwise_ratio(rates)
    return {
        "selection_rates": rates,
        "min_ratio": ratio,
        "four_fifths_threshold": threshold,
        "four_fifths_pass": bool(ratio >= threshold),
    }


def calibration_by_group(
    y_true: Sequence[Any],
    scores: Sequence[float],
    groups: Sequence[Hashable],
    n_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Calibration diagnostics per group from predicted probabilities.

    For each group, bins scores into ``n_bins`` equal-width bins and reports
    the expected calibration error (ECE), per-bin observed positive rates,
    and the maximum bin-level absolute gap between predicted and observed.
    """
    yt = _binary_array(y_true, "y_true")
    sc = np.asarray(scores, dtype=float)
    gr = _group_array(groups, "groups")
    _check_aligned(yt, sc, gr)
    if sc.ndim != 1 or np.any((sc < 0) | (sc > 1)):
        raise ValueError("scores must be a 1-D sequence of probabilities in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result: Dict[str, Dict[str, Any]] = {}
    for label in sorted(set(gr.tolist()), key=str):
        mask = gr == label
        y_g, s_g = yt[mask], sc[mask]
        bins = []
        ece = 0.0
        max_gap = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            in_bin = (s_g > lo) & (s_g <= hi) if i > 0 else (s_g >= lo) & (s_g <= hi)
            count = int(np.sum(in_bin))
            if count == 0:
                continue
            mean_pred = float(np.mean(s_g[in_bin]))
            mean_obs = float(np.mean(y_g[in_bin]))
            gap = abs(mean_pred - mean_obs)
            ece += (count / len(y_g)) * gap
            max_gap = max(max_gap, gap)
            bins.append(
                {
                    "bin": [float(lo), float(hi)],
                    "count": count,
                    "mean_predicted": mean_pred,
                    "mean_observed": mean_obs,
                    "gap": gap,
                }
            )
        result[str(label)] = {
            "expected_calibration_error": float(ece),
            "max_bin_gap": float(max_gap),
            "bins": bins,
        }
    return result


def fairness_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Sequence[Hashable],
    scores: Optional[Sequence[float]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute the full fairness diagnostics bundle.

    Returns a JSON-serializable dict with per-group metrics, overall metrics,
    and the disparity summaries (demographic parity, equalized odds, equal
    opportunity, disparate impact), plus calibration-by-group when ``scores``
    are provided.
    """
    grp = per_group_metrics(y_true, y_pred, groups)
    report: Dict[str, Any] = {
        "n_samples": len(np.asarray(y_true)),
        "n_groups": len(grp),
        "overall": overall_metrics(y_true, y_pred).to_dict(),
        "groups": {g: gm.to_dict() for g, gm in grp.items()},
        "demographic_parity": demographic_parity(grp),
        "equalized_odds": equalized_odds(grp),
        "equal_opportunity": equal_opportunity(grp),
        "disparate_impact": disparate_impact(grp),
    }
    if scores is not None:
        report["calibration_by_group"] = calibration_by_group(
            y_true, scores, groups, n_bins=n_bins
        )
    return report
