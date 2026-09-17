"""Threshold scanning and fairness-constrained threshold selection.

A single global threshold often produces unequal outcomes across groups. This
module sweeps decision thresholds — shared or per-group — and finds operating
points that keep a chosen disparity measure under a bound while maximizing a
utility objective.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Sequence

import numpy as np

from .metrics import (
    demographic_parity,
    disparate_impact,
    equal_opportunity,
    equalized_odds,
    per_group_metrics,
)

__all__ = [
    "threshold_scan",
    "find_fair_thresholds",
    "CONSTRAINT_MEASURES",
]

_CONSTRAINT_FUNCS = {
    "demographic_parity": lambda grp: demographic_parity(grp)["max_difference"],
    "equalized_odds": lambda grp: equalized_odds(grp)["max_difference"],
    "equal_opportunity": lambda grp: equal_opportunity(grp)["tpr_max_difference"],
    "disparate_impact_gap": lambda grp: 1.0 - disparate_impact(grp)["min_ratio"],
}

CONSTRAINT_MEASURES = sorted(_CONSTRAINT_FUNCS)

_OBJECTIVES = {
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
    "true_positive_rate": "true_positive_rate",
}


def _validate_scores(
    y_true: Sequence[Any], scores: Sequence[float], groups: Sequence[Hashable]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .metrics import _binary_array, _check_aligned, _group_array  # noqa: PLC2701

    yt = _binary_array(y_true, "y_true")
    sc = np.asarray(scores, dtype=float)
    gr = _group_array(groups, "groups")
    _check_aligned(yt, sc, gr)
    if sc.ndim != 1 or np.any((sc < 0) | (sc > 1)):
        raise ValueError("scores must be a 1-D sequence of probabilities in [0, 1]")
    return yt, sc, gr


def threshold_scan(
    y_true: Sequence[Any],
    scores: Sequence[float],
    groups: Sequence[Hashable],
    thresholds: Optional[Sequence[float]] = None,
    per_group: bool = False,
) -> Dict[str, Any]:
    """Sweep decision thresholds and record group metrics at each one.

    With ``per_group=False`` every threshold applies to all groups jointly;
    with ``per_group=True`` the returned table is indexed per group, which
    lets you compare how the same operating point treats each group.
    """
    yt, sc, gr = _validate_scores(y_true, scores, groups)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = [float(t) for t in thresholds]
    if not thresholds or any(not 0 < t < 1 for t in thresholds):
        raise ValueError("thresholds must be a non-empty sequence of values in (0, 1)")

    labels = sorted(set(gr.tolist()), key=str)
    rows = []
    for t in thresholds:
        preds = (sc >= t).astype(int)
        grp = per_group_metrics(yt, preds, gr)
        row: Dict[str, Any] = {
            "threshold": t,
            "groups": {g: gm.to_dict() for g, gm in grp.items()},
            "demographic_parity_max_difference": demographic_parity(grp)["max_difference"],
            "disparate_impact_min_ratio": disparate_impact(grp)["min_ratio"],
            "equalized_odds_max_difference": equalized_odds(grp)["max_difference"],
        }
        rows.append(row)
    return {"thresholds": thresholds, "per_group_thresholds": per_group, "scan": rows, "groups": labels}


def find_fair_thresholds(
    y_true: Sequence[Any],
    scores: Sequence[float],
    groups: Sequence[Hashable],
    constraint: str = "demographic_parity",
    max_gap: float = 0.1,
    objective: str = "balanced_accuracy",
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Choose per-group thresholds that satisfy a fairness constraint.

    For each group independently, searches the threshold grid for the point
    that maximizes ``objective``. Then, among candidate per-group threshold
    combinations on the grid, selects the one maximizing the worst-group
    objective while keeping the chosen disparity measure at or below
    ``max_gap``. The search is greedy but exact on the provided grid: group
    thresholds are first picked individually, then adjusted only if needed to
    meet the constraint.

    Returns the selected thresholds, the resulting per-group metrics, and the
    achieved disparity — so the trade-off is auditable, not a black box.
    """
    if constraint not in _CONSTRAINT_FUNCS:
        raise ValueError(f"constraint must be one of {CONSTRAINT_MEASURES}, got {constraint!r}")
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(_OBJECTIVES)}, got {objective!r}")
    if not 0 <= max_gap <= 1:
        raise ValueError("max_gap must be in [0, 1]")

    yt, sc, gr = _validate_scores(y_true, scores, groups)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    grid = [float(t) for t in thresholds]

    labels = sorted(set(gr.tolist()), key=str)
    # Best unconstrained threshold per group (objective only).
    per_group_best: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        mask = gr == label
        best_t, best_obj = grid[0], float("-inf")
        for t in grid:
            preds = (sc[mask] >= t).astype(int)
            gm = per_group_metrics(yt[mask], preds, np.zeros(int(np.sum(mask)), dtype=object))
            obj = getattr(gm["0"], _OBJECTIVES[objective])
            if obj > best_obj:
                best_obj, best_t = obj, t
        per_group_best[label] = {"threshold": best_t, "objective": best_obj}

    def evaluate(chosen: Dict[str, float]) -> Dict[str, Any]:
        preds = np.zeros_like(yt)
        for label in labels:
            mask = gr == label
            preds[mask] = (sc[mask] >= chosen[label]).astype(int)
        grp = per_group_metrics(yt, preds, gr)
        gap = _CONSTRAINT_FUNCS[constraint](grp)
        objs = [getattr(gm, _OBJECTIVES[objective]) for gm in grp.values()]
        return {"gap": gap, "min_objective": min(objs), "group_metrics": grp}

    # Start from the per-group best thresholds.
    chosen = {label: info["threshold"] for label, info in per_group_best.items()}
    ev = evaluate(chosen)

    # If the constraint is violated, coordinate-descent on the grid: at each
    # step try every single-group threshold move and take the one that keeps
    # the minimum objective highest while reducing the gap.
    for _ in range(50):
        if ev["gap"] <= max_gap:
            break
        best_move = None
        for label in labels:
            for t in grid:
                if t == chosen[label]:
                    continue
                trial = dict(chosen)
                trial[label] = t
                trial_ev = evaluate(trial)
                if trial_ev["gap"] < ev["gap"] and (
                    best_move is None
                    or trial_ev["min_objective"] > best_move[1]["min_objective"]
                ):
                    best_move = (trial, trial_ev)
        if best_move is None:
            break  # no single move improves the gap: grid-optimal stall
        chosen, ev = best_move

    return {
        "constraint": constraint,
        "max_gap": max_gap,
        "objective": objective,
        "thresholds": {label: chosen[label] for label in labels},
        "achieved_gap": float(ev["gap"]),
        "constraint_satisfied": bool(ev["gap"] <= max_gap),
        "min_group_objective": float(ev["min_objective"]),
        "groups": {g: gm.to_dict() for g, gm in ev["group_metrics"].items()},
        "unconstrained_per_group_best": per_group_best,
    }
