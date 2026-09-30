"""Toy bias demo: watch fairness metrics surface a skewed classifier.

Builds a small synthetic loan-approval dataset (purely illustrative - no
real data, no claims about any real population). A single 0.5 decision
threshold approves far fewer "rural" applicants than "urban" ones. The demo
prints the plain-text audit report, then uses find_fair_thresholds to pick
per-group thresholds that shrink the demographic-parity gap, and prints the
before/after comparison.

Run from the repo root:

    python examples/toy_bias_demo.py
"""

import numpy as np

from fairness_metrics import (
    fairness_report,
    find_fair_thresholds,
    to_text,
)

rng = np.random.default_rng(20260929)


def make_group(n: int, score_shift: float):
    """Synthetic applicants: same base approval-worthiness, shifted scores.

    ``score_shift`` models a miscalibrated scorecard that systematically
    under-scores one group - the kind of skew fairness metrics exist to
    surface.
    """
    y_true = (rng.random(n) < 0.5).astype(int)
    scores = np.clip(
        y_true * 0.55 + 0.10 + score_shift + rng.normal(0, 0.15, n),
        0.01,
        0.99,
    )
    return y_true, scores


def main() -> None:
    n = 400
    y_a, s_a = make_group(n, score_shift=0.00)    # group "urban"
    y_b, s_b = make_group(n, score_shift=-0.22)   # group "rural" (under-scored)

    y_true = np.concatenate([y_a, y_b]).tolist()
    scores = np.concatenate([s_a, s_b]).tolist()
    groups = ["urban"] * n + ["rural"] * n

    # 1. One shared threshold for everyone - the naive default.
    y_pred = [1 if s >= 0.5 else 0 for s in scores]
    before = fairness_report(y_true, y_pred, groups, scores=scores)
    print(to_text(before, title="Before: single 0.5 threshold for all groups"))
    print()

    dp_before = before["demographic_parity"]["max_difference"]
    di_before = before["disparate_impact"]["min_ratio"]
    print(f"Parity gap before: {dp_before:.3f} | DI ratio before: {di_before:.3f}")
    print()

    # 2. Diagnose with per-group thresholds under a parity constraint.
    result = find_fair_thresholds(
        y_true,
        scores,
        groups,
        constraint="demographic_parity",
        max_gap=0.05,
        objective="balanced_accuracy",
    )
    print("Per-group thresholds chosen by find_fair_thresholds:")
    for group, threshold in result["thresholds"].items():
        print(f"  {group:8s} threshold = {threshold:.2f}")
    print(f"Achieved parity gap: {result['achieved_gap']:.3f} "
          f"(constraint satisfied: {result['constraint_satisfied']})")

    # 3. Re-score with the chosen per-group thresholds and compare.
    y_pred_after = [
        1 if s >= result["thresholds"][g] else 0
        for s, g in zip(scores, groups)
    ]
    after = fairness_report(y_true, y_pred_after, groups, scores=scores)
    dp_after = after["demographic_parity"]["max_difference"]
    di_after = after["disparate_impact"]["min_ratio"]
    print()
    print("Before -> after (per-group thresholds):")
    print(f"  parity gap: {dp_before:.3f} -> {dp_after:.3f}")
    print(f"  DI ratio:   {di_before:.3f} -> {di_after:.3f}")
    print(f"  min group balanced accuracy: "
          f"{min(g['balanced_accuracy'] for g in before['groups'].values()):.3f} -> "
          f"{min(g['balanced_accuracy'] for g in after['groups'].values()):.3f}")
    print()
    print("Note: per-group thresholds trade one concern for another - this demo")
    print("diagnoses the trade-off; it is not a deployment recommendation.")


if __name__ == "__main__":
    main()
