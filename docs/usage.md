# Usage guide

A walkthrough of `fairness-metrics` for auditing a binary classifier.

## 1. What you need

- `y_true`: true binary labels (0/1)
- `y_pred`: predicted binary labels (0/1), **or** `scores`: predicted
  probabilities in [0, 1] plus a threshold
- `groups`: the sensitive attribute (any hashable labels — strings, ints —
  any number of groups)

```python
from fairness_metrics import fairness_report, to_json, to_markdown

report = fairness_report(y_true, y_pred, groups, scores=scores)
open("audit.json", "w").write(to_json(report))
print(to_markdown(report))
```

## 2. Reading the report

- **`overall` / `groups`**: per-group confusion-derived metrics — selection
  rate P(ŷ=1), TPR/recall, FPR, TNR/specificity, precision, accuracy,
  balanced accuracy, and sample counts. Start here: eyeball the table before
  trusting any summary number.
- **`demographic_parity`**: max pairwise difference and min ratio of
  selection rates. Difference 0 / ratio 1 means perfect parity. This measure
  ignores ground truth — it is the right lens when outcomes themselves should
  be allocated equally (e.g. ad targeting), the wrong lens when base rates
  legitimately differ.
- **`equalized_odds`**: TPR *and* FPR parity — the model's errors are equally
  distributed across groups. Use when both false positives and false negatives
  carry cost.
- **`equal_opportunity`**: TPR parity only — qualified members of each group
  are selected at equal rates. Use when the positive class is an opportunity
  (hiring, lending) and recall matters most.
- **`disparate_impact`**: the min selection-rate ratio plus the four-fifths
  rule screen (ratio ≥ 0.8 passes). A flag here starts an investigation; it
  does not prove discrimination.
- **`calibration_by_group`** (requires `scores`): expected calibration error
  per group. A model can satisfy parity while being badly miscalibrated for
  one group — check both.

## 3. Finding fairer operating points

A single global threshold rarely treats groups equally. Two tools:

```python
from fairness_metrics import threshold_scan, find_fair_thresholds

# How do the disparities move as the threshold changes?
scan = threshold_scan(y_true, scores, groups)
for row in scan["scan"]:
    print(row["threshold"],
          "parity gap:", round(row["demographic_parity_max_difference"], 3),
          "DI ratio:", round(row["disparate_impact_min_ratio"], 3))

# Per-group thresholds: parity gap <= 0.1, maximizing worst-group
# balanced accuracy.
result = find_fair_thresholds(
    y_true, scores, groups,
    constraint="demographic_parity", max_gap=0.1,
    objective="balanced_accuracy",
)
print(result["thresholds"])            # {'a': 0.45, 'b': 0.55, ...}
print(result["constraint_satisfied"])   # True/False — always check
```

The search is exact on the threshold grid (default 19 points from 0.05 to
0.95) via greedy coordinate descent: it starts from each group's individually
best threshold and only moves thresholds while that reduces the disparity
gap. The returned dict includes the achieved gap and the unconstrained
per-group best, so the fairness/utility trade-off is auditable.

Available constraints: `demographic_parity`, `equalized_odds`,
`equal_opportunity`, `disparate_impact_gap`. Available objectives:
`balanced_accuracy`, `accuracy`, `true_positive_rate`.

## 4. The CLI

```bash
# JSON report to stdout
fairness-metrics score preds.csv --y-true y --y-pred yhat --group segment

# Markdown report to a file, probabilities instead of hard predictions
fairness-metrics score preds.csv --y-true y --scores p --group segment \
    --threshold 0.4 --markdown -o audit.md

# Threshold scan (JSON)
fairness-metrics scan preds.csv --y-true y --scores p --group segment

# Fair threshold search, Markdown output
fairness-metrics scan preds.csv --y-true y --scores p --group segment \
    --find --constraint equalized_odds --max-gap 0.05 --markdown
```

The CSV just needs the named columns; extra columns are ignored.

## 5. Tips

- Always inspect per-group `n` before acting: disparity measures on tiny
  groups are noise.
- `find_fair_thresholds` may report `constraint_satisfied: False` if no grid
  point meets the bound — that is a finding, not a bug. Loosen `max_gap`,
  widen the grid, or fix the model.
- Metrics that are undefined for a group (e.g. TPR when the group has no
  positive labels) surface as NaN / "n/a" rather than silently dropping the
  group.
