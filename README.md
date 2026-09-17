# fairness-metrics

Group fairness diagnostics for binary classifiers: **demographic parity,
equalized odds, equal opportunity, disparate impact, calibration-by-group**,
threshold scanning for fairer operating points, and JSON/Markdown audit
reports — plus a CLI that scores a predictions CSV. Only dependency: `numpy`.

By [Anusha Mukka](https://anushamukka.com).

## Install

```bash
pip install git+https://github.com/anushamukka9/fairness-metrics.git
# or, from a clone:
pip install .
```

## Quickstart

```python
from fairness_metrics import fairness_report, to_markdown

y_true  = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred  = [0, 1, 0, 1, 0, 0, 1, 1]   # model predictions at some threshold
groups  = ["a", "a", "a", "a", "b", "b", "b", "b"]  # sensitive attribute
scores  = [0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.7, 0.6]  # optional probabilities

report = fairness_report(y_true, y_pred, groups, scores=scores)
print(to_markdown(report))
```

A runnable version lives in `examples/quickstart.py` with
`examples/sample_predictions.csv`.

## CLI

Score a predictions CSV and emit JSON (default) or Markdown:

```bash
fairness-metrics score examples/sample_predictions.csv \
  --y-true label --scores score --group group --markdown

# binary predictions instead of scores:
fairness-metrics score preds.csv --y-true y --y-pred yhat --group segment \
  -o report.json
```

Scan decision thresholds and find per-group operating points that satisfy a
fairness constraint (default: demographic parity gap ≤ 0.1):

```bash
fairness-metrics scan examples/sample_predictions.csv \
  --y-true label --scores score --group group \
  --find --constraint demographic_parity --max-gap 0.1 --markdown
```

Constraints: `demographic_parity`, `equalized_odds`, `equal_opportunity`,
`disparate_impact_gap`. Objectives: `balanced_accuracy`, `accuracy`,
`true_positive_rate`.

## API

```python
from fairness_metrics import (
    per_group_metrics,      # per-group confusion metrics
    overall_metrics,        # aggregate metrics
    demographic_parity,     # selection-rate parity across groups
    equalized_odds,         # TPR and FPR parity
    equal_opportunity,      # TPR parity
    disparate_impact,       # min selection-rate ratio + four-fifths rule
    calibration_by_group,   # per-group ECE from probabilities
    threshold_scan,         # metrics across a threshold grid
    find_fair_thresholds,   # per-group thresholds under a fairness constraint
    fairness_report,        # full diagnostics bundle (JSON-serializable)
    to_json, to_markdown,   # report rendering
)
```

See [`docs/usage.md`](docs/usage.md) for the full guide.

## Architecture

```
src/fairness_metrics/
├── __init__.py      # public API surface
├── metrics.py       # confusion metrics + disparity measures + calibration
├── thresholds.py    # threshold scanning + fairness-constrained search
├── report.py        # JSON / Markdown report rendering
└── cli.py           # `fairness-metrics` console script (score / scan)
tests/               # pytest suite
examples/            # runnable quickstart + sample predictions CSV
docs/                # usage guide
```

`metrics.py` is the foundation: everything reduces predictions/labels/groups
to per-group confusion tables, then derives disparity summaries from them.
`thresholds.py` builds on those metrics to explore and constrain operating
points. `report.py` renders the resulting dicts; `cli.py` wires it all to
CSVs.

## Notes

- Labels and predictions must be binary (0/1). Scores must be probabilities
  in [0, 1].
- The four-fifths rule is a screening heuristic, not a legal test of
  discrimination. Per-group thresholds can raise their own compliance and
  product questions — use `find_fair_thresholds` to *diagnose* trade-offs,
  not as an automatic deployment rule.

## License

MIT — see [LICENSE](LICENSE). Copyright 2026 Anusha Mukka.
