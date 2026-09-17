"""Quickstart: score a predictions CSV and print a fairness report.

Run from the repo root::

    python examples/quickstart.py

Or generate the example CSV yourself and call the CLI::

    fairness-metrics score examples/sample_predictions.csv \\
        --y-true label --scores score --group group --markdown
"""

import csv
from pathlib import Path

from fairness_metrics import fairness_report, to_markdown

HERE = Path(__file__).parent
CSV = HERE / "sample_predictions.csv"


def main() -> None:
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    y_true = [int(r["label"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    groups = [r["group"] for r in rows]
    y_pred = [1 if s >= 0.5 else 0 for s in scores]

    report = fairness_report(y_true, y_pred, groups, scores=scores)
    print(to_markdown(report, title="Fairness Audit — sample_predictions.csv"))

    dp = report["demographic_parity"]
    print()
    print(f"Parity gap: {dp['max_difference']:.3f} | "
          f"DI ratio: {report['disparate_impact']['min_ratio']:.3f}")


if __name__ == "__main__":
    main()
