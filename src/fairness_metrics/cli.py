"""Command-line interface: score a predictions CSV or scan thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

from .metrics import fairness_report
from .report import to_json, to_markdown
from .thresholds import CONSTRAINT_MEASURES, find_fair_thresholds, threshold_scan

PROG = "fairness-metrics"


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"error: {path} is empty or has no data rows")
    return rows


def _column(rows: List[dict], name: str):
    if name not in rows[0]:
        available = ", ".join(rows[0].keys())
        raise SystemExit(f"error: column {name!r} not found (available: {available})")
    return [r[name] for r in rows]


def cmd_score(args: argparse.Namespace) -> int:
    rows = _read_csv(args.csv)
    y_true = [int(float(v)) for v in _column(rows, args.y_true)]
    group = _column(rows, args.group)
    if args.scores:
        scores = [float(v) for v in _column(rows, args.scores)]
        y_pred = [1 if s >= args.threshold else 0 for s in scores]
    else:
        scores = None
        y_pred = [int(float(v)) for v in _column(rows, args.y_pred)]
    report = fairness_report(y_true, y_pred, group, scores=scores, n_bins=args.bins)
    output = (
        to_markdown(report, title=f"Fairness Audit — {Path(args.csv).name}")
        if args.markdown
        else to_json(report)
    )
    if args.output:
        Path(args.output).write_text(output)
        print(f"wrote {args.output}")
    else:
        print(output)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    rows = _read_csv(args.csv)
    y_true = [int(float(v)) for v in _column(rows, args.y_true)]
    scores = [float(v) for v in _column(rows, args.scores)]
    group = _column(rows, args.group)
    thresholds = (
        [float(t) for t in args.grid.split(",")] if args.grid else None
    )
    if args.find:
        result = find_fair_thresholds(
            y_true,
            scores,
            group,
            constraint=args.constraint,
            max_gap=args.max_gap,
            objective=args.objective,
            thresholds=thresholds,
        )
        output = to_json(result) if not args.markdown else _fair_thresholds_markdown(result)
    else:
        result = threshold_scan(y_true, scores, group, thresholds=thresholds)
        if args.markdown:
            lines = ["# Threshold Scan", ""]
            for row in result["scan"]:
                lines.append(
                    f"- t={row['threshold']:.2f}: parity gap "
                    f"{row['demographic_parity_max_difference']:.4f}, "
                    f"DI ratio {row['disparate_impact_min_ratio']:.4f}"
                )
            output = "\n".join(lines)
        else:
            output = to_json(result)
    if args.output:
        Path(args.output).write_text(output)
        print(f"wrote {args.output}")
    else:
        print(output)
    return 0


def _fair_thresholds_markdown(result: dict) -> str:
    lines = ["# Fair Threshold Selection", ""]
    lines.append(f"- Constraint: **{result['constraint']}** (max gap {result['max_gap']})")
    lines.append(f"- Achieved gap: **{result['achieved_gap']:.4f}**")
    lines.append(
        f"- Constraint satisfied: **{'YES' if result['constraint_satisfied'] else 'NO'}**"
    )
    lines.append("")
    lines.append("| Group | Threshold |")
    lines.append("| --- | --- |")
    for g, t in result["thresholds"].items():
        lines.append(f"| {g} | {t:.3f} |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Compute group fairness metrics from model predictions.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score predictions and emit a fairness report")
    s.add_argument("csv", help="CSV with labels, predictions/scores, and group column")
    s.add_argument("--y-true", required=True, help="column with true binary labels (0/1)")
    s.add_argument("--y-pred", help="column with predicted binary labels (0/1)")
    s.add_argument("--scores", help="column with predicted probabilities in [0, 1]")
    s.add_argument("--group", required=True, help="column with sensitive group labels")
    s.add_argument("--threshold", type=float, default=0.5,
                   help="threshold applied to --scores (default 0.5)")
    s.add_argument("--bins", type=int, default=10, help="calibration bins (default 10)")
    s.add_argument("--markdown", action="store_true", help="emit Markdown instead of JSON")
    s.add_argument("--output", "-o", help="write report to file instead of stdout")
    s.set_defaults(func=cmd_score)

    t = sub.add_parser("scan", help="sweep thresholds and find fair operating points")
    t.add_argument("csv", help="CSV with labels, scores, and group column")
    t.add_argument("--y-true", required=True, help="column with true binary labels (0/1)")
    t.add_argument("--scores", required=True, help="column with predicted probabilities")
    t.add_argument("--group", required=True, help="column with sensitive group labels")
    t.add_argument("--grid", help="comma-separated threshold grid, e.g. 0.3,0.5,0.7")
    t.add_argument("--find", action="store_true",
                   help="search for per-group thresholds meeting a fairness constraint")
    t.add_argument("--constraint", default="demographic_parity",
                   choices=CONSTRAINT_MEASURES,
                   help="disparity measure to constrain (default demographic_parity)")
    t.add_argument("--max-gap", type=float, default=0.1,
                   help="maximum allowed disparity gap (default 0.1)")
    t.add_argument("--objective", default="balanced_accuracy",
                   choices=["balanced_accuracy", "accuracy", "true_positive_rate"],
                   help="utility to maximize (default balanced_accuracy)")
    t.add_argument("--markdown", action="store_true", help="emit Markdown instead of JSON")
    t.add_argument("--output", "-o", help="write output to file instead of stdout")
    t.set_defaults(func=cmd_scan)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score" and not args.y_pred and not args.scores:
        parser.error("score requires one of --y-pred or --scores")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
