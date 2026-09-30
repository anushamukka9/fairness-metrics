"""JSON, Markdown, and plain-text report rendering for fairness audits."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

__all__ = ["to_json", "to_markdown", "to_text"]


def to_json(report: Dict[str, Any], indent: int = 2) -> str:
    """Serialize a fairness report to JSON."""
    return json.dumps(report, indent=indent, default=str)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def to_markdown(report: Dict[str, Any], title: str = "Fairness Audit Report") -> str:
    """Render a fairness report as a human-readable Markdown document."""
    lines = [f"# {title}", ""]
    lines.append(f"- Samples: **{report.get('n_samples', 'n/a')}**")
    lines.append(f"- Groups: **{report.get('n_groups', 'n/a')}**")
    lines.append("")

    overall = report.get("overall", {})
    lines.append("## Overall performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key in (
        "accuracy",
        "balanced_accuracy",
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
    ):
        lines.append(f"| {key.replace('_', ' ').title()} | {_fmt(overall.get(key))} |")
    lines.append("")

    groups = report.get("groups", {})
    lines.append("## Per-group breakdown")
    lines.append("")
    header = (
        "| Group | n | Sel. rate | TPR | FPR | TNR | Precision | Accuracy | Bal. acc. |"
    )
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, gm in groups.items():
        lines.append(
            f"| {name} | {gm.get('n')} | {_fmt(gm.get('selection_rate'))} | "
            f"{_fmt(gm.get('true_positive_rate'))} | {_fmt(gm.get('false_positive_rate'))} | "
            f"{_fmt(gm.get('true_negative_rate'))} | {_fmt(gm.get('precision'))} | "
            f"{_fmt(gm.get('accuracy'))} | {_fmt(gm.get('balanced_accuracy'))} |"
        )
    lines.append("")

    def disparity_section(heading: str, block: Dict[str, Any], keys: list) -> None:
        lines.append(f"## {heading}")
        lines.append("")
        for key in keys:
            label = key.replace("_", " ").title()
            lines.append(f"- {label}: **{_fmt(block.get(key))}**")
        lines.append("")

    if "demographic_parity" in report:
        disparity_section(
            "Demographic parity", report["demographic_parity"],
            ["max_difference", "min_ratio"],
        )
    if "equalized_odds" in report:
        disparity_section(
            "Equalized odds", report["equalized_odds"],
            ["tpr_max_difference", "fpr_max_difference", "max_difference"],
        )
    if "equal_opportunity" in report:
        disparity_section(
            "Equal opportunity", report["equal_opportunity"], ["tpr_max_difference"]
        )
    if "predictive_parity" in report:
        disparity_section(
            "Predictive parity", report["predictive_parity"],
            ["max_difference", "min_ratio"],
        )
    if "disparate_impact" in report:
        block = report["disparate_impact"]
        lines.append("## Disparate impact")
        lines.append("")
        lines.append(f"- Min selection-rate ratio: **{_fmt(block.get('min_ratio'))}**")
        lines.append(f"- Four-fifths threshold: **{_fmt(block.get('four_fifths_threshold'))}**")
        passed = block.get("four_fifths_pass")
        lines.append(f"- Four-fifths rule: **{'PASS' if passed else 'FAIL'}**")
        lines.append("")

    calib = report.get("calibration_by_group")
    if calib:
        lines.append("## Calibration by group")
        lines.append("")
        lines.append("| Group | ECE | Max bin gap |")
        lines.append("| --- | --- | --- |")
        for name, cb in calib.items():
            lines.append(
                f"| {name} | {_fmt(cb.get('expected_calibration_error'))} | "
                f"{_fmt(cb.get('max_bin_gap'))} |"
            )
        lines.append("")

    lines.append(
        "_Lower disparity differences and higher ratios are fairer. "
        "The four-fifths rule is a screening heuristic, not a legal test._"
    )
    return "\n".join(lines)


_PER_GROUP_COLUMNS = [
    ("n", 6),
    ("sel_rate", 9),
    ("tpr", 7),
    ("fpr", 7),
    ("precision", 10),
    ("accuracy", 9),
]


def to_text(report: Dict[str, Any], title: str = "Fairness Audit Report") -> str:
    """Render a fairness report as plain ASCII text.

    Same content as :func:`to_markdown` in a terminal/CLI-friendly layout -
    no Markdown syntax, no Unicode beyond ASCII.
    """
    bar = "=" * 64
    lines = [bar, title, bar, ""]
    lines.append(f"Samples: {report.get('n_samples', 'n/a')}   "
                 f"Groups: {report.get('n_groups', 'n/a')}")
    lines.append("")

    overall = report.get("overall", {})
    lines.append("Overall performance")
    lines.append("-" * 64)
    for key in ("accuracy", "balanced_accuracy", "selection_rate",
                "true_positive_rate", "false_positive_rate", "precision"):
        lines.append(f"  {key:20s} {_fmt(overall.get(key))}")
    lines.append("")

    lines.append("Per-group breakdown")
    lines.append("-" * 64)
    header = "  " + f"{'group':10s}" + "".join(f"{col:>{w}s}" for col, w in _PER_GROUP_COLUMNS)
    lines.append(header)
    for name, gm in report.get("groups", {}).items():
        row = (
            f"  {str(name)[:10]:10s}"
            f"{_fmt(gm.get('n')):>6s}"
            f"{_fmt(gm.get('selection_rate')):>9s}"
            f"{_fmt(gm.get('true_positive_rate')):>7s}"
            f"{_fmt(gm.get('false_positive_rate')):>7s}"
            f"{_fmt(gm.get('precision')):>10s}"
            f"{_fmt(gm.get('accuracy')):>9s}"
        )
        lines.append(row)
    lines.append("")

    def disparity_block(heading: str, block: Dict[str, Any], keys: List[str]) -> None:
        lines.append(heading)
        lines.append("-" * 64)
        for key in keys:
            lines.append(f"  {key:24s} {_fmt(block.get(key))}")
        lines.append("")

    if "demographic_parity" in report:
        disparity_block("Demographic parity", report["demographic_parity"],
                        ["max_difference", "min_ratio"])
    if "equalized_odds" in report:
        disparity_block("Equalized odds", report["equalized_odds"],
                        ["tpr_max_difference", "fpr_max_difference", "max_difference"])
    if "equal_opportunity" in report:
        disparity_block("Equal opportunity", report["equal_opportunity"],
                        ["tpr_max_difference"])
    if "predictive_parity" in report:
        disparity_block("Predictive parity", report["predictive_parity"],
                        ["max_difference", "min_ratio"])
    if "disparate_impact" in report:
        block = report["disparate_impact"]
        lines.append("Disparate impact")
        lines.append("-" * 64)
        lines.append(f"  {'min_ratio':24s} {_fmt(block.get('min_ratio'))}")
        lines.append(f"  {'four_fifths_threshold':24s} {_fmt(block.get('four_fifths_threshold'))}")
        passed = block.get("four_fifths_pass")
        lines.append(f"  {'four_fifths_rule':24s} {'PASS' if passed else 'FAIL'}")
        lines.append("")

    calib = report.get("calibration_by_group")
    if calib:
        lines.append("Calibration by group")
        lines.append("-" * 64)
        for name, cb in calib.items():
            lines.append(f"  {str(name)[:12]:12s} ECE={_fmt(cb.get('expected_calibration_error'))} "
                         f"max_bin_gap={_fmt(cb.get('max_bin_gap'))}")
        lines.append("")

    lines.append("Lower disparity differences and higher ratios are fairer. "
                 "The four-fifths rule is a screening heuristic, not a legal test.")
    lines.append(bar)
    return "\n".join(lines)
