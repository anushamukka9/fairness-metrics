"""JSON and Markdown report rendering for fairness audits."""

from __future__ import annotations

import json
from typing import Any, Dict

__all__ = ["to_json", "to_markdown"]


def to_json(report: Dict[str, Any], indent: int = 2) -> str:
    """Serialize a fairness report to JSON."""
    return json.dumps(report, indent=indent, default=str)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def to_markdown(report: Dict[str, Any], title: str = "Fairness Audit Report") -> str:
    """Render a fairness report as a human-readable Markdown document."""
    lines = [f"# {title}", ""]
    lines.append(f"- Samples: **{report.get('n_samples', '—')}**")
    lines.append(f"- Groups: **{report.get('n_groups', '—')}**")
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
