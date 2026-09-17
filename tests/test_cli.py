"""CLI tests: score and scan subcommands against a temp predictions CSV."""

import csv
import json

import pytest

from fairness_metrics.cli import main


@pytest.fixture()
def predictions_csv(tmp_path):
    path = tmp_path / "predictions.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "score", "group"])
        rows = [
            (1, 0.9, "a"), (0, 0.2, "a"), (1, 0.8, "a"), (0, 0.3, "a"),
            (1, 0.9, "b"), (0, 0.2, "b"), (1, 0.1, "b"), (0, 0.8, "b"),
        ]
        w.writerows(rows)
    return str(path)


def test_cli_score_json(predictions_csv, tmp_path, capsys):
    out = tmp_path / "report.json"
    rc = main(["score", predictions_csv, "--y-true", "label", "--scores", "score",
               "--group", "group", "--output", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["n_samples"] == 8
    assert report["n_groups"] == 2
    assert "demographic_parity" in report
    assert "calibration_by_group" in report  # scores -> calibration included


def test_cli_score_markdown_stdout(predictions_csv, capsys):
    rc = main(["score", predictions_csv, "--y-true", "label", "--scores", "score",
               "--group", "group", "--markdown"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "# Fairness Audit" in text
    assert "Four-fifths rule" in text


def test_cli_score_with_y_pred_column(predictions_csv, tmp_path):
    # predictions.csv has no y_pred column -> error mentioning available columns
    with pytest.raises(SystemExit):
        main(["score", predictions_csv, "--y-true", "label",
              "--y-pred", "prediction", "--group", "group"])


def test_cli_scan_find_fair_thresholds(predictions_csv, tmp_path, capsys):
    out = tmp_path / "scan.json"
    rc = main(["scan", predictions_csv, "--y-true", "label", "--scores", "score",
               "--group", "group", "--find", "--constraint", "demographic_parity",
               "--max-gap", "0.2", "--output", str(out)])
    assert rc == 0
    result = json.loads(out.read_text())
    assert set(result["thresholds"]) == {"a", "b"}
    assert "achieved_gap" in result


def test_cli_missing_column_errors(predictions_csv):
    with pytest.raises(SystemExit):
        main(["score", predictions_csv, "--y-true", "nope",
              "--scores", "score", "--group", "group"])
