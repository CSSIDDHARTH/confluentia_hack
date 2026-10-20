import csv
import io
import json
from pathlib import Path

from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.compliance import compute_report
from shelfsense.report.export import (
    report_to_json,
    save_report_json,
    save_violations_csv,
    violations_to_csv,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


def _report():
    det = load_detection_output(CONTRACTS / "detection_output.example.json")
    planogram = load_planogram(CONTRACTS / "planogram.example.json")
    catalogue = load_catalogue(CONTRACTS / "catalogue.example.csv")
    return compute_report(det, planogram, catalogue)


def test_json_export_round_trips_score_and_violation_count():
    report = _report()
    data = json.loads(report_to_json(report))

    assert data["score"] == report.score
    assert len(data["violations"]) == len(report.violations)
    assert data["image_id"] == report.image_id


def test_csv_export_has_one_row_per_violation_with_expected_columns():
    report = _report()
    rows = list(csv.DictReader(io.StringIO(violations_to_csv(report))))

    assert len(rows) == len(report.violations)
    assert set(rows[0].keys()) == {
        "violation_id", "kind", "shelf_id", "slot_id", "sku", "detail", "recommended_action",
    }
    assert {r["violation_id"] for r in rows} == {v.violation_id for v in report.violations}


def test_save_report_json_writes_loadable_file(tmp_path):
    report = _report()
    out = tmp_path / "report.json"

    save_report_json(report, out)
    data = json.loads(out.read_text())

    assert data["score"] == report.score


def test_save_violations_csv_writes_loadable_file(tmp_path):
    report = _report()
    out = tmp_path / "violations.csv"

    save_violations_csv(report, out)
    rows = list(csv.DictReader(out.open()))

    assert len(rows) == len(report.violations)


def test_empty_violations_csv_still_has_header():
    report = _report()
    report = report.model_copy(update={"violations": []})

    rows = list(csv.DictReader(io.StringIO(violations_to_csv(report))))

    assert rows == []
