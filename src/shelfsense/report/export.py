"""Export a ComplianceReport to JSON or CSV, for download from the
dashboard or for feeding into some other tool.

JSON keeps the full report (score + every violation field). CSV is one
row per violation -- handy for a spreadsheet, but it drops the score and
counts, so pair it with the JSON export if those matter too.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from shelfsense.report.compliance import ComplianceReport

CSV_FIELDS = [
    "violation_id",
    "kind",
    "shelf_id",
    "slot_id",
    "sku",
    "detail",
    "recommended_action",
]


def report_to_json(report: ComplianceReport) -> str:
    return report.model_dump_json(indent=2)


def violations_to_csv(report: ComplianceReport) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for v in report.violations:
        writer.writerow({field: getattr(v, field) for field in CSV_FIELDS})
    return buf.getvalue()


def save_report_json(report: ComplianceReport, path: str | Path) -> None:
    Path(path).write_text(report_to_json(report))


def save_violations_csv(report: ComplianceReport, path: str | Path) -> None:
    Path(path).write_text(violations_to_csv(report))
