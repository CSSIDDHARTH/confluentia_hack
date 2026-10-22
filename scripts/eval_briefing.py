"""Run the shift briefing on every dashboard scenario and print each summary,
whether the small model's summary passed the check, and how long it took.

    python scripts/eval_briefing.py
"""

from pathlib import Path

from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.briefing import write_briefing
from shelfsense.report.compliance import compute_report

ROOT = Path(__file__).resolve().parents[1]
MOCK = ("contracts/planogram.example.json", "contracts/catalogue.example.csv")


def scenarios():
    for det in ("contracts/detection_output.example.json", "data/mock/mock_shelf_002_lowstock.json",
                "data/mock/mock_shelf_003_clean.json"):
        yield det, *MOCK, False
    for bay in sorted((ROOT / "data" / "demo_bays").iterdir()):
        rel = bay.relative_to(ROOT)
        yield f"{rel}/detection_compliant.json", f"{rel}/planogram_compliant.json", f"{rel}/catalogue.csv", True
        cat = "catalogue_scenario.csv" if (bay / "catalogue_scenario.csv").exists() else "catalogue.csv"
        yield f"{rel}/detection_scenario.json", f"{rel}/planogram_scenario.json", f"{rel}/{cat}", True


def main() -> None:
    passed = total = 0
    seconds = []
    for det_path, pg_path, cat_path, group in scenarios():
        catalogue = load_catalogue(ROOT / cat_path)
        report = compute_report(load_detection_output(ROOT / det_path), load_planogram(ROOT / pg_path), catalogue,
                                group_items=group)
        b = write_briefing(report, catalogue)
        if b.seconds:
            total += 1
            passed += b.summary_source == "slm"
            seconds.append(b.seconds)
        print(f"== {det_path} (score {report.score:.0f}, {len(report.violations)} violations, "
              f"{b.summary_source}, {b.seconds}s)\n{b.summary}")
        if b.rejected_summary:
            print(f"   REJECTED: {b.rejected_summary}")
    if total:
        print(f"\nmodel summaries accepted {passed}/{total}, mean {sum(seconds) / total:.1f}s, max {max(seconds):.1f}s")


if __name__ == "__main__":
    main()
