"""Shift briefing: task grouping, template summary and the check on the model's summary.
Only the last one uses the language model (skipped when it is not installed)."""

from pathlib import Path

import pytest

from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.briefing import check_summary, group_tasks, most_urgent, template_summary, write_briefing
from shelfsense.report.compliance import ViolationKind, compute_report

ROOT = Path(__file__).resolve().parents[1]


def mock_report():
    catalogue = load_catalogue(ROOT / "contracts/catalogue.example.csv")
    det = load_detection_output(ROOT / "data/mock/mock_shelf_002_lowstock.json")
    return compute_report(det, load_planogram(ROOT / "contracts/planogram.example.json"), catalogue), catalogue


def test_tasks_cover_every_violation_top_shelf_first():
    report, _ = mock_report()
    tasks = group_tasks(report)
    assert sum(len(t.actions) for t in tasks) == len(report.violations)
    shelves = [t.shelf_id for t in tasks]
    assert shelves == sorted(shelves)


def test_most_urgent_puts_stock_outs_first():
    report, _ = mock_report()
    assert most_urgent(report)[0].kind == ViolationKind.STOCK_OUT


def test_template_without_model():
    report, catalogue = mock_report()
    b = write_briefing(report, catalogue, model=None)
    assert b.summary_source == "template"
    assert b.summary == template_summary(report)
    assert "1 stock-out" in b.summary and "price mismatches" not in b.summary


def test_check_accepts_a_faithful_summary():
    report, catalogue = mock_report()
    first = most_urgent(report)[0]
    name = catalogue[first.sku].name
    text = f"The shelf scores {report.score:.0f}/100. Fix the stock-out of {name.lower()} at {first.slot_id} first."
    assert check_summary(text, report, catalogue)


def test_check_rejects_wrong_kind_numbers_other_slots_and_cut_off_text():
    report, catalogue = mock_report()
    first = most_urgent(report)[0]  # a stock-out
    assert not check_summary(f"{first.slot_id} is running low.", report, catalogue)
    assert not check_summary(f"Restock {first.slot_id} with 3 packs.", report, catalogue)
    assert not check_summary("Fix S9-09 first.", report, catalogue)
    assert not check_summary(f"Fix the stock-out at {first.slot_id} and then", report, catalogue)
    assert not check_summary(f"{first.slot_id} is compliant.", report, catalogue)


def test_two_sessions_asking_the_model_at_once_do_not_crash_the_server():
    # llama.cpp aborts the whole process (GGML_ASSERT) when two threads call it at once; briefing.py locks it.
    import threading

    pytest.importorskip("llama_cpp")
    from shelfsense.report.briefing import DEFAULT_MODEL

    if not DEFAULT_MODEL.exists():
        pytest.skip("no model file (python scripts/download_data.py slm)")
    bay = Path(__file__).resolve().parents[1] / "data" / "demo_bays" / "store4_9"
    cat = load_catalogue(bay / "catalogue_scenario.csv")
    rep = compute_report(load_detection_output(bay / "detection_scenario.json"), load_planogram(bay / "planogram_scenario.json"), cat, group_items=True)
    out: list[str] = []
    threads = [threading.Thread(target=lambda: out.append(write_briefing(rep, cat, model=DEFAULT_MODEL).summary_source)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 3 and set(out) <= {"slm", "template"}
