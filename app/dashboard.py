"""shelfSense dashboard (mock stage).

Reads the three contracts, shows them joined per slot, and runs the
price-check and compliance-report modules on top: price mismatches, a
compliance score, and a violations/recommended-actions table.

    streamlit run app/dashboard.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from shelfsense.contracts import (
    DetectionOutput,
    load_catalogue,
    load_detection_output,
    load_planogram,
)
from shelfsense.pricing.label_detector import DEFAULT_WEIGHTS, detect_labels
from shelfsense.pricing.price_check import PriceIssueType, check_prices
from shelfsense.report.compliance import compute_report
from shelfsense.report.export import report_to_json, violations_to_csv

ROOT = Path(__file__).resolve().parents[1]

ITEM_COLOUR = "#2e9e44"
UNKNOWN_COLOUR = "#f0a020"
GAP_COLOUR = "#d62828"
LABEL_COLOUR = "#1d6fd6"
LABEL_ISSUE_COLOUR = "#d62828"

ISSUE_LABELS = {
    PriceIssueType.PRICE_MISMATCH: "price mismatch",
    PriceIssueType.UNREADABLE_PRICE: "unreadable price",
    PriceIssueType.UNKNOWN_PRODUCT: "unknown product",
}


def overlay(det: DetectionOutput, price_issues: list) -> Image.Image:
    path = ROOT / det.image_path
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (det.image_width, det.image_height), "white")
    draw = ImageDraw.Draw(img)
    flagged_label_ids = {issue.label_id for issue in price_issues}
    for it in det.items:
        draw.rectangle(it.bbox, outline=ITEM_COLOUR if it.sku else UNKNOWN_COLOUR, width=4)
    for gap in det.gaps:
        draw.rectangle(gap.bbox, outline=GAP_COLOUR, width=5)
        draw.text((gap.bbox[0] + 8, gap.bbox[1] + 8), "EMPTY", fill=GAP_COLOUR)
    for lb in det.labels:
        flagged = lb.label_id in flagged_label_ids
        colour = LABEL_ISSUE_COLOUR if flagged else LABEL_COLOUR
        draw.rectangle(lb.bbox, outline=colour, width=3)
        if flagged:
            draw.text((lb.bbox[0], lb.bbox[1] - 14), "PRICE?", fill=LABEL_ISSUE_COLOUR)
    return img


def slot_table(det, planogram, catalogue) -> pd.DataFrame:
    rows = []
    for shelf in planogram.shelves:
        for slot in shelf.slots:
            in_slot = [i for i in det.items if i.slot_id == slot.slot_id]
            labels = [lb for lb in det.labels if lb.slot_id == slot.slot_id]
            cat = catalogue.get(slot.sku)
            rows.append({
                "slot": slot.slot_id,
                "expected_sku": slot.sku,
                "product": cat.name if cat else "?",
                "expected_facings": slot.facings,
                "matching_facings": sum(i.sku == slot.sku for i in in_slot),
                "other_skus": ", ".join(sorted({i.sku or "unknown" for i in in_slot if i.sku != slot.sku})),
                "gap_facings": sum(g.estimated_missing_facings for g in det.gaps if g.slot_id == slot.slot_id),
                "label_price": labels[0].price if labels else None,
                "catalogue_price": cat.price if cat else None,
            })
    return pd.DataFrame(rows)


st.set_page_config(page_title="shelfSense", layout="wide")
st.title("shelfSense: shelf audit")

MOCK_PLANOGRAM = "contracts/planogram.example.json"
MOCK_CATALOGUE = "contracts/catalogue.example.csv"

# name -> (detection output, planogram, catalogue)
PRESETS = {
    "Mock shelf (as shipped)": ("contracts/detection_output.example.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Low-stock demo (all 5 violation types)": ("data/mock/mock_shelf_002_lowstock.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Near-clean shelf (one unreadable label)": ("data/mock/mock_shelf_003_clean.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
}
for bay_dir in sorted((ROOT / "data" / "demo_bays").glob("*/detection_compliant.json")):
    bay = bay_dir.parent
    rel = bay.relative_to(ROOT)
    PRESETS[f"Real bay {bay.name}: compliant plan"] = (
        f"{rel}/detection_compliant.json", f"{rel}/planogram_compliant.json", f"{rel}/catalogue.csv")
    scenario_catalogue = "catalogue_scenario.csv" if (bay / "catalogue_scenario.csv").exists() else "catalogue.csv"
    PRESETS[f"Real bay {bay.name}: planted changes"] = (
        f"{rel}/detection_scenario.json", f"{rel}/planogram_scenario.json", f"{rel}/{scenario_catalogue}")

with st.sidebar:
    preset = st.selectbox("Scenario", list(PRESETS))
    default_det, default_pg, default_cat = PRESETS[preset]
    with st.expander("Advanced: custom paths"):
        det_path = st.text_input("Detection output", "", placeholder=default_det)
        pg_path = st.text_input("Planogram", "", placeholder=default_pg)
        cat_path = st.text_input("Catalogue", "", placeholder=default_cat)
    det_path = det_path or default_det
    pg_path = pg_path or default_pg
    cat_path = cat_path or default_cat

det = load_detection_output(ROOT / det_path)
planogram = load_planogram(ROOT / pg_path)
catalogue = load_catalogue(ROOT / cat_path)

if det.planogram_id != planogram.planogram_id:
    st.warning(f"Image is for planogram {det.planogram_id}, but {planogram.planogram_id} is loaded.")

price_issues = check_prices(det, catalogue)
report = compute_report(det, planogram, catalogue, group_items=preset.startswith("Real bay"))

score_col, c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1, 1])
score_col.metric("Compliance score", f"{report.score:.0f} / 100")
c1.metric("Items detected", len(det.items))
c2.metric("Unrecognised", sum(i.sku is None for i in det.items))
c3.metric("Gaps", len(det.gaps))
c4.metric("Labels read", len(det.labels))
c5.metric("Price issues", len(price_issues))

left, right = st.columns([3, 2])
with left:
    st.image(
        overlay(det, price_issues),
        caption=(
            f"{det.image_id}: green = product, orange = unrecognised, "
            "red = empty/price issue, blue = label"
        ),
    )
with right:
    st.subheader("Planogram vs. detected, per slot")
    st.dataframe(slot_table(det, planogram, catalogue), hide_index=True, width="stretch")

st.subheader("Price-label checks")
if price_issues:
    issue_rows = pd.DataFrame(
        [
            {
                "label": issue.label_id,
                "slot": issue.slot_id,
                "sku": issue.sku,
                "issue": ISSUE_LABELS[issue.issue_type],
                "verify": issue.low_confidence,
                "ocr_text": issue.ocr_text,
                "label_price": issue.label_price,
                "catalogue_price": issue.catalogue_price,
                "difference": issue.difference,
                "detail": issue.detail,
            }
            for issue in price_issues
        ]
    )
    st.dataframe(issue_rows, hide_index=True, width="stretch")
else:
    st.success("No price-label issues found.")

st.subheader(f"Violations & recommended actions ({len(report.violations)})")
if report.violations:
    violation_rows = pd.DataFrame(
        [
            {
                "kind": v.kind.value.replace("_", " "),
                "shelf": v.shelf_id,
                "slot": v.slot_id,
                "sku": v.sku,
                "detail": v.detail,
                "recommended_action": v.recommended_action,
            }
            for v in report.violations
        ]
    )
    st.dataframe(violation_rows, hide_index=True, width="stretch")
else:
    st.success("No violations found; shelf is fully compliant.")

dl1, dl2 = st.columns(2)
dl1.download_button(
    "Download report (JSON)",
    data=report_to_json(report),
    file_name=f"{det.image_id}_compliance_report.json",
    mime="application/json",
)
dl2.download_button(
    "Download violations (CSV)",
    data=violations_to_csv(report),
    file_name=f"{det.image_id}_violations.csv",
    mime="text/csv",
)

st.subheader("Find price labels in a shelf photo")
st.caption(
    "Runs the trained label detector and OCR on a real photo. Labels are not yet "
    "tied to slots or SKUs (that needs the vision stage), so there is no catalogue check here."
)
if not DEFAULT_WEIGHTS.exists():
    st.info(f"Label detector weights not found at {DEFAULT_WEIGHTS.relative_to(ROOT)}; train it first.")
else:
    photo = st.file_uploader("Shelf photo", type=["jpg", "jpeg", "png"])
    if photo is not None:
        with st.spinner("Detecting labels and reading prices..."):
            shelf_img = Image.open(photo).convert("RGB")
            found = detect_labels(shelf_img)
        shown = shelf_img.copy()
        draw = ImageDraw.Draw(shown)
        line = max(3, shown.width // 300)
        for lb in found:
            draw.rectangle(lb.bbox, outline=LABEL_COLOUR if lb.price is not None else LABEL_ISSUE_COLOUR, width=line)
        pl, pr = st.columns([3, 2])
        pl.image(shown, caption=f"{len(found)} labels detected (blue = price read, red = unreadable)")
        pr.dataframe(
            pd.DataFrame(
                [
                    {"label": lb.label_id, "price": lb.price, "confidence": lb.ocr_confidence, "ocr_text": lb.ocr_text}
                    for lb in found
                ]
            ),
            hide_index=True,
            width="stretch",
        )

with st.expander("Raw detection output"):
    st.json(det.model_dump())
