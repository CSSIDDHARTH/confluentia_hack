"""shelfSense dashboard: the shelf audit.

One flow: pick a real shelf and its planogram, optionally read the photo live (products, packs, empty
space and price labels), and get one report: a compliance score, the money at stake, the steps for staff,
and a shift-briefing summary written by a small language model and checked against the report. Mock shelves
and custom file paths are behind the "Developer inputs" switch.

    streamlit run app/dashboard.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from shelfsense.advisor.plan import plan as plan_steps
from shelfsense.contracts import (
    DetectionOutput,
    load_catalogue,
    load_detection_output,
    load_planogram,
)
from shelfsense.pricing.label_detector import DEFAULT_WEIGHTS, detect_labels
from shelfsense.pricing.price_check import PriceIssueType, check_prices
from shelfsense.report.briefing import DEFAULT_MODEL as SLM_MODEL
from shelfsense.report.briefing import write_briefing
from shelfsense.report.compliance import ComplianceReport, ViolationKind, compute_report
from shelfsense.report.export import report_to_json, violations_to_csv
from shelfsense.report.impact import at_stake, currency
from shelfsense.vision import live

ROOT = Path(__file__).resolve().parents[1]
BAYS = ROOT / "data" / "demo_bays"

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
    line = max(3, img.width // 500)
    flagged_label_ids = {issue.label_id for issue in price_issues}
    for it in det.items:
        draw.rectangle(it.bbox, outline=ITEM_COLOUR if it.sku else UNKNOWN_COLOUR, width=line)
    for gap in det.gaps:
        draw.rectangle(gap.bbox, outline=GAP_COLOUR, width=line + 1)
        draw.text((gap.bbox[0] + 8, gap.bbox[1] + 8), "EMPTY", fill=GAP_COLOUR)
    for lb in det.labels:
        flagged = lb.label_id in flagged_label_ids
        colour = LABEL_ISSUE_COLOUR if flagged else LABEL_COLOUR
        draw.rectangle(lb.bbox, outline=colour, width=line)
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
                "label_confidence": labels[0].ocr_confidence if labels else None,
                "catalogue_price": cat.price if cat else None,
            })
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Loading the detectors and the product index...")
def load_chain() -> live.LiveChain:
    return live.shared()  # the same models as the What to do page


@st.cache_data(show_spinner=False)
def cached_briefing(report_json: str, catalogue_path: str, use_model: bool):
    rep = ComplianceReport.model_validate_json(report_json)
    return write_briefing(rep, load_catalogue(ROOT / catalogue_path), model=SLM_MODEL if use_model else None)


st.set_page_config(page_title="shelfSense", layout="wide")
st.title("shelfSense: shelf audit")
st.caption(
    "One photo of a shelf bay, checked against its planogram and price list: right product, right place, right count, "
    "right price, and no empty space. Rules decide every violation; a small language model only writes the summary."
)

PLANS = {"compliant": "Matches the shelf", "scenario": "Store changed the plan (planted)"}
MOCK_PLANOGRAM = "contracts/planogram.example.json"
MOCK_CATALOGUE = "contracts/catalogue.example.csv"
# name -> (detection output, planogram, catalogue)
MOCKS = {
    "Mock shelf (as shipped)": ("contracts/detection_output.example.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Low-stock demo (all 5 violation types)": ("data/mock/mock_shelf_002_lowstock.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Near-clean shelf (one unreadable label)": ("data/mock/mock_shelf_003_clean.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
}


def bay_files(bay: str, plan: str) -> tuple[str, str, str]:
    rel = BAYS.relative_to(ROOT) / bay
    catalogue = "catalogue_scenario.csv" if plan == "scenario" and (BAYS / bay / "catalogue_scenario.csv").exists() else "catalogue.csv"
    return f"{rel}/detection_{plan}.json", f"{rel}/planogram_{plan}.json", f"{rel}/{catalogue}"


bays = sorted(p.name for p in BAYS.glob("*") if (p / "detection_scenario.json").exists()) if BAYS.exists() else []
bay = None
with st.sidebar:
    developer = st.toggle("Developer inputs", value=not bays, help="Mock shelves and custom file paths, for testing the report without photos.")
    if developer:
        preset = st.selectbox("Mock scenario", list(MOCKS))
        default_det, default_pg, default_cat = MOCKS[preset]
        with st.expander("Custom paths"):
            det_path = st.text_input("Detection output", "", placeholder=default_det) or default_det
            pg_path = st.text_input("Planogram", "", placeholder=default_pg) or default_pg
            cat_path = st.text_input("Catalogue", "", placeholder=default_cat) or default_cat
    else:
        st.header("Shelf")
        bay = st.selectbox("Real shelf (held-out Grozi photo)", bays, index=bays.index("store4_9") if "store4_9" in bays else 0)
        plan = st.radio("Planogram", list(PLANS), index=1, format_func=PLANS.get)
        det_path, pg_path, cat_path = bay_files(bay, plan)
        st.caption("The planted plan swaps two products, expects one the shelf does not have, raises one product's "
                   "facings by two, and raises one price by 10% in the price list.")

det = load_detection_output(ROOT / det_path)
planogram = load_planogram(ROOT / pg_path)
catalogue = load_catalogue(ROOT / cat_path)

if bay is not None:
    photo, is_original = live.bay_photo(BAYS / bay)
    live_key = f"audit-live-{bay}-{plan}"
    with st.container(border=True):
        st.markdown("**Read the photo now**")
        with_prices = live.prices_available()
        st.caption(
            "Finds the product blocks, names each one, lines them up with the plan, counts packs, finds empty spaces"
            + (", and finds and reads the price labels" if with_prices else "")
            + ", from the photo, on this machine. Without it the page shows the saved reading of the same photo."
        )
        if not live.available():
            st.info("The live chain needs the Grozi product index (`python scripts/build_sku_index.py`) and the three detectors in `models/`; showing saved results.")
        else:
            if not with_prices:
                st.caption("Prices are not read live here: that needs `models/label_det/weights/best.pt` and `tesseract` (`sudo apt install tesseract-ocr`).")
            if st.button("Run it live", type="primary", key=f"run-{live_key}"):
                with st.spinner("Reading the photo..."):
                    st.session_state[live_key] = load_chain().run(
                        photo, planogram, read_prices=with_prices,
                        shown_path=str(photo.relative_to(ROOT)) if photo.is_relative_to(ROOT) else photo.name,
                    )
    result = st.session_state.get(live_key)
    if result is not None:
        det = result.det
        st.success(
            f"Live run on {result.device}: {result.total:.2f} s in total ("
            + ", ".join(f"{k} {v:.2f} s" for k, v in result.timings.items()) + "). Everything below comes from this run."
        )

if det.planogram_id != planogram.planogram_id:
    st.warning(f"Image is for planogram {det.planogram_id}, but {planogram.planogram_id} is loaded.")

price_issues = check_prices(det, catalogue)
report = compute_report(det, planogram, catalogue, group_items=True)
steps = plan_steps(report, catalogue, planogram)
stake = at_stake(report, det, planogram, catalogue)
money = currency(catalogue)
firm = report.counts.get(ViolationKind.PRICE_MISMATCH.value, 0)
to_check = report.counts.get(ViolationKind.UNREADABLE_PRICE.value, 0)

score_col, c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1, 1])
score_col.metric("Compliance score", f"{report.score:.0f} / 100")
c1.metric(
    "Value at stake (estimate)", f"{money} {sum(stake.values()):.2f}".strip(),
    help="Missing packs (planned facings minus matching packs seen) times their price, plus each firm price "
    "mismatch's price gap times the slot's planned facings. No sales rates: an estimate for putting work in order.",
)
c2.metric("Steps for staff", len(steps))
c3.metric("Empty spaces", len(det.gaps))
c4.metric("Price mismatches", firm, help=f"Firm mismatches only. {to_check} more label(s) to check by eye: not read, or read with OCR confidence below 0.5.")

left, right = st.columns([3, 2])
with left:
    st.image(
        overlay(det, price_issues),
        caption=(
            f"{det.image_id}: green = product, orange = unrecognised, "
            "red = empty space or price issue, blue = price label"
        ),
    )
with right:
    st.subheader("What to do, most severe first")
    if not steps:
        st.success("Nothing to fix: the shelf matches the plan.")
    for k, step in enumerate(steps, 1):
        value = sum(stake.get(v, 0.0) for v in step.violation_ids)
        st.markdown(f"**{k}.** {step.text}" + (f"  \n:red[{money} {value:.2f} at stake]" if value else ""))

st.subheader("Shift briefing")
use_slm = st.checkbox(
    "Write the summary with the small language model (Qwen2.5-1.5B, 4-bit, on CPU)",
    value=SLM_MODEL.exists() and os.environ.get("SHELFSENSE_SLM") != "0",  # tests set 0 to skip the model
    disabled=not SLM_MODEL.exists(),
    help="The steps are always built in code. The model only writes the summary, and a summary that "
    "names a wrong slot, problem or number is replaced by the template.",
)
with st.spinner("Writing the summary..."):
    briefing = cached_briefing(report.model_dump_json(), cat_path, use_slm)
st.info(briefing.summary)
if briefing.summary_source == "slm":
    st.caption(f"Summary written by the small language model in {briefing.seconds:.1f}s and checked against the report.")
elif briefing.rejected_summary:
    st.caption(f"The model's summary failed the check, so the template is shown. Rejected: \"{briefing.rejected_summary}\"")
with st.expander("The same work grouped by shelf, top shelf first"):
    for tasks in briefing.tasks:
        st.markdown(f"**Shelf {tasks.shelf_id or '(labels not on a slot)'}**")
        st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(tasks.actions, 1)))

with st.expander("Plan vs. shelf, per slot"):
    st.dataframe(slot_table(det, planogram, catalogue), hide_index=True, width="stretch")

with st.expander(f"Price-label checks ({len(price_issues)})"):
    if price_issues:
        st.dataframe(
            pd.DataFrame([
                {
                    "label": issue.label_id,
                    "slot": issue.slot_id,
                    "sku": issue.sku,
                    "issue": ISSUE_LABELS[issue.issue_type] + (" (check by eye)" if issue.low_confidence else ""),
                    "ocr_text": issue.ocr_text,
                    "label_price": issue.label_price,
                    "catalogue_price": issue.catalogue_price,
                    "difference": issue.difference,
                    "detail": issue.detail,
                }
                for issue in price_issues
            ]),
            hide_index=True, width="stretch",
        )
    else:
        st.success("No price-label issues found.")

with st.expander(f"All {len(report.violations)} violation lines and actions"):
    if report.violations:
        st.dataframe(
            pd.DataFrame([
                {
                    "kind": v.kind.value.replace("_", " "),
                    "shelf": v.shelf_id,
                    "slot": v.slot_id,
                    "sku": v.sku,
                    "at_stake": stake.get(v.violation_id),
                    "detail": v.detail,
                    "recommended_action": v.recommended_action,
                }
                for v in report.violations
            ]),
            hide_index=True, width="stretch",
        )
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

with st.expander("Read the price labels on any shelf photo"):
    st.caption(
        "Runs the label detector and OCR on an uploaded photo. With no planogram the labels are not tied to slots, "
        "so there is no catalogue check here; use a shelf above for the full audit."
    )
    if not DEFAULT_WEIGHTS.exists():
        st.info(f"Label detector weights not found at {DEFAULT_WEIGHTS.relative_to(ROOT)}; train it first.")
    else:
        upload = st.file_uploader("Shelf photo", type=["jpg", "jpeg", "png"])
        if upload is not None:
            with st.spinner("Detecting labels and reading prices..."):
                shelf_img = Image.open(upload).convert("RGB")
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
