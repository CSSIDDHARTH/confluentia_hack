"""shelfSense dashboard: the whole app on one page.

"A shelf and its plan": pick a real shelf and its planogram, read the photo live (products, packs, empty
space and price labels), and get one report: a compliance score, the money at stake, the steps for staff,
and a shift-briefing summary written by a small language model and checked against the report. "Edit the
plan yourself" changes the plan on the same reading, so anyone can plant a violation. Mock shelves and
custom file paths are behind the "Developer inputs" switch.

"Just a photo": empty space, row fill and a refill list for any shelf photo, no planogram (app/views/availability.py).

    streamlit run app/dashboard.py
"""

import hashlib
import html
import os
import runpy
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from shelfsense.advisor.edit import apply_rows, changes, plan_rows, swap_products
from shelfsense.advisor.plan import plan as plan_steps
from shelfsense.contracts import (
    DetectionOutput,
    load_catalogue,
    load_detection_output,
    load_planogram,
)
from shelfsense.pricing.label_assign import assign_labels_to_slots
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


# slot status -> (background, label), worst first; the text on every background is white
SLOT_STATUS = {
    ViolationKind.STOCK_OUT: ("#b3261e", "missing"),
    ViolationKind.MISPLACED: ("#b3261e", "wrong product here"),
    ViolationKind.LOW_STOCK: ("#b86e00", "low stock"),
    ViolationKind.UNRECOGNISED_ITEM: ("#7a6a00", "check product"),
}
SLOT_OK = ("#2e7d32", "as planned")
SLOT_UNREAD = ("#5f6368", "not read yet")


def plan_html(planogram, catalogue, det: DetectionOutput | None, report: ComplianceReport | None) -> str:
    """The plan drawn shelf by shelf, top first: one box per slot, as wide as its planned facings, coloured by
    what the photo shows there (no report: grey). Price problems get a note on the slot."""
    kinds: dict[str, set] = {}
    for v in report.violations if report else []:
        kinds.setdefault(v.slot_id, set()).add(v.kind)
    rows = []
    for shelf in planogram.shelves:
        cells = []
        for slot in shelf.slots:
            item = catalogue.get(slot.sku)
            name = html.escape(item.name if item else slot.sku)
            here = kinds.get(slot.slot_id, set())
            colour, status = SLOT_UNREAD if report is None else next(
                (SLOT_STATUS[k] for k in SLOT_STATUS if k in here), SLOT_OK
            )
            seen = "" if det is None or report is None else (
                f" · seen {sum(i.slot_id == slot.slot_id and i.sku == slot.sku for i in det.items)}"
            )
            price = ""
            if ViolationKind.PRICE_MISMATCH in here:
                price = "<br>price label wrong"
            elif ViolationKind.UNREADABLE_PRICE in here:
                price = "<br>check price label"
            cells.append(
                f'<div style="flex:{slot.facings} 1 0;min-width:0;background:{colour};color:#fff;border-radius:6px;'
                f'padding:6px 8px;margin:2px;font-size:0.8rem;line-height:1.25;overflow:hidden">'
                f"<b>{slot.slot_id}</b> {name}<br>planned {slot.facings}{seen}<br><i>{status}</i>{price}</div>"
            )
        rows.append(
            f'<div style="display:flex;align-items:stretch;margin-bottom:6px;padding-bottom:6px;'
            f'border-bottom:3px solid #9aa0a6">{"".join(cells)}</div>'
        )
    return "".join(rows)


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
    return live.shared()


@st.cache_data(show_spinner=False)
def cached_briefing(report_json: str, catalogue_path: str, use_model: bool):
    rep = ComplianceReport.model_validate_json(report_json)
    return write_briefing(rep, load_catalogue(ROOT / catalogue_path), model=SLM_MODEL if use_model else None)


st.set_page_config(page_title="shelfSense", layout="wide")
st.title("shelfSense: shelf audit")
if "show-saved" in st.session_state:  # keep the switch's value on runs where it is not drawn (edit mode)
    st.session_state["show-saved"] = st.session_state["show-saved"]

MODES = ["A shelf and its plan", "Just a photo"]
with st.sidebar:
    mode = st.radio("What do you have?", MODES, key="mode", help="With a planogram: placement, facings, prices and empty "
                    "space against the plan. Just a photo: empty space and what to refill, on any shop photo.")
if mode == MODES[1]:
    runpy.run_path(str(ROOT / "app" / "views" / "availability.py"))
    st.stop()

st.caption(
    "One photo of a shelf bay, checked against its planogram and price list: right product, right place, right count, "
    "right price, and no empty space. Rules decide every violation; a small language model only writes the summary."
)

EDIT = "edit"
PLANS = {"compliant": "Matches the shelf", "scenario": "Store changed the plan (planted)", EDIT: "Edit the plan yourself"}
MOCK_PLANOGRAM = "contracts/planogram.example.json"
MOCK_CATALOGUE = "contracts/catalogue.example.csv"
# name -> (detection output, planogram, catalogue)
MOCKS = {
    "Mock shelf (as shipped)": ("contracts/detection_output.example.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Low-stock demo (all 5 violation types)": ("data/mock/mock_shelf_002_lowstock.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Near-clean shelf (one unreadable label)": ("data/mock/mock_shelf_003_clean.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
}


def bay_files(bay: str, plan: str) -> tuple[str, str, str]:
    """Detection, planogram and catalogue for a saved plan; `edit` starts from the compliant plan on the same reading."""
    rel = BAYS.relative_to(ROOT) / bay
    plan = "compliant" if plan == EDIT else plan
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
                   "facings by two, and raises one price by 10% in the price list. Or edit the plan yourself.")

det = load_detection_output(ROOT / det_path)
planogram = load_planogram(ROOT / pg_path)
catalogue = load_catalogue(ROOT / cat_path)
base_score = None

if bay is not None and plan == EDIT:
    shelf_plan = planogram  # the plan that matches the photo
    rows_key, ver_key = f"edit-rows-{bay}", f"edit-ver-{bay}"
    if rows_key not in st.session_state:
        st.session_state[rows_key], st.session_state[ver_key] = plan_rows(shelf_plan), 0
    original = plan_rows(shelf_plan)
    names = {sku: f"{c.name} ({sku})" for sku, c in sorted(catalogue.items())}
    to_sku = {label: sku for sku, label in names.items()}
    with st.container(border=True):
        st.markdown("**Edit the plan**")
        st.caption(
            "The photo has been read once and is fixed. Change what a slot **expects** and the score moves because the shelf no longer "
            "matches. Raise *facings planned* and *minimum* together to simulate a promotion reset. Low stock means fewer packs than the minimum."
        )
        table = pd.DataFrame(st.session_state[rows_key])
        table["product"] = table["sku"].map(names).fillna(table["sku"])
        editor = st.data_editor(
            table[["slot", "product", "facings", "minimum"]], hide_index=True, width="stretch", key=f"editor-{bay}-{st.session_state[ver_key]}",
            disabled=["slot"],
            column_config={
                "slot": st.column_config.TextColumn("Slot"),
                "product": st.column_config.SelectboxColumn("Expected product", options=list(names.values()), required=True),
                "facings": st.column_config.NumberColumn("Facings planned", min_value=1, max_value=30, step=1),
                "minimum": st.column_config.NumberColumn("Minimum (low stock below)", min_value=1, max_value=30, step=1),
            },
        )
        rows = [
            {"slot": r.slot, "sku": to_sku.get(r.product, r.product), "facings": int(r.facings), "minimum": int(r.minimum)}
            for r in editor.itertuples()
        ]
        slots = [r["slot"] for r in rows]
        c1, c2, c3, c4 = st.columns(4)
        a = c1.selectbox("Swap this slot", slots, index=0, key=f"swap-a-{bay}")
        b = c2.selectbox("with this slot", slots, index=min(1, len(slots) - 1), key=f"swap-b-{bay}")
        if c3.button("Swap their products", disabled=a == b):
            st.session_state[rows_key] = swap_products(rows, a, b)
            st.session_state[ver_key] += 1
            st.rerun()
        if c4.button("Reset to the shelf's plan"):
            st.session_state[rows_key] = original
            st.session_state[ver_key] += 1
            st.rerun()
        planogram = apply_rows(shelf_plan, rows)
        edits = changes(original, rows)
        st.caption(f"{len(edits)} change{'s' if len(edits) != 1 else ''} to the plan" + (": " + "; ".join(edits) if edits else "."))
    # labels sit under the same slots, which now expect the edited products (a swap leaves each label under the other product)
    det = det.model_copy(update={"labels": assign_labels_to_slots(det.labels, det, planogram)})
    base_score = compute_report(det.model_copy(update={"labels": assign_labels_to_slots(det.labels, det, shelf_plan)}),
                                shelf_plan, catalogue, group_items=True).score

if bay is not None:
    photo, is_original = live.bay_photo(BAYS / bay)
    live_key = f"audit-live-{bay}-{plan}-{hashlib.md5(planogram.model_dump_json().encode()).hexdigest()[:10]}"
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
    elif plan != EDIT and live.available() and not st.toggle(
        "Show the saved reading", key="show-saved",
        help="The result saved from an earlier run on the same photo, which the live run reproduces exactly.",
    ):
        # Nothing is read yet: the plain photo, so the live run is what draws the findings.
        left, right = st.columns([3, 2])
        left.image(str(BAYS / bay / "photo.jpg"), caption=f"{bay}: not read yet. Press Run it live.")
        with right:
            st.subheader("The plan")
            st.markdown(plan_html(planogram, catalogue, None, None), unsafe_allow_html=True)
            st.caption("What this shelf should hold, top shelf first; each box is as wide as its planned facings.")
        st.stop()

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
score_col.metric(
    "Compliance score", f"{report.score:.0f} / 100",
    delta=None if base_score is None else f"{report.score - base_score:+.0f} vs the real plan",
)
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
    st.subheader("The plan")
    st.markdown(plan_html(planogram, catalogue, det, report), unsafe_allow_html=True)
    st.caption("Top shelf first; each box is as wide as its planned facings. Green = as planned, orange = low stock, "
               "red = missing or the wrong product, olive = product not recognised.")

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
