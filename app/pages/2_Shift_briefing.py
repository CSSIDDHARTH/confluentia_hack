"""shelfSense: what to do.

The planogram check on a demo shelf, with its violation lines merged into a few ordered steps: two
products sitting in each other's slots become one "swap", a wrong product in a slot that lacks the right
one becomes one "replace", and each misplaced product says where it belongs. Rules do the merging
(shelfsense.advisor.plan), so it is exact.

Choose "Edit the plan yourself" to be the one who plants the violation: change what a slot expects, or swap
two slots, and the score and steps update. The photo reading stays fixed; only the plan changes.

Part of the same app as dashboard.py (Streamlit multipage): `streamlit run app/dashboard.py`.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from shelfsense.advisor.edit import apply_rows, changes, plan_rows, swap_products
from shelfsense.advisor.plan import plan as plan_steps
from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.compliance import compute_report

ROOT = Path(__file__).resolve().parents[2]
BAYS = ROOT / "data" / "demo_bays"
EDIT = "edit"


@st.cache_data
def load_bay(bay: str, plan: str):
    """(detection, planogram, catalogue) for a saved plan; `edit` starts from the compliant plan on the same reading."""
    d = BAYS / bay
    kind = "compliant" if plan == EDIT else plan
    return (
        load_detection_output(d / f"detection_{kind}.json"), load_planogram(d / f"planogram_{kind}.json"),
        load_catalogue(d / "catalogue.csv"),
    )


st.set_page_config(page_title="shelfSense: what to do", layout="wide")
st.title("What to do")
st.caption(
    "The planogram check lists every violation, so one product swap shows up as four lines. Rules merge related "
    "violations into steps, say where a misplaced product belongs, and put the most severe first."
)

bays = sorted(p.name for p in BAYS.iterdir() if (p / "detection_scenario.json").exists())
if not bays:
    st.error("No demo shelves found in data/demo_bays/.")
    st.stop()

LABELS = {
    "compliant": "Matches the shelf",
    "scenario": "Store changed the plan (planted)",
    EDIT: "Edit the plan yourself",
}
with st.sidebar:
    st.header("Shelf")
    bay = st.selectbox("Demo shelf", bays, index=bays.index("store1_47") if "store1_47" in bays else 0)
    plan = st.radio("Planogram", list(LABELS), index=1, format_func=LABELS.get)

det, planogram, catalogue = load_bay(bay, plan)
base_score = None

if plan == EDIT:
    shelf_plan = planogram  # the plan that matches the photo
    rows_key, ver_key = f"edit-rows-{bay}", f"edit-ver-{bay}"
    if rows_key not in st.session_state:
        st.session_state[rows_key], st.session_state[ver_key] = plan_rows(shelf_plan), 0
    original = plan_rows(shelf_plan)
    base_score = compute_report(det, shelf_plan, catalogue, group_items=True).score
    names = {sku: f"{c.name} ({sku})" for sku, c in sorted(catalogue.items())}
    to_sku = {label: sku for sku, label in names.items()}

    st.subheader("Edit the plan")
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
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
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

report = compute_report(det, planogram, catalogue, group_items=True)
steps = plan_steps(report, catalogue, planogram)
c1, c2, c3 = st.columns(3)
c1.metric("Compliance score", f"{report.score:.0f} / 100", delta=None if base_score is None else f"{report.score - base_score:+.0f} vs the real plan")
c2.metric("Violation lines", len(report.violations))
c3.metric("Steps for staff", len(steps))

left, right = st.columns([3, 2])
with left:
    overlay = BAYS / bay / f"overlay_{'compliant' if plan == EDIT else plan}.jpg"
    if overlay.exists():
        st.image(str(overlay), caption=f"{bay}: detected blocks and the slot each sits in" + (" (the photo reading does not change when you edit the plan)" if plan == EDIT else ""))
with right:
    st.subheader("Steps, most severe first")
    if not steps:
        st.success("Nothing to fix: the shelf matches the plan.")
    for k, step in enumerate(steps, 1):
        st.markdown(f"**{k}.** {step.text}")
        if len(step.violation_ids) > 1:
            st.caption(f"replaces {len(step.violation_ids)} violation lines")
    with st.expander(f"The {len(report.violations)} violation lines these steps replace"):
        for v in report.violations:
            st.text(v.detail)
