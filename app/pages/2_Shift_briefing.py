"""shelfSense: what to do.

The planogram check on a demo shelf, with its violation lines merged into a few ordered steps: two
products sitting in each other's slots become one "swap", a wrong product in a slot that lacks the right
one becomes one "replace". Rules do the merging (shelfsense.advisor.plan), so it is exact.

Part of the same app as dashboard.py (Streamlit multipage): `streamlit run app/dashboard.py`.
"""

from pathlib import Path

import streamlit as st

from shelfsense.advisor.plan import plan as plan_steps
from shelfsense.contracts import load_catalogue, load_detection_output, load_planogram
from shelfsense.report.compliance import compute_report

ROOT = Path(__file__).resolve().parents[2]
BAYS = ROOT / "data" / "demo_bays"


@st.cache_data
def report_for(bay: str, plan: str):
    d = BAYS / bay
    cat = load_catalogue(d / "catalogue.csv")
    report = compute_report(
        load_detection_output(d / f"detection_{plan}.json"), load_planogram(d / f"planogram_{plan}.json"), cat, group_items=True,
    )
    return report, cat


st.set_page_config(page_title="shelfSense: what to do", layout="wide")
st.title("What to do")
st.caption(
    "The planogram check lists every violation, so one product swap shows up as four lines. Rules merge related "
    "violations into steps and put the most severe first, so a staff member gets a short list."
)

bays = sorted(p.name for p in BAYS.iterdir() if (p / "detection_scenario.json").exists())
if not bays:
    st.error("No demo shelves found in data/demo_bays/.")
    st.stop()

with st.sidebar:
    st.header("Shelf")
    bay = st.selectbox("Demo shelf", bays, index=bays.index("store1_47") if "store1_47" in bays else 0)
    plan = st.radio(
        "Planogram", ["compliant", "scenario"], index=1,
        format_func=lambda k: "Matches the shelf" if k == "compliant" else "Store changed the plan (planted)",
    )

report, catalogue = report_for(bay, plan)
steps = plan_steps(report, catalogue)
c1, c2, c3 = st.columns(3)
c1.metric("Compliance score", f"{report.score:.0f} / 100")
c2.metric("Violation lines", len(report.violations))
c3.metric("Steps for staff", len(steps))

left, right = st.columns([3, 2])
with left:
    overlay = BAYS / bay / f"overlay_{plan}.jpg"
    if overlay.exists():
        st.image(str(overlay), caption=f"{bay}: detected blocks matched to the planogram")
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
