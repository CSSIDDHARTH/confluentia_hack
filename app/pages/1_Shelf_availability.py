"""shelfSense: shelf availability (wide-shelf mode).

Empty spaces, per-row fill and a replenishment list for any shelf photo, with no planogram.
Works from saved pack detections (no GPU needed) or, when the detector is installed, from an
uploaded photo. The two sliders recompute from the saved packs instantly.

Part of the same app as dashboard.py (Streamlit multipage): `streamlit run app/dashboard.py`.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from shelfsense.contracts import load_detection_output
from shelfsense.vision.availability import analyze_packs
from shelfsense.vision.gaps import LOW_FILL_RATIO, MIN_GAP_WIDTHS
from shelfsense.vision.render import draw_availability

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "availability_demo"
WEIGHTS = ROOT / "models" / "pack_detector.pt"


def detector_available() -> bool:
    if not WEIGHTS.exists():
        return False
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


@st.cache_resource(show_spinner="Loading the pack detector...")
def load_detector():
    from shelfsense.vision.detect import ProductDetector

    return ProductDetector(WEIGHTS, conf=0.25, imgsz=1024, max_det=600)


@st.cache_data(show_spinner="Finding packs...")
def detect_upload(data: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    boxes, confs, size = load_detector().boxes(path)
    return boxes, confs, size


st.set_page_config(page_title="shelfSense: availability", layout="wide")
st.title("Shelf availability")
st.caption(
    "Empty spaces, low stock and what to refill, from one photo and no planogram. "
    "Packs come from a detector trained on SKU-110K; empty space is any stretch of a shelf row "
    "that no pack covers but one could fit. Gap accuracy is judged by eye, not measured "
    "(CLAUDE.md, assumption 12)."
)

live = detector_available()
saved = sorted(p.stem for p in DEMO.glob("*.json") if not p.name.endswith(".report.json"))

with st.sidebar:
    st.header("Shelf photo")
    sources = ["Saved test shelves"] + (["Upload a photo (runs the detector)"] if live else [])
    source = st.radio("Source", sources, label_visibility="collapsed")
    if not live:
        st.caption("Live detection needs `ultralytics` and `models/pack_detector.pt`; showing saved shelves only.")
    st.header("Sensitivity")
    gap_widths = st.slider(
        "Smallest empty space, in pack widths", 0.4, 2.0, MIN_GAP_WIDTHS, 0.1,
        help="A stretch of shelf this many median pack widths wide, with no pack in it, counts as empty.",
    )
    low_fill = st.slider(
        "Low stock below this fill", 0.2, 0.95, LOW_FILL_RATIO, 0.05, format="%.2f",
        help="Share of the bay's width covered by packs. A row below it needs replenishing.",
    )

boxes = confs = size = image = None
image_id = shown = ""
if source == "Saved test shelves":
    if not saved:
        st.error(f"No saved results in {DEMO.relative_to(ROOT)}. Run: python scripts/run_availability.py --images data/test_images --out data/availability_demo")
        st.stop()
    with st.sidebar:
        image_id = st.selectbox("Shelf", saved)
    saved_det = load_detection_output(DEMO / f"{image_id}.json")
    boxes = [i.bbox for i in saved_det.items]
    confs = [i.det_confidence for i in saved_det.items]
    size = (saved_det.image_width, saved_det.image_height)
    shown = saved_det.image_path
    image = Image.open(ROOT / shown)
else:
    up = st.file_uploader("A shelf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Upload a shelf photo to find its empty spaces.")
        st.stop()
    boxes, confs, size = detect_upload(up.getvalue(), Path(up.name).suffix or ".jpg")
    image_id, shown = Path(up.name).stem, up.name
    image = Image.open(up)

det, report = analyze_packs(image_id, shown, size, boxes, confs, low_fill=low_fill, min_gap_widths=gap_widths)

low_rows = [r.shelf_id for r in report.rows if r.low_stock]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall fill", f"{report.overall_fill:.0%}")
c2.metric("Empty spaces", report.empty_spaces)
c3.metric("Packs to refill", report.missing_facings)
c4.metric("Low-stock rows", len(low_rows), help=", ".join(low_rows) or "none")
c5.metric("Packs found", len(det.items))

for w in report.warnings:
    st.warning(w)

left, right = st.columns([3, 2])
with left:
    view = draw_availability(image, det, report, footer=False)
    view.thumbnail((1400, 1800))
    st.image(view, caption=f"{image_id}: green = pack, red = empty space, label = row fill")
with right:
    st.subheader("What to do")
    if report.actions:
        for a in report.actions:
            st.markdown(f"- {a}")
    elif not report.warnings:
        st.success("No replenishment needed at these settings.")
    st.subheader("Shelf rows")
    rows = pd.DataFrame(
        [{"shelf": r.shelf_id, "packs": r.packs, "fill": r.fill_ratio, "status": "LOW STOCK" if r.low_stock else "ok"} for r in report.rows]
    )
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_config={"fill": st.column_config.ProgressColumn("fill", format="percent", min_value=0.0, max_value=1.0)},
    )

if det.gaps:
    st.subheader(f"Empty spaces ({len(det.gaps)})")
    gaps = pd.DataFrame(
        [{"shelf": g.shelf_id, "about (packs)": g.estimated_missing_facings, "confidence": g.confidence, "bbox": [round(v) for v in g.bbox]} for g in det.gaps]
    )
    st.dataframe(gaps, hide_index=True, width="stretch")

d1, d2, d3 = st.columns(3)
d1.download_button("Download report (JSON)", report.model_dump_json(indent=2), f"{image_id}_availability.json", "application/json")
d2.download_button("Download detections (JSON)", det.model_dump_json(indent=2), f"{image_id}_detections.json", "application/json")
d3.download_button(
    "Download empty spaces (CSV)",
    pd.DataFrame([{"gap_id": g.gap_id, "shelf": g.shelf_id, "missing_packs": g.estimated_missing_facings, "confidence": g.confidence, "bbox": g.bbox} for g in det.gaps]).to_csv(index=False),
    f"{image_id}_empty_spaces.csv", "text/csv",
)
