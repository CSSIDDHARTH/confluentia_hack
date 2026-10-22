"""shelfSense: check any shelf photo (run by app/dashboard.py when "Check a photo" is chosen).

Empty spaces, per-row fill and a refill list for any shelf photo, with no planogram. Works from saved pack
detections (no GPU needed) or, when the detector is installed, from an uploaded photo. The sensitivity sliders
recompute from the saved packs instantly.
"""

import html
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from shelfsense.contracts import load_detection_output
from shelfsense.pricing.label_detector import DEFAULT_WEIGHTS
from shelfsense.vision.availability import analyze_packs
from shelfsense.vision.gaps import LOW_FILL_RATIO, MIN_GAP_WIDTHS
from shelfsense.vision.render import TONAL, draw_availability

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "availability_demo"
WEIGHTS = ROOT / "models" / "pack_detector.pt"
VOID_WEIGHTS = ROOT / "models" / "void_detector.pt"
VOID_DEFAULT = 0.3  # chosen on held-out labelled data (scripts/eval_voids.py)
FIRST = "shelf_06_coffee_us"  # clear empty spaces and a low-stock row: the most telling of the saved shelves
COUNTRIES = {"uk": "UK", "us": "US", "de": "Germany", "cn": "China"}


def detector_available() -> bool:
    if not WEIGHTS.exists():
        return False
    try:
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


def sample_name(stem: str) -> str:
    """shelf_06_coffee_us -> Coffee (US)"""
    parts = stem.split("_")[2:]
    country = COUNTRIES.get(parts[-1], parts[-1].upper()) if parts else ""
    return f"{' '.join(parts[:-1]).capitalize()} ({country})"


def chip(text: str, key: str) -> str:
    bg, fg = TONAL[key]
    return f'<span class="ss-chip" style="background:{bg};color:{fg}">{html.escape(text)}</span>'


@st.cache_resource(show_spinner="Getting the pack detector ready...")
def load_detector():
    from shelfsense.vision.detect import ProductDetector

    return ProductDetector(WEIGHTS, conf=0.25, imgsz=1024, max_det=600)


@st.cache_resource(show_spinner="Getting the empty-space detector ready...")
def load_voider():
    from shelfsense.vision.detect import VOID_CONF_FLOOR, ProductDetector

    return ProductDetector(VOID_WEIGHTS, conf=VOID_CONF_FLOOR, imgsz=640, max_det=150)


@st.cache_data(show_spinner="Looking at the photo...")
def detect_upload(data: bytes, suffix: str):
    from shelfsense.vision.detect import detect_voids

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    boxes, confs, size = load_detector().boxes(path)
    voids = detect_voids(load_voider(), path) if VOID_WEIGHTS.exists() else None
    return boxes, confs, size, voids



live = detector_available()
saved = sorted(p.stem for p in DEMO.glob("*.json") if not p.name.endswith((".report.json", ".voids.json")))
saved = sorted(saved, key=lambda s: s != FIRST)  # the most telling shelf first
use_voids = VOID_WEIGHTS.exists()

title, src, sens = st.columns([3, 2, 1], vertical_alignment="bottom")
title.html('<h1 class="ss-h1">Check any shelf photo</h1>'
           '<p class="ss-lead">No plan needed: finds the empty space on a shelf and tells you what to refill.</p>')
sources = ["Try a sample"] + (["Upload a photo"] if live else [])
source = src.segmented_control("Source", sources, default=sources[0], key="scan-source", label_visibility="collapsed") or sources[0]
with sens.popover("Sensitivity", icon=":material/tune:", width="stretch"):
    if use_voids:
        void_conf = st.slider(
            "Empty-space confidence", 0.1, 0.8, VOID_DEFAULT, 0.05, format="%.2f",
            help="Higher keeps fewer, surer empty spaces. On labelled photos: 0.2 finds more (recall 0.72, precision 0.54), 0.4 is surer (precision 0.78, recall 0.48).",
        )
        gap_widths = MIN_GAP_WIDTHS
    else:
        gap_widths = st.slider(
            "Smallest empty space, in pack widths", 0.4, 2.0, MIN_GAP_WIDTHS, 0.1,
            help="A stretch of shelf this many median pack widths wide, with no pack in it, counts as empty.",
        )
    low_fill = st.slider(
        "Low stock below this fill", 0.2, 0.95, LOW_FILL_RATIO, 0.05, format="%.2f",
        help="Share of the shelf's width covered by packs. A row below it needs refilling.",
    )

boxes = confs = size = image = voids = None
image_id = shown = ""
if source == "Try a sample":
    if not saved:
        st.error(f"No saved results in {DEMO.relative_to(ROOT)}. Run: python scripts/run_availability.py --images data/test_images --out data/availability_demo")
        st.stop()
    image_id = st.pills("Sample shelf", saved, format_func=sample_name, default=saved[0], key="sample", label_visibility="collapsed") or saved[0]
    saved_det = load_detection_output(DEMO / f"{image_id}.json")
    boxes = [i.bbox for i in saved_det.items]
    confs = [i.det_confidence for i in saved_det.items]
    size = (saved_det.image_width, saved_det.image_height)
    shown = saved_det.image_path
    image = Image.open(ROOT / shown)
    voids_file = DEMO / f"{image_id}.voids.json"
    if use_voids and voids_file.exists():
        voids = [(v["bbox"], v["conf"]) for v in json.loads(voids_file.read_text())]
else:
    up = st.file_uploader("A shelf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Take a straight-on photo of a whole shelf of packaged goods, from one or two metres, and drop it here.",
                icon=":material/photo_camera:")
        st.stop()
    boxes, confs, size, voids = detect_upload(up.getvalue(), Path(up.name).suffix or ".jpg")
    image_id, shown = Path(up.name).stem, up.name
    image = Image.open(up)

kept = [v for v in voids if v[1] >= void_conf] if (use_voids and voids is not None) else None
det, report = analyze_packs(image_id, shown, size, boxes, confs, low_fill=low_fill, min_gap_widths=gap_widths, voids=kept)

low_rows = [r.shelf_id for r in report.rows if r.low_stock]
with st.container(border=True, key="sheet-scanfigures"):  # one sheet for the figures, not a tile each
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall fill", f"{report.overall_fill:.0%}")
    c2.metric("Empty spaces", report.empty_spaces)
    c3.metric("Packs to refill", report.missing_facings)
    c4.metric("Low-stock rows", len(low_rows), help=", ".join(low_rows) or "none")
    c5.metric("Packs found", len(det.items))

for w in report.warnings:
    st.warning(w, icon=":material/warning:")

left, right = st.columns([1.2, 1], gap="medium")
with left.container(border=True, key="sheet-scanphoto"):
    view = draw_availability(image, det, report, footer=False)
    view.thumbnail((1100, 760))
    st.image(view, width="stretch")
    st.html(f'<div class="ss-legend">{chip("pack", "ok")}{chip("empty space", "bad")}'
            "<span>The bar and badge on the left of each shelf row show how full it is.</span></div>")
with right:
    with st.container(border=True, key="sheet-scantodo"):
        st.html('<div class="ss-t2">What to refill</div>')
        if report.actions:
            for k, a in enumerate(report.actions, 1):
                with st.container(key=f"{'rowtop' if k == 1 else 'rowline'}-refill-{k}"):
                    st.html(f'<div class="ss-todo"><b>{k}.</b> {html.escape(a)}</div>')
        elif not report.warnings:
            st.success("Nothing to refill at these settings.", icon=":material/check_circle:")
    with st.container(border=True, key="sheet-scanrows"):
        st.html('<div class="ss-t2">How full each shelf row is</div>')
        rows = pd.DataFrame(
            [{"row": r.shelf_id, "packs": r.packs, "fill": r.fill_ratio, "status": "low stock" if r.low_stock else "ok"} for r in report.rows]
        )
        st.dataframe(
            rows, hide_index=True, width="stretch",
            column_config={"fill": st.column_config.ProgressColumn("fill", format="percent", min_value=0.0, max_value=1.0)},
        )

with st.expander("Details and downloads"):
    if det.gaps:
        st.dataframe(
            pd.DataFrame([{"row": g.shelf_id, "about (packs)": g.estimated_missing_facings, "confidence": g.confidence,
                           "bbox": [round(v) for v in g.bbox]} for g in det.gaps]),
            hide_index=True, width="stretch",
        )
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download report (JSON)", report.model_dump_json(indent=2), f"{image_id}_availability.json", "application/json",
                       icon=":material/download:", width="stretch")
    d2.download_button("Download detections (JSON)", det.model_dump_json(indent=2), f"{image_id}_detections.json", "application/json",
                       icon=":material/download:", width="stretch")
    d3.download_button(
        "Download empty spaces (CSV)",
        pd.DataFrame([{"gap_id": g.gap_id, "shelf": g.shelf_id, "missing_packs": g.estimated_missing_facings, "confidence": g.confidence, "bbox": g.bbox} for g in det.gaps]).to_csv(index=False),
        f"{image_id}_empty_spaces.csv", "text/csv", icon=":material/download:", width="stretch",
    )

if source != "Try a sample" and DEFAULT_WEIGHTS.exists():
    with st.expander("Read the price labels on this photo"):
        st.caption("Finds the shelf-edge price labels and reads them. With no plan they are not matched to products.")
        if st.button("Read the labels", key="scan-labels"):
            from shelfsense.pricing.label_detector import detect_labels

            with st.spinner("Reading the labels..."):
                found = detect_labels(image.convert("RGB"))
            st.dataframe(
                pd.DataFrame([{"label": lb.label_id, "price": lb.price, "confidence": lb.ocr_confidence} for lb in found]),
                hide_index=True, width="stretch",
            )
