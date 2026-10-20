"""shelfSense dashboard (mock stage).

Reads the three contracts and shows them joined per slot. This proves the
A -> B handoff works; compliance scoring and recommendations come later.

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

ROOT = Path(__file__).resolve().parents[1]

ITEM_COLOUR = "#2e9e44"
UNKNOWN_COLOUR = "#f0a020"
GAP_COLOUR = "#d62828"
LABEL_COLOUR = "#1d6fd6"


def overlay(det: DetectionOutput) -> Image.Image:
    path = ROOT / det.image_path
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (det.image_width, det.image_height), "white")
    draw = ImageDraw.Draw(img)
    for it in det.items:
        draw.rectangle(it.bbox, outline=ITEM_COLOUR if it.sku else UNKNOWN_COLOUR, width=4)
    for gap in det.gaps:
        draw.rectangle(gap.bbox, outline=GAP_COLOUR, width=5)
        draw.text((gap.bbox[0] + 8, gap.bbox[1] + 8), "EMPTY", fill=GAP_COLOUR)
    for lb in det.labels:
        draw.rectangle(lb.bbox, outline=LABEL_COLOUR, width=3)
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
st.title("shelfSense: shelf audit (mock)")

with st.sidebar:
    det_path = st.text_input("Detection output", "contracts/detection_output.example.json")
    pg_path = st.text_input("Planogram", "contracts/planogram.example.json")
    cat_path = st.text_input("Catalogue", "contracts/catalogue.example.csv")

det = load_detection_output(ROOT / det_path)
planogram = load_planogram(ROOT / pg_path)
catalogue = load_catalogue(ROOT / cat_path)

if det.planogram_id != planogram.planogram_id:
    st.warning(f"Image is for planogram {det.planogram_id}, but {planogram.planogram_id} is loaded.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Items detected", len(det.items))
c2.metric("Unrecognised", sum(i.sku is None for i in det.items))
c3.metric("Gaps", len(det.gaps))
c4.metric("Labels read", len(det.labels))

left, right = st.columns([3, 2])
with left:
    st.image(overlay(det), caption=f"{det.image_id}: green = product, orange = unrecognised, red = empty, blue = label")
with right:
    st.subheader("Planogram vs. detected, per slot")
    st.dataframe(slot_table(det, planogram, catalogue), hide_index=True, width="stretch")

with st.expander("Raw detection output"):
    st.json(det.model_dump())
