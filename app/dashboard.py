"""shelfSense dashboard.

Home: one card per shelf (score, money at stake, to-dos) and one switch for the whole store: the current plan or
the new plan from head office. A shelf: check it live from its photo, see the photo and the plan side by side in
the same colours, and work through the to-do list. "Check any shelf photo": empty space and what to refill, no
plan needed (app/views/availability.py). Mock shelves and custom files are behind the sidebar's "Developer inputs".

    streamlit run app/dashboard.py
"""

import hashlib
import html
import os
import runpy
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from shelfsense.advisor.edit import apply_rows, changes, plan_rows, swap_products
from shelfsense.advisor.plan import plan as plan_steps
from shelfsense.contracts import DetectionOutput, load_catalogue, load_detection_output, load_planogram
from shelfsense.pricing.label_assign import assign_labels_to_slots
from shelfsense.pricing.price_check import PriceIssueType, check_prices
from shelfsense.report.briefing import DEFAULT_MODEL as SLM_MODEL
from shelfsense.report.briefing import write_briefing
from shelfsense.report.compliance import WEIGHTS, ComplianceReport, compute_report
from shelfsense.report.export import report_to_json, violations_to_csv
from shelfsense.report.impact import at_stake, currency
from shelfsense.vision import live
from shelfsense.vision.render import STATUS, TONAL, draw_audit, slot_status

ROOT = Path(__file__).resolve().parents[1]
BAYS = ROOT / "data" / "demo_bays"
SHOWN_SIZE = (1100, 760)  # the photo is shrunk to fit this box, so photo and plan fit on one screen
EDIT = "edit"
PLAN_NAMES = {"compliant": "Current plan", "scenario": "New plan from head office", EDIT: "Try your own plan"}
PLAN_HELP = {
    "compliant": "The plan this shelf was stocked to.",
    "scenario": "Head office swapped two products, added one this shelf doesn't carry, gave one product more space "
    "and raised a price. The shelf hasn't been changed yet.",
    EDIT: "Change what any slot should hold, and watch the check update. The photo stays the same.",
}
AISLES = {"Oil-Vinegar": "Oil & vinegar", "Canned": "Canned food", "Cereals": "Breakfast cereals"}
STAGES = {
    "find product blocks": "Found the products",
    "name each block": "Recognised each product",
    "align to the plan": "Matched them to the plan",
    "count packs": "Counted the packs",
    "find empty spaces": "Looked for empty space",
    "read price labels": "Read the price labels",
}
MOCK_PLANOGRAM = "contracts/planogram.example.json"
MOCK_CATALOGUE = "contracts/catalogue.example.csv"
MOCKS = {  # name -> (detection output, planogram, catalogue)
    "Mock shelf (as shipped)": ("contracts/detection_output.example.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Low-stock demo (all 5 violation types)": ("data/mock/mock_shelf_002_lowstock.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Near-clean shelf (one unreadable label)": ("data/mock/mock_shelf_003_clean.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
}

CSS = """
<style>
.block-container {padding-top: 3.6rem; max-width: 1440px;}
[data-testid="stHeader"] {background: transparent;}
div[class*="st-key-card-photo"] [data-testid="stFullScreenFrame"] > div,
div[class*="st-key-card-scan-photo"] [data-testid="stFullScreenFrame"] > div {width: 100%; display: flex; justify-content: center;}
div[class*="st-key-card-photo"] img, div[class*="st-key-card-scan-photo"] img {max-height: 70vh; width: auto !important;
  max-width: 100%; display: block; margin: 0 auto; object-fit: contain;}
div[class*="st-key-card"] {background: #fff; border: none !important; border-radius: 16px;
  box-shadow: 0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15);}
.ss-bar {display: flex; align-items: center; gap: 14px;}
.ss-name {font-size: 1.55rem; font-weight: 500; color: #202124; line-height: 1.1;}
.ss-tagline {font-size: .95rem; color: #5f6368;}
.ss-ring {border-radius: 50%; display: flex; align-items: center; justify-content: center; flex: none;}
.ss-ring-in {width: 80%; height: 80%; border-radius: 50%; background: #fff; display: flex; align-items: center;
  justify-content: center; font-weight: 500;}
.ss-chip {display: inline-block; padding: 2px 12px; border-radius: 16px; font-size: .82rem; font-weight: 500;}
.ss-muted {color: #5f6368; font-size: .9rem;}
.ss-shelf {margin-bottom: 10px;}
.ss-shelf-name {font-size: .78rem; color: #5f6368; margin: 0 0 4px 2px; letter-spacing: .02em;}
.ss-row {display: flex; gap: 6px; padding-bottom: 8px; border-bottom: 3px solid #dadce0;}
.ss-tile {min-width: 0; border-radius: 10px; padding: 8px 10px; line-height: 1.3; overflow: hidden;}
.ss-tile-id {font-size: .72rem; opacity: .8;}
.ss-tile-name {font-weight: 500; font-size: .9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.ss-tile-meta {font-size: .78rem;}
.ss-tile-status {font-size: .78rem; font-weight: 500; margin-top: 2px;}
.ss-big {font-size: 1.05rem; line-height: 1.5;}
.ss-done {text-decoration: line-through; color: #80868b;}
</style>
"""

LOGO = '<img src="app/static/logo.svg" width="40" height="40" alt="">'  # served from app/static (st.html strips inline SVG)


# ---------- small pieces of HTML ----------

def ring(score: float, size: int = 96) -> str:
    """A score ring in the style of Google's Lighthouse: green from 90, orange from 50, red below."""
    key = "ok" if score >= 90 else "low" if score >= 50 else "bad"
    colour = STATUS[key]
    return (
        f'<div class="ss-ring" data-score="{score:.0f}" style="width:{size}px;height:{size}px;'
        f'background:conic-gradient({colour} 0 {score:.0f}%, #e8eaed {score:.0f}% 100%)">'
        f'<div class="ss-ring-in" style="color:{colour};font-size:{size * 0.32:.0f}px">{score:.0f}</div></div>'
    )


def chip(text: str, key: str) -> str:
    bg, fg = TONAL[key]
    return f'<span class="ss-chip" style="background:{bg};color:{fg}">{html.escape(text)}</span>'


def plan_html(
    planogram, catalogue, det: DetectionOutput | None, report: ComplianceReport | None,
    highlight: set[str] | None = None, done: set[str] | None = None,
) -> str:
    """The plan shelf by shelf, top first: one tile per slot, as wide as its planned facings, coloured by what the
    photo shows there (grey before the photo is read). Price problems get a note on the tile. `done`: slots whose
    problems were all ticked off on the to-do list."""
    status = slot_status(report) if report else {}
    price_note: dict[str, str] = {}
    for v in report.violations if report else []:
        if v.kind.value == "price_mismatch":
            price_note[v.slot_id] = "price label wrong"
        elif v.kind.value == "unreadable_price":
            price_note.setdefault(v.slot_id, "check price label")
    shelves = []
    for shelf in planogram.shelves:
        tiles = []
        for slot in shelf.slots:
            item = catalogue.get(slot.sku)
            name = html.escape(item.name if item else slot.sku)
            key, words = ("unread", "not checked yet") if report is None else status.get(
                slot.slot_id, ("ok", "done \u2713" if done and slot.slot_id in done else "as planned")
            )
            bg, fg = TONAL[key]
            seen = "" if det is None or report is None else (
                f" · {sum(i.slot_id == slot.slot_id and i.sku == slot.sku for i in det.items)} seen"
            )
            note = f'<div class="ss-tile-meta">{price_note[slot.slot_id]}</div>' if slot.slot_id in price_note else ""
            faded = ";opacity:.3" if highlight and slot.slot_id not in highlight else ""
            tiles.append(
                f'<div class="ss-tile" style="flex:{slot.facings} 1 0;background:{bg};color:{fg};'
                f'border-left:5px solid {STATUS[key]}{faded}">'
                f'<div class="ss-tile-id">{slot.slot_id}</div><div class="ss-tile-name">{name}</div>'
                f'<div class="ss-tile-meta">{slot.facings} planned{seen}</div>'
                f'<div class="ss-tile-status">{words}</div>{note}</div>'
            )
        shelves.append(
            f'<div class="ss-shelf"><div class="ss-shelf-name">SHELF {shelf.index + 1}</div>'
            f'<div class="ss-row">{"".join(tiles)}</div></div>'
        )
    return "".join(shelves)


def label_notes(price_issues: list) -> dict[str, tuple[str, str]]:
    """label_id -> (tag text, status key) for the labels with a price issue; the rest show their price."""
    notes = {}
    for i in price_issues:
        if i.issue_type == PriceIssueType.PRICE_MISMATCH and not i.low_confidence:
            notes[i.label_id] = (f"{i.label_price:.2f} ≠ {i.catalogue_price:.2f}", "bad")
        elif i.issue_type == PriceIssueType.PRICE_MISMATCH:
            notes[i.label_id] = (f"{i.label_price:.2f}? check", "check")
        elif i.issue_type == PriceIssueType.UNREADABLE_PRICE:
            notes[i.label_id] = ("unreadable", "check")
        else:
            notes[i.label_id] = ((f"{i.label_price:.2f} " if i.label_price is not None else "") + "no slot", "check")
    return notes


def shown(im: Image.Image) -> Image.Image:
    im = im.copy()
    im.thumbnail(SHOWN_SIZE)
    return im


# ---------- data ----------

def bay_files(bay: str, plan: str) -> tuple[str, str, str]:
    """Detection, planogram and catalogue for a saved plan; `edit` starts from the current plan on the same reading."""
    rel = BAYS.relative_to(ROOT) / bay
    plan = "compliant" if plan == EDIT else plan
    catalogue = "catalogue_scenario.csv" if plan == "scenario" and (BAYS / bay / "catalogue_scenario.csv").exists() else "catalogue.csv"
    return f"{rel}/detection_{plan}.json", f"{rel}/planogram_{plan}.json", f"{rel}/{catalogue}"


def bay_title(bay: str) -> tuple[str, str]:
    """(aisle, where) for a demo bay, e.g. ("Oil & vinegar", "Store 4 · photo 9")."""
    cat = load_catalogue(BAYS / bay / "catalogue.csv")
    pg = load_planogram(BAYS / bay / "planogram_compliant.json")
    common = Counter(cat[s.sku].category.split("/")[-1] for sh in pg.shelves for s in sh.slots if s.sku in cat).most_common(1)
    aisle = AISLES.get(common[0][0], common[0][0]) if common else bay
    store, _, photo = bay.partition("_")
    return aisle, f"{store.replace('store', 'Store ')} · photo {photo}"


@st.cache_data(show_spinner=False)
def bay_summary(bay: str, plan: str) -> dict:
    """Score, money at stake, to-dos and the worst problem from the saved reading, for the home cards."""
    det_p, pg_p, cat_p = bay_files(bay, plan)
    det, pg, cat = load_detection_output(ROOT / det_p), load_planogram(ROOT / pg_p), load_catalogue(ROOT / cat_p)
    report = compute_report(det, pg, cat, group_items=True)
    steps = plan_steps(report, cat, pg)
    status = slot_status(report)
    worst = next(((key, w) for k in ("bad", "low", "check") for key, w in status.values() if key == k), None)
    expected = {s.slot_id: s.sku for sh in pg.shelves for s in sh.slots}
    im = draw_audit(Image.open(ROOT / det.image_path), det, report, expected, label_notes(check_prices(det, cat)))
    im = ImageOps.fit(im, (720, 540), centering=(0.5, 0.35))  # every card the same shape
    return {
        "score": report.score, "money": sum(at_stake(report, det, pg, cat).values()), "currency": currency(cat),
        "todo": len(steps), "worst": worst, "image": im,
    }


@st.cache_resource(show_spinner="Getting the models ready...")
def load_chain() -> live.LiveChain:
    return live.shared()


@st.cache_data(show_spinner=False)
def cached_briefing(report_json: str, catalogue_path: str, use_model: bool):
    rep = ComplianceReport.model_validate_json(report_json)
    return write_briefing(rep, load_catalogue(ROOT / catalogue_path), model=SLM_MODEL if use_model else None)


# ---------- page pieces ----------

def go(view: str, bay: str | None = None) -> None:
    st.session_state["view"] = view
    if bay is not None:
        st.session_state["bay"] = bay
    st.session_state.pop("focus", None)


def app_bar() -> None:
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.html(
        f'<div class="ss-bar">{LOGO}<div><div class="ss-name">shelfSense</div>'
        '<div class="ss-tagline">One photo of a shelf. Every problem on it. What to fix first.</div></div></div>'
    )
    with right.popover("How it works", width="stretch"):
        st.markdown(
            "**From one photo to a to-do list**\n"
            "1. Find every product on the shelf\n"
            "2. Recognise each one, from product photos\n"
            "3. Match them to the store's plan, shelf by shelf\n"
            "4. Count the packs and look for empty space\n"
            "5. Read the price labels (two text readers vote)\n"
            "6. Score the shelf, put a price on each problem and list what to do. A small language model "
            "(Qwen 2.5, 1.5B, on this laptop) writes a two-line summary, and a checker makes sure it is right.\n\n"
            "**Measured on photos the models never saw**\n"
            "- Whole check, 35 shelves: median 53 on the correct plan; the best three (100, 98, 98) are shown here\n"
            "- Empty space: finds 62% of hand-marked gaps; 66% of its flags are right (94 photos)\n"
            "- Prices: 29 to 30 of 31 hand-checked labels read right, none wrong\n"
            "- Speed: about 2 s a shelf on a laptop GPU, 4.8 s on 4 CPU cores with no GPU\n\n"
            "**Limits**: no store shares real plans or prices, so plans come from a public dataset's labels and the "
            "changes are planted; most shelves score lower than these three; gaps between bottles can be missed."
        )


def plan_switch(key: str, with_edit: bool) -> str:
    """The plan choice, kept when moving between pages."""
    options = list(PLAN_NAMES) if with_edit else ["compliant", "scenario"]
    current = st.session_state.get("plan", "compliant")
    if current not in options:
        current = "scenario"
    if st.session_state.get(key) not in options:
        st.session_state[key] = current
    choice = st.segmented_control("Plan", options, format_func=PLAN_NAMES.get, key=key, label_visibility="collapsed")
    choice = choice or current
    st.session_state["plan"] = choice
    return choice


# ---------- home ----------

def view_home(bays: list[str]) -> None:
    st.markdown("### Today's shelf check")
    plan = plan_switch("plan-home", with_edit=False)
    st.caption(PLAN_HELP[plan])
    summaries = {b: bay_summary(b, plan) for b in bays}
    order = sorted(bays, key=lambda b: -summaries[b]["money"])
    money = summaries[order[0]]["currency"] if order else ""
    k1, k2, k3 = st.columns(3)
    k1.metric("Shelves checked", len(bays), border=True)
    k2.metric("Money at stake", f"{money} {sum(s['money'] for s in summaries.values()):.2f}".strip(), border=True,
              help="Missing packs times their price, plus wrong prices times the packs they apply to. An estimate for putting work in order.")
    k3.metric("To-dos", sum(s["todo"] for s in summaries.values()), border=True)

    cols = st.columns(len(order) + 1)
    for col, bay in zip(cols, order):
        s = summaries[bay]
        aisle, where = bay_title(bay)
        with col.container(border=True, key=f"card-{bay}"):
            st.image(s["image"], width="stretch")
            top, ring_col = st.columns([3, 2], vertical_alignment="center")
            top.markdown(f"**{aisle}**  \n<span class='ss-muted'>{where}</span>", unsafe_allow_html=True)
            ring_col.html(ring(s["score"], 72))
            st.html(
                f'<div class="ss-big"><b>{s["currency"]} {s["money"]:.2f}</b> at stake · <b>{s["todo"]}</b> to-do'
                f'{"" if s["todo"] == 1 else "s"}</div>'
                + (f'<div style="margin-top:6px">{chip(s["worst"][1], s["worst"][0])}</div>'
                   if s["worst"] else f'<div style="margin-top:6px">{chip("all as planned", "ok")}</div>')
            )
            if st.button("Open shelf", key=f"open-{bay}", type="primary", width="stretch"):
                go("shelf", bay)
                st.rerun()
    with cols[-1].container(border=True, key="card-scan"):
        st.markdown("#### Check any shelf photo")
        st.markdown("<span class='ss-muted'>No plan needed. Finds empty space and tells you what to refill, "
                    "on a photo from any shop.</span>", unsafe_allow_html=True)
        if st.button("Check a photo", key="open-scan", width="stretch"):
            go("scan")
            st.rerun()


# ---------- one shelf ----------

def view_shelf(bay: str | None, files: tuple[str, str, str] | None = None) -> None:
    """A demo bay (bay) or, in developer mode, any files (detection, planogram, catalogue)."""
    if bay is not None:
        if st.button("← All shelves", key="back", type="tertiary"):
            go("home")
            st.rerun()
        aisle, where = bay_title(bay)
        st.markdown(f"### {aisle}")
        st.caption(f"{where} · a real supermarket photo from a store the models never saw")
        plan = plan_switch("plan-shelf", with_edit=True)
        st.caption(PLAN_HELP[plan])
        det_path, pg_path, cat_path = bay_files(bay, plan)
    else:
        plan, (det_path, pg_path, cat_path) = "compliant", files
    det = load_detection_output(ROOT / det_path)
    planogram = load_planogram(ROOT / pg_path)
    catalogue = load_catalogue(ROOT / cat_path)
    base_score = None

    if bay is not None and plan == EDIT:
        planogram, base_score, det = plan_editor(bay, planogram, catalogue, det)

    if bay is not None:
        photo, _ = live.bay_photo(BAYS / bay)
        live_key = f"live-{bay}-{plan}-{hashlib.md5(planogram.model_dump_json().encode()).hexdigest()[:10]}"
        with st.container(border=True, key="card-read"):
            if not live.available():
                st.info("Checking live needs the product index and the detectors in `models/`; showing the saved check of this photo.")
            else:
                b, t = st.columns([2, 3], vertical_alignment="center")
                run = b.button("Check the shelf now", key=f"run-{live_key}", type="primary", icon=":material/photo_camera:")
                result = st.session_state.get(live_key)
                if result is not None:
                    t.html(chip(f"Checked live in {result.total:.1f} s on this laptop's {'GPU' if result.device == 'cuda' else 'CPU'}", "ok"))
                elif plan != EDIT:
                    t.toggle("Show the saved check", key="show-saved",
                             help="The result of an earlier check of the same photo; checking now gives the same result.")
                if run:
                    with st.status("Checking the shelf...", expanded=True) as status:
                        def on_stage(stage: str, seconds: float) -> None:
                            status.write(f"✓ {STAGES.get(stage, stage)} · {seconds:.2f} s")

                        result = load_chain().run(
                            photo, planogram, read_prices=live.prices_available(), on_stage=on_stage,
                            shown_path=str(photo.relative_to(ROOT)) if photo.is_relative_to(ROOT) else photo.name,
                        )
                        st.session_state[live_key] = result
                        status.update(label=f"Checked in {result.total:.1f} s", state="complete", expanded=False)
                    st.rerun()
        result = st.session_state.get(live_key)
        if result is not None:
            det = result.det
        elif plan != EDIT and live.available() and not st.session_state.get("show-saved"):
            left, right = st.columns([1.35, 1])
            with left.container(border=True, key="card-photo"):
                st.image(shown(Image.open(BAYS / bay / "photo.jpg")), width="stretch")
            with right.container(border=True, key="card-plan"):
                st.markdown("**The plan for this shelf**")
                st.html(plan_html(planogram, catalogue, None, None))
                st.caption("Press **Check the shelf now** to compare the photo with it.")
            return

    report = compute_report(det, planogram, catalogue, group_items=True)
    steps = plan_steps(report, catalogue, planogram)
    stake = at_stake(report, det, planogram, catalogue)
    money = currency(catalogue)
    price_issues = check_prices(det, catalogue)
    focus = st.session_state.get("focus")
    focus_slots = set(focus[1]) if focus else None

    # to-dos ticked off: their findings count as fixed in the score, the money and the plan (the photo stays as checked)
    done_key = f"done-{hashlib.md5(det_path.encode()).hexdigest()[:6]}"
    done_steps = [k for k in range(1, len(steps) + 1) if st.session_state.get(f"{done_key}-{k}")]
    done_ids = {v for k in done_steps for v in steps[k - 1].violation_ids}
    open_ = [v for v in report.violations if v.violation_id not in done_ids]
    after = report.model_copy(update={
        "violations": open_, "score": round(max(0.0, 100.0 - sum(WEIGHTS[v.kind] for v in open_)), 1),
    })
    done_slots = {v.slot_id for v in report.violations if v.violation_id in done_ids} - {v.slot_id for v in open_}
    fixed = sum(stake.get(v, 0.0) for v in done_ids)

    left, right = st.columns([1.35, 1])
    with left.container(border=True, key="card-photo"):
        photo_path = ROOT / det.image_path
        base = Image.open(photo_path) if photo_path.exists() else Image.new("RGB", (det.image_width, det.image_height), "white")
        expected = {s.slot_id: s.sku for sh in planogram.shelves for s in sh.slots}
        st.image(shown(draw_audit(base, det, report, expected, label_notes(price_issues), focus_slots)), width="stretch")
        st.html(
            f'<div class="ss-muted">{chip("right product", "ok")} {chip("wrong product", "bad")} '
            f'{chip("not recognised / check", "check")} {chip("price label", "label")} · red fill = empty space'
            + (" · the photo shows the shelf as checked" if done_steps else "") + "</div>"
        )
    with right:
        with st.container(border=True, key="card-score"):
            r, m1, m2 = st.columns([1.1, 1.3, 1], vertical_alignment="center")
            change = "" if base_score is None else f" · {report.score - base_score:+.0f} vs current plan"
            if done_steps:
                change += f" · checked {report.score:.0f}, {after.score - report.score:+.0f} with {len(done_steps)} done"
            r.html(ring(after.score, 92) + f'<div class="ss-muted" style="margin-top:4px">Shelf score{change}</div>')
            m1.metric("Money at stake", f"{money} {sum(stake.values()) - fixed:.2f}".strip(),
                      delta=f"-{(money + ' ') if money else ''}{fixed:.2f} done" if fixed else None, delta_color="inverse",
                      help="Missing packs times their price, plus wrong prices times the packs they apply to. An estimate for putting work in order.")
            m2.metric("To-dos", len(steps) - len(done_steps),
                      delta=f"-{len(done_steps)} done" if done_steps else None, delta_color="inverse")
        with st.container(border=True, key="card-plan"):
            st.markdown("**The plan for this shelf**")
            st.html(plan_html(planogram, catalogue, det, after, focus_slots, done_slots))

    todo, prices, summary, details = st.tabs(["To-do list", "Prices", "Shift summary", "Details"])
    with todo:
        todo_list(steps, report, stake, money, done_key, fixed)
    with prices:
        price_list(det, catalogue, price_issues, base)
    with summary:
        shift_summary(report, cat_path)
    with details:
        details_tab(det, planogram, catalogue, report, stake)


def plan_editor(bay, shelf_plan, catalogue, det):
    """The editable plan; returns (edited plan, score on the real plan, reading with labels under the edited plan)."""
    rows_key, ver_key = f"edit-rows-{bay}", f"edit-ver-{bay}"
    if rows_key not in st.session_state:
        st.session_state[rows_key], st.session_state[ver_key] = plan_rows(shelf_plan), 0
    original = plan_rows(shelf_plan)
    names = {sku: f"{c.name} ({sku})" for sku, c in sorted(catalogue.items())}
    to_sku = {label: sku for sku, label in names.items()}
    with st.container(border=True, key="card-edit"):
        st.markdown("**Change the plan**")
        table = pd.DataFrame(st.session_state[rows_key])
        table["product"] = table["sku"].map(names).fillna(table["sku"])
        editor = st.data_editor(
            table[["slot", "product", "facings", "minimum"]], hide_index=True, width="stretch",
            key=f"editor-{bay}-{st.session_state[ver_key]}", disabled=["slot"],
            column_config={
                "slot": st.column_config.TextColumn("Slot"),
                "product": st.column_config.SelectboxColumn("Should hold", options=list(names.values()), required=True),
                "facings": st.column_config.NumberColumn("Facings planned", min_value=1, max_value=30, step=1),
                "minimum": st.column_config.NumberColumn("Low stock below", min_value=1, max_value=30, step=1),
            },
        )
        rows = [
            {"slot": r.slot, "sku": to_sku.get(r.product, r.product), "facings": int(r.facings), "minimum": int(r.minimum)}
            for r in editor.itertuples()
        ]
        slots = [r["slot"] for r in rows]
        c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
        a = c1.selectbox("Swap this slot", slots, index=0, key=f"swap-a-{bay}")
        b = c2.selectbox("with this slot", slots, index=min(1, len(slots) - 1), key=f"swap-b-{bay}")
        if c3.button("Swap their products", disabled=a == b, width="stretch"):
            st.session_state[rows_key] = swap_products(rows, a, b)
            st.session_state[ver_key] += 1
            st.rerun()
        if c4.button("Reset to the shelf's plan", width="stretch"):
            st.session_state[rows_key] = original
            st.session_state[ver_key] += 1
            st.rerun()
        planogram = apply_rows(shelf_plan, rows)
        edits = changes(original, rows)
        st.caption(f"{len(edits)} change{'s' if len(edits) != 1 else ''}" + (": " + "; ".join(edits) if edits else ". The photo stays the same."))
    # labels sit under the same slots, which now expect the edited products (a swap leaves each label under the other product)
    det = det.model_copy(update={"labels": assign_labels_to_slots(det.labels, det, planogram)})
    base_score = compute_report(
        det.model_copy(update={"labels": assign_labels_to_slots(det.labels, det, shelf_plan)}), shelf_plan, catalogue, group_items=True
    ).score
    return planogram, base_score, det


def todo_list(steps, report, stake, money, done_key: str, fixed: float) -> None:
    if not steps:
        st.success("Nothing to do: the shelf matches the plan.")
        return
    n_done = sum(bool(st.session_state.get(f"{done_key}-{k}")) for k in range(1, len(steps) + 1))
    total = sum(stake.values())
    st.progress(n_done / len(steps), text=f"{n_done} of {len(steps)} done" + (f" · {money} {fixed:.2f} of {total:.2f} sorted" if total else ""))
    if n_done == len(steps):
        st.success("All done. Press **Check the shelf now** with a new photo to confirm the shelf matches the plan.")
    slots_of = {v.violation_id: v.slot_id for v in report.violations}
    focus = st.session_state.get("focus")
    if focus:
        if st.button(f"Showing to-do {focus[0]} on the photo. Show everything", key="focus-clear", type="tertiary"):
            st.session_state.pop("focus", None)
            st.rerun()
    for k, step in enumerate(steps, 1):
        value = sum(stake.get(v, 0.0) for v in step.violation_ids)
        slots = sorted({slots_of[v] for v in step.violation_ids if slots_of.get(v)})
        with st.container(border=True, key=f"card-todo-{k}"):
            c0, c1, c2 = st.columns([0.5, 8, 2.2], vertical_alignment="center")
            done = c0.checkbox("done", key=f"{done_key}-{k}", label_visibility="collapsed")
            text = html.escape(step.text)
            c1.html(f'<div class="ss-big{" ss-done" if done else ""}"><b>{k}.</b> {text}</div>'
                    + (f'<div style="margin-top:4px">{chip(f"{money} {value:.2f} at stake", "bad")}</div>' if value else ""))
            if slots and c2.button("Show on photo", key=f"show-{k}", width="stretch", icon=":material/center_focus_strong:"):
                st.session_state["focus"] = (k, slots)
                st.rerun()


def price_list(det, catalogue, price_issues, photo: Image.Image) -> None:
    if not det.labels:
        st.info("No price labels were read on this photo.")
        return
    notes = label_notes(price_issues)
    for lb in det.labels:
        text, key = notes.get(lb.label_id, (f"{lb.price:.2f}" if lb.price is not None else "?", "label"))
        item = catalogue.get(lb.sku) if lb.sku else None
        with st.container(border=True, key=f"card-label-{lb.label_id}"):
            c0, c1 = st.columns([1, 3], vertical_alignment="center")
            c0.image(photo.crop(tuple(int(v) for v in lb.bbox)).convert("RGB"), width="stretch")
            verdict = {"bad": "wrong price", "check": "check by eye", "label": "price is right"}[key]
            read = f"Label reads <b>{lb.price:.2f}</b>" if lb.price is not None else "Label could not be read"
            where = (f"{html.escape(item.name)} in {lb.slot_id} · price list {item.price:.2f}" if item
                     else "not under a product in the plan")
            c1.html(
                f'<div class="ss-big">{read}</div><div class="ss-muted">{where}</div>'
                f'<div style="margin-top:6px">{chip(verdict, key)}</div>'
            )
    st.caption("A price is called wrong only when the text readers are confident; otherwise the label is marked to check by eye.")


def shift_summary(report, cat_path: str) -> None:
    use_slm = st.toggle(
        "Written by a small language model (Qwen 2.5, 1.5B, on this laptop)",
        value=SLM_MODEL.exists() and os.environ.get("SHELFSENSE_SLM") != "0",  # tests set 0 to skip the model
        disabled=not SLM_MODEL.exists(),
    )
    with st.spinner("Writing the summary..."):
        briefing = cached_briefing(report.model_dump_json(), cat_path, use_slm)
    with st.container(border=True, key="card-summary"):
        st.markdown(f"<div class='ss-big'>{html.escape(briefing.summary)}</div>", unsafe_allow_html=True)
    if briefing.summary_source == "slm":
        st.html(chip(f"Written by the model in {briefing.seconds:.1f} s and checked: right slots, right problems, no invented numbers", "ok"))
    elif briefing.rejected_summary:
        st.html(chip("The model got a detail wrong, so the checked template is shown", "check"))
        st.caption(f"The model wrote: “{briefing.rejected_summary}”")
    st.caption("The to-do list is always built by rules. The model only writes these two lines, and a checker rejects them "
               "if they name a wrong slot, a wrong problem or a number that is not in the report.")


def details_tab(det, planogram, catalogue, report, stake) -> None:
    rows = []
    for shelf in planogram.shelves:
        for slot in shelf.slots:
            in_slot = [i for i in det.items if i.slot_id == slot.slot_id]
            labels = [lb for lb in det.labels if lb.slot_id == slot.slot_id]
            cat = catalogue.get(slot.sku)
            rows.append({
                "slot": slot.slot_id, "should hold": cat.name if cat else slot.sku, "planned": slot.facings,
                "seen": sum(i.sku == slot.sku for i in in_slot),
                "other products": ", ".join(sorted({i.sku or "unknown" for i in in_slot if i.sku != slot.sku})),
                "label price": labels[0].price if labels else None, "price list": cat.price if cat else None,
            })
    st.markdown("**Plan vs. shelf, slot by slot**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(f"**All {len(report.violations)} findings** (the to-do list merges related ones)")
    if report.violations:
        st.dataframe(
            pd.DataFrame([
                {"kind": v.kind.value.replace("_", " "), "slot": v.slot_id, "at stake": stake.get(v.violation_id),
                 "detail": v.detail, "action": v.recommended_action}
                for v in report.violations
            ]),
            hide_index=True, width="stretch",
        )
    d1, d2 = st.columns(2)
    d1.download_button("Download report (JSON)", report_to_json(report), f"{det.image_id}_report.json", "application/json", width="stretch")
    d2.download_button("Download findings (CSV)", violations_to_csv(report), f"{det.image_id}_findings.csv", "text/csv", width="stretch")
    with st.expander("Raw reading of the photo (JSON)"):
        st.json(det.model_dump())


# ---------- the app ----------

st.set_page_config(page_title="shelfSense", page_icon=":material/shelves:", layout="wide", initial_sidebar_state="collapsed")
st.html(CSS)
if "show-saved" in st.session_state:  # keep the switch's value on runs where it is not drawn
    st.session_state["show-saved"] = st.session_state["show-saved"]

bays = sorted(p.name for p in BAYS.glob("*") if (p / "detection_scenario.json").exists()) if BAYS.exists() else []
with st.sidebar:
    developer = st.toggle("Developer inputs", value=not bays, help="Mock shelves and custom file paths, for testing the report without photos.")
    if developer:
        preset = st.selectbox("Mock scenario", list(MOCKS))
        default_det, default_pg, default_cat = MOCKS[preset]
        with st.expander("Custom paths"):
            det_path = st.text_input("Detection output", "", placeholder=default_det) or default_det
            pg_path = st.text_input("Planogram", "", placeholder=default_pg) or default_pg
            cat_path = st.text_input("Catalogue", "", placeholder=default_cat) or default_cat

app_bar()
view = st.session_state.get("view", "home")
if developer:
    view_shelf(None, (det_path, pg_path, cat_path))
elif view == "scan":
    if st.button("← All shelves", key="back", type="tertiary"):
        go("home")
        st.rerun()
    runpy.run_path(str(ROOT / "app" / "views" / "availability.py"))
elif view == "shelf" and st.session_state.get("bay") in bays:
    view_shelf(st.session_state["bay"])
else:
    view_home(bays)
