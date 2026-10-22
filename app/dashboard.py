"""shelfSense dashboard.

Home: today's shelves in one list, worst first (score, money at stake, to-dos), with one switch for the whole store:
the current plan or the new plan from head office. A shelf: check it live from its photo, see the photo and the plan
side by side in the same colours, and work down the to-do list beside them. "Check a photo": empty space and what to
refill on any photo, no plan needed (app/views/availability.py). Mock shelves and custom files are behind the
sidebar's "Developer inputs". The look (Google Sans, Google's colours) is set in .streamlit/config.toml and CSS below.

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
from shelfsense.report.compliance import WEIGHTS, ComplianceReport, ViolationKind, compute_report
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
    EDIT: "Change what any slot should hold and watch the check update. The photo stays the same.",
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
ICONS = {"ok": "check_circle", "low": "warning", "bad": "cancel", "check": "help", "label": "sell", "unread": "schedule"}
STAKE_HELP = ("Missing packs times their price, plus wrong prices times the packs they apply to. "
              "An estimate for putting work in order, not accounting.")
MOCK_PLANOGRAM = "contracts/planogram.example.json"
MOCK_CATALOGUE = "contracts/catalogue.example.csv"
MOCKS = {  # name -> (detection output, planogram, catalogue)
    "Mock shelf (as shipped)": ("contracts/detection_output.example.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Low-stock demo (all 5 violation types)": ("data/mock/mock_shelf_002_lowstock.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
    "Near-clean shelf (one unreadable label)": ("data/mock/mock_shelf_003_clean.json", MOCK_PLANOGRAM, MOCK_CATALOGUE),
}

CSS = """
<style>
@font-face {font-family: "SS Symbols"; font-style: normal; font-weight: 400; font-display: block;
  src: url("app/static/fonts/MaterialSymbolsRounded-subset.woff2") format("woff2");}
@property --p {syntax: "\\3C number>"; inherits: false; initial-value: 0;}  /* \\3C is the less-than sign, escaped: the HTML sanitiser drops a style block with tag-like text */
@keyframes ss-sweep {from {--p: 0;}}
:root {--ss-ground: #f6f8fc; --ss-sheet: #ffffff; --ss-edge: #e1e3e1; --ss-rule: #eceef1; --ss-ink: #1f1f1f;
  --ss-muted: #444746; --ss-faint: #5e5e5e; --ss-blue: #0b57d0; --ss-blue-soft: #d3e3fd; --ss-track: #e3e3e3;}
html {scrollbar-color: #c4c7c5 transparent; -webkit-font-smoothing: antialiased;}
::selection {background: var(--ss-blue-soft); color: #041e49;}
:focus-visible {outline: 2px solid var(--ss-blue) !important; outline-offset: 2px;}
.block-container {padding-top: 3.4rem; padding-bottom: 4rem; max-width: 1480px;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stMetricValue"], .ss-num {font-variant-numeric: tabular-nums;}
[data-testid="stMetricLabel"] p {color: var(--ss-muted); font-size: .875rem;}
div[class*="st-key-metric-todo"] [data-testid="stMetricValue"] {font-size: 1.5rem;}  /* money at stake leads; to-dos is its sibling */
/* sheets: white on the ground, one 1px edge, no resting shadow */
div[class*="st-key-sheet"] {background: var(--ss-sheet); border: 1px solid var(--ss-edge) !important;
  border-radius: 16px; box-shadow: none;}
div[class*="st-key-rowline"] {border-top: 1px solid var(--ss-rule); padding-top: 14px;}
div[class*="st-key-sheet-photo"] [data-testid="stFullScreenFrame"] > div,
div[class*="st-key-sheet-scanphoto"] [data-testid="stFullScreenFrame"] > div {width: 100%; display: flex; justify-content: center;}
div[class*="st-key-sheet-photo"] img, div[class*="st-key-sheet-scanphoto"] img {max-height: 70vh; width: auto !important;
  max-width: 100%; display: block; margin: 0 auto; object-fit: contain; border-radius: 4px;
  outline: 1px solid rgba(0,0,0,.1); outline-offset: -1px;}
div[class*="st-key-thumb"] img {border-radius: 8px; outline: 1px solid rgba(0,0,0,.1); outline-offset: -1px;}
.ss-i {font-family: "SS Symbols"; font-weight: normal; font-style: normal; font-size: 1.15em; line-height: 1;
  display: inline-block; vertical-align: -0.2em; letter-spacing: normal; text-transform: none; white-space: nowrap;
  font-feature-settings: "liga"; -webkit-font-smoothing: antialiased; font-variation-settings: "FILL" 1;}
.ss-bar {display: flex; align-items: center; gap: 12px;}
.ss-name {font-size: 1.375rem; font-weight: 500; color: var(--ss-ink); letter-spacing: -0.01em;}
.ss-where {font-size: .9375rem; color: var(--ss-muted); padding-left: 12px; margin-left: 4px; border-left: 1px solid var(--ss-edge);}
.ss-h1 {font-size: 1.75rem; line-height: 1.2; font-weight: 500; color: var(--ss-ink); margin: 0; letter-spacing: -0.015em; text-wrap: balance;}
.ss-lead {font-size: 1.0625rem; line-height: 1.5; color: var(--ss-ink); margin: 6px 0 0; text-wrap: pretty; max-width: 70ch;}
.ss-meta {font-size: .9375rem; color: var(--ss-muted); margin-top: 4px;}
.ss-help {font-size: .875rem; color: var(--ss-muted); margin-top: 2px; text-wrap: pretty;}
.ss-t2 {font-size: 1.125rem; font-weight: 500; color: var(--ss-ink);}
.ss-label {font-size: .875rem; color: var(--ss-muted);}
.ss-fig {font-size: 1.75rem; font-weight: 500; color: var(--ss-ink); font-variant-numeric: tabular-nums; line-height: 1.2;}
.ss-fig-s {font-size: 1.5rem; font-weight: 500; color: var(--ss-ink); font-variant-numeric: tabular-nums; line-height: 1.2;}
.ss-code {font-family: "Google Sans Code", monospace; font-variant-numeric: tabular-nums;}
.ss-chips {display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;}
.ss-chip {display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px 3px 8px; border-radius: 8px;
  font-size: .875rem; font-weight: 500; line-height: 1.3;}
.ss-ring {--p: 0; border-radius: 50%; display: grid; place-items: center; flex: none;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--ss-track) 0);
  animation: ss-sweep 900ms cubic-bezier(.2, 0, 0, 1) both;}
.ss-ring-in {width: 82%; height: 82%; border-radius: 50%; background: var(--ss-sheet); display: grid; place-items: center;
  font-weight: 500; font-variant-numeric: tabular-nums;}
.ss-gauge {display: flex; align-items: center; gap: 14px;}
@media (prefers-reduced-motion: reduce) {.ss-ring {animation: none;}}
.ss-shelf {margin-bottom: 12px;}
.ss-shelf-name {font-size: .8125rem; color: var(--ss-muted); margin: 0 0 6px 2px;}
.ss-row {display: flex; gap: 6px; padding-bottom: 6px;}
.ss-edge {height: 4px; background: #c4c7c5; border-radius: 2px;}  /* the shelf edge under each row of slots */
.ss-tile {min-width: 0; border-radius: 10px; padding: 8px 10px 9px; line-height: 1.3; overflow: hidden;
  transition-property: opacity; transition-duration: 200ms;}
.ss-tile-id {font-family: "Google Sans Code", monospace; font-size: .75rem; opacity: .85;}
.ss-tile-name {font-weight: 500; font-size: .9375rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.ss-tile-meta {font-size: .8125rem; font-variant-numeric: tabular-nums;}
.ss-tile-status {font-size: .8125rem; font-weight: 500; margin-top: 3px; display: flex; align-items: center; gap: 4px;}
.ss-todo {font-size: 1rem; line-height: 1.5; color: var(--ss-ink); text-wrap: pretty;}
.ss-todo b {font-variant-numeric: tabular-nums;}
.ss-done {text-decoration: line-through; color: var(--ss-faint);}
.ss-legend {display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; font-size: .875rem; color: var(--ss-muted); margin-top: 6px;}
.ss-summary {font-size: 1.125rem; line-height: 1.55; color: var(--ss-ink); max-width: 70ch; text-wrap: pretty;}
</style>
"""

LOGO = '<img src="app/static/logo.svg" width="32" height="32" alt="">'  # served from app/static (st.html strips inline SVG)


# ---------- small pieces of HTML ----------

def icon(name: str) -> str:
    return f'<span class="ss-i" aria-hidden="true">{name}</span>'


def ring(score: float, size: int = 96) -> str:
    """A score gauge in the style of Google's Lighthouse: green from 90, orange from 50, red below. It sweeps to its
    value once when drawn (not with reduced motion)."""
    key = "ok" if score >= 90 else "low" if score >= 50 else "bad"
    return (
        f'<div class="ss-ring" role="img" aria-label="Shelf score {score:.0f} out of 100" data-score="{score:.0f}" '
        f'style="--p:{score:.0f};--c:{STATUS[key]};width:{size}px;height:{size}px">'
        f'<div class="ss-ring-in" style="color:{STATUS[key]};font-size:{size * 0.34:.0f}px">{score:.0f}</div></div>'
    )


def chip(text: str, key: str, with_icon: bool = True) -> str:
    bg, fg = TONAL[key]
    return (f'<span class="ss-chip" style="background:{bg};color:{fg}">'
            f'{icon(ICONS[key]) if with_icon else ""}{html.escape(text)}</span>')


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def problem_chips(report: ComplianceReport) -> list[str]:
    """What is wrong on a shelf, in a few words each, worst first."""
    status = slot_status(report)
    keys = Counter(words for _, words in status.values())
    out = []
    if keys["wrong product"]:
        out.append(chip(plural(keys["wrong product"], "wrong product"), "bad"))
    if keys["missing"]:
        out.append(chip(plural(keys["missing"], "product") + " missing", "bad"))
    wrong_prices = report.counts.get(ViolationKind.PRICE_MISMATCH.value, 0)
    if wrong_prices:
        out.append(chip(plural(wrong_prices, "wrong price"), "bad"))
    if keys["low stock"]:
        out.append(chip(plural(keys["low stock"], "slot") + " low", "low"))
    to_check = keys["check product"] + report.counts.get(ViolationKind.UNREADABLE_PRICE.value, 0) \
        + report.counts.get(ViolationKind.UNKNOWN_PRODUCT.value, 0)
    if to_check:
        out.append(chip(f"{to_check} to check by eye", "check"))
    return out or [chip("All as planned", "ok")]


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
        if v.kind == ViolationKind.PRICE_MISMATCH:
            price_note[v.slot_id] = "price label wrong"
        elif v.kind == ViolationKind.UNREADABLE_PRICE:
            price_note.setdefault(v.slot_id, "check the price label")
    shelves = []
    for shelf in planogram.shelves:
        tiles = []
        for slot in shelf.slots:
            item = catalogue.get(slot.sku)
            name = html.escape(item.name if item else slot.sku)
            is_done = bool(report is not None and done and slot.slot_id in done and slot.slot_id not in status)
            key, words = ("unread", "not checked yet") if report is None else status.get(
                slot.slot_id, ("ok", "done" if is_done else "as planned")
            )
            bg, fg = TONAL[key]
            seen = "" if det is None or report is None else (
                f" · {sum(i.slot_id == slot.slot_id and i.sku == slot.sku for i in det.items)} seen"
            )
            note = f'<div class="ss-tile-meta">{price_note[slot.slot_id]}</div>' if slot.slot_id in price_note else ""
            faded = "opacity:.28;" if highlight and slot.slot_id not in highlight else ""
            tiles.append(
                f'<div class="ss-tile" data-slot="{slot.slot_id}" data-status="{"done" if is_done else key}" '
                f'style="flex:{slot.facings} 1 0;background:{bg};color:{fg};{faded}">'
                f'<div class="ss-tile-id">{slot.slot_id}</div><div class="ss-tile-name">{name}</div>'
                f'<div class="ss-tile-meta">{slot.facings} planned{seen}</div>'
                f'<div class="ss-tile-status">{icon(ICONS[key]) if key != "unread" else ""}{words}</div>{note}</div>'
            )
        shelves.append(
            f'<div class="ss-shelf"><div class="ss-shelf-name">Shelf {shelf.index + 1}</div>'
            f'<div class="ss-row">{"".join(tiles)}</div><div class="ss-edge"></div></div>'
        )
    return '<div role="group" aria-label="The plan for this shelf">' + "".join(shelves) + "</div>"


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
    """Score, money at stake, to-dos and what is wrong, from the saved reading, for the home list."""
    det_p, pg_p, cat_p = bay_files(bay, plan)
    det, pg, cat = load_detection_output(ROOT / det_p), load_planogram(ROOT / pg_p), load_catalogue(ROOT / cat_p)
    report = compute_report(det, pg, cat, group_items=True)
    expected = {s.slot_id: s.sku for sh in pg.shelves for s in sh.slots}
    im = draw_audit(Image.open(ROOT / det.image_path), det, report, expected, label_notes(check_prices(det, cat)))
    return {
        "score": report.score, "money": sum(at_stake(report, det, pg, cat).values()), "currency": currency(cat),
        "todo": len(plan_steps(report, cat, pg)), "chips": problem_chips(report),
        "image": ImageOps.fit(im, (480, 360), centering=(0.5, 0.35)),  # every thumbnail the same shape
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


def app_bar(where: str) -> None:
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.html(f'<div class="ss-bar">{LOGO}<span class="ss-name">shelfSense</span>'
              f'<span class="ss-where">{html.escape(where)}</span></div>')
    with right.popover("How it works", icon=":material/info:", width="stretch"):
        st.markdown(
            "**One photo of a shelf. Every problem on it. What to fix first.**\n\n"
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
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    with right:
        plan = plan_switch("plan-home", with_edit=False)
        st.html(f'<div class="ss-help">{PLAN_HELP[plan]}</div>')
    summaries = {b: bay_summary(b, plan) for b in bays}
    order = sorted(bays, key=lambda b: (-summaries[b]["money"], summaries[b]["score"]))
    cur = summaries[order[0]]["currency"] if order else ""
    total, todos = sum(s["money"] for s in summaries.values()), sum(s["todo"] for s in summaries.values())
    if not order or (total == 0 and todos == 0):
        finding = "Every shelf matches its plan."
    else:
        first = summaries[order[0]]
        finding = (f"<b>{html.escape(bay_title(order[0])[0])}</b> needs you first: {cur} {first['money']:.2f} at stake, "
                   f"{plural(first['todo'], 'to-do')}. Across {plural(len(bays), 'shelf', 'shelves')}: "
                   f'{cur} {total:.2f} at stake and {plural(todos, "to-do")}.')
    left.html(f'<h1 class="ss-h1">Today\'s shelf check</h1><p class="ss-lead">{finding}</p>')

    with st.container(border=True, key="sheet-list"):
        for i, bay in enumerate(order):
            s = summaries[bay]
            aisle, where = bay_title(bay)
            with st.container(key=f"{'rowtop' if i == 0 else 'rowline'}-{bay}"):
                c_img, c_info, c_score, c_money, c_todo, c_go = st.columns([1.5, 3.2, 1.1, 1.5, 0.9, 1.2], vertical_alignment="center")
                with c_img.container(key=f"thumb-{bay}"):
                    st.image(s["image"], width="stretch")
                c_info.html(f'<div class="ss-t2">{html.escape(aisle)}</div><div class="ss-meta">{where}</div>'
                            f'<div class="ss-chips">{"".join(s["chips"])}</div>')
                c_score.html(ring(s["score"], 68))
                c_money.html(f'<div class="ss-label">At stake</div><div class="ss-fig ss-num">{s["currency"]} {s["money"]:.2f}</div>')
                c_todo.html(f'<div class="ss-label">To-dos</div><div class="ss-fig-s ss-num">{s["todo"]}</div>')
                if c_go.button("Open", key=f"open-{bay}", type="primary" if i == 0 else "secondary",
                               icon=":material/arrow_forward:", icon_position="right", width="stretch"):
                    go("shelf", bay)
                    st.rerun()

    with st.container(border=True, key="sheet-scan"):
        a, b = st.columns([5, 1.2], vertical_alignment="center")
        a.html('<div class="ss-t2">No plan for a shelf?</div>'
               '<div class="ss-meta">Check any shop photo for empty space and get a refill list.</div>')
        if b.button("Check a photo", key="open-scan", icon=":material/photo_camera:", width="stretch"):
            go("scan")
            st.rerun()


# ---------- one shelf ----------

def view_shelf(bay: str | None, files: tuple[str, str, str] | None = None) -> None:
    """A demo bay (bay) or, in developer mode, any files (detection, planogram, catalogue)."""
    run = False
    if bay is not None:
        if st.button("All shelves", key="back", type="tertiary", icon=":material/arrow_back:"):
            go("home")
            st.rerun()
        aisle, where = bay_title(bay)
        c_title, c_plan, c_check = st.columns([2.2, 2.4, 1.4], vertical_alignment="bottom")
        c_title.html(f'<h1 class="ss-h1">{html.escape(aisle)}</h1>'
                     f'<div class="ss-meta">{where} · a real supermarket photo from a store the models never saw</div>')
        with c_plan:
            plan = plan_switch("plan-shelf", with_edit=True)
        det_path, pg_path, cat_path = bay_files(bay, plan)
    else:
        plan, (det_path, pg_path, cat_path) = "compliant", files
    det = load_detection_output(ROOT / det_path)
    planogram = load_planogram(ROOT / pg_path)
    catalogue = load_catalogue(ROOT / cat_path)
    base_score = None

    if bay is not None:
        photo, _ = live.bay_photo(BAYS / bay)
        with c_plan:
            st.html(f'<div class="ss-help">{PLAN_HELP[plan]}</div>')
        if plan == EDIT:
            planogram, base_score, det = plan_editor(bay, planogram, catalogue, det)
        live_key = f"live-{bay}-{plan}-{hashlib.md5(planogram.model_dump_json().encode()).hexdigest()[:10]}"
        result = st.session_state.get(live_key)
        with c_check:
            if not live.available():
                st.html('<div class="ss-help">Checking live needs the product index and the detectors in models/; '
                        "this is the saved check of the photo.</div>")
            else:
                run = st.button("Check the shelf now", key=f"run-{live_key}", type="primary",
                                icon=":material/photo_camera:", width="stretch")
                if result is not None:
                    st.html(chip(f"Checked live in {result.total:.1f} s on this laptop's "
                                 f"{'GPU' if result.device == 'cuda' else 'CPU'}", "ok"))
                elif plan != EDIT:
                    st.toggle("Show the saved check", key="show-saved",
                              help="The result of an earlier check of the same photo; checking now gives the same result.")
        if run:
            with st.status("Checking the shelf...", expanded=True) as status:
                def on_stage(stage: str, seconds: float) -> None:
                    status.write(f":material/check_circle: {STAGES.get(stage, stage)} · {seconds:.2f} s")

                result = load_chain().run(
                    photo, planogram, read_prices=live.prices_available(), on_stage=on_stage,
                    shown_path=str(photo.relative_to(ROOT)) if photo.is_relative_to(ROOT) else photo.name,
                )
                st.session_state[live_key] = result
                status.update(label=f"Checked in {result.total:.1f} s", state="complete", expanded=False)
            st.rerun()
        if result is not None:
            det = result.det
        elif plan != EDIT and live.available() and not st.session_state.get("show-saved"):
            left, right = st.columns([1.2, 1], gap="medium")
            with left.container(border=True, key="sheet-photo"):
                st.image(shown(Image.open(BAYS / bay / "photo.jpg")), width="stretch")
            with right.container(border=True, key="sheet-plan"):
                st.html('<div class="ss-t2">The plan for this shelf</div>')
                st.html(plan_html(planogram, catalogue, None, None))
                st.html('<div class="ss-help">Press <b>Check the shelf now</b> to compare the photo with it.</div>')
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

    left, right = st.columns([1.2, 1], gap="medium")
    with left.container(border=True, key="sheet-photo"):
        photo_path = ROOT / det.image_path
        base = Image.open(photo_path) if photo_path.exists() else Image.new("RGB", (det.image_width, det.image_height), "white")
        expected = {s.slot_id: s.sku for sh in planogram.shelves for s in sh.slots}
        st.image(shown(draw_audit(base, det, report, expected, label_notes(price_issues), focus_slots)), width="stretch")
        st.html(
            f'<div class="ss-legend">{chip("right product", "ok")}{chip("wrong product", "bad")}'
            f'{chip("not recognised", "check")}{chip("price label", "label")}<span>Red fill: empty space</span>'
            + ("<span>The photo shows the shelf as it was checked.</span>" if done_steps else "") + "</div>"
        )
    with right:
        with st.container(border=True, key="sheet-score"):
            g, m1, m2 = st.columns([1.25, 1.35, 0.9], vertical_alignment="center")
            change = "" if base_score is None else f"{report.score - base_score:+.0f} vs current plan"
            if done_steps:
                change = (change + " · " if change else "") + \
                    f"checked {report.score:.0f}, {after.score - report.score:+.0f} with {len(done_steps)} done"
            g.html(f'<div class="ss-gauge">{ring(after.score, 84)}<div><div class="ss-label">Shelf score</div>'
                   f'<div class="ss-help">{change or "out of 100"}</div></div></div>')
            with m1.container(key="metric-money"):
                st.metric("Money at stake", f"{money} {sum(stake.values()) - fixed:.2f}".strip(),
                          delta=f"-{(money + ' ') if money else ''}{fixed:.2f} done" if fixed else None,
                          delta_color="inverse", help=STAKE_HELP)
            with m2.container(key="metric-todo"):
                st.metric("To-dos", len(steps) - len(done_steps),
                          delta=f"-{len(done_steps)} done" if done_steps else None, delta_color="inverse")
        with st.container(border=True, key="sheet-plan"):
            st.html('<div class="ss-t2">The plan for this shelf</div>')
            st.html(plan_html(planogram, catalogue, det, after, focus_slots, done_slots))
        with st.container(border=True, key="sheet-todos"):
            todo_list(steps, report, stake, money, done_key, fixed)

    prices, summary, details = st.tabs(["Prices", "Shift summary", "Details"])
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
    with st.container(border=True, key="sheet-edit"):
        st.html('<div class="ss-t2">Change the plan</div>')
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
        if c3.button("Swap their products", disabled=a == b, icon=":material/swap_horiz:", width="stretch"):
            st.session_state[rows_key] = swap_products(rows, a, b)
            st.session_state[ver_key] += 1
            st.rerun()
        if c4.button("Reset to the shelf's plan", icon=":material/restart_alt:", width="stretch"):
            st.session_state[rows_key] = original
            st.session_state[ver_key] += 1
            st.rerun()
        planogram = apply_rows(shelf_plan, rows)
        edits = changes(original, rows)
        st.html(f'<div class="ss-help">{plural(len(edits), "change")}'
                + (": " + html.escape("; ".join(edits)) if edits else ". The photo stays the same.") + "</div>")
    # labels sit under the same slots, which now expect the edited products (a swap leaves each label under the other product)
    det = det.model_copy(update={"labels": assign_labels_to_slots(det.labels, det, planogram)})
    base_score = compute_report(
        det.model_copy(update={"labels": assign_labels_to_slots(det.labels, det, shelf_plan)}), shelf_plan, catalogue, group_items=True
    ).score
    return planogram, base_score, det


def todo_list(steps, report, stake, money, done_key: str, fixed: float) -> None:
    st.html('<div class="ss-t2">To-do list</div>')
    if not steps:
        st.success("Nothing to do: the shelf matches the plan.", icon=":material/check_circle:")
        return
    n_done = sum(bool(st.session_state.get(f"{done_key}-{k}")) for k in range(1, len(steps) + 1))
    total = sum(stake.values())
    st.progress(n_done / len(steps), text=f"{n_done} of {len(steps)} done" + (f" · {money} {fixed:.2f} of {total:.2f} sorted" if total else ""))
    if n_done == len(steps):
        st.success("All done. Check the shelf again with a new photo to confirm it matches the plan.", icon=":material/task_alt:")
    slots_of = {v.violation_id: v.slot_id for v in report.violations}
    focus = st.session_state.get("focus")
    if focus and st.button(f"Showing to-do {focus[0]} on the photo and plan. Show everything", key="focus-clear",
                           type="tertiary", icon=":material/close:"):
        st.session_state.pop("focus", None)
        st.rerun()
    for k, step in enumerate(steps, 1):
        value = sum(stake.get(v, 0.0) for v in step.violation_ids)
        slots = sorted({slots_of[v] for v in step.violation_ids if slots_of.get(v)})
        with st.container(key=f"{'rowtop' if k == 1 else 'rowline'}-todo-{k}"):
            c0, c1, c2 = st.columns([0.35, 6, 1.5], vertical_alignment="top")
            done = c0.checkbox("Done", key=f"{done_key}-{k}", label_visibility="collapsed")
            c1.html(f'<div class="ss-todo{" ss-done" if done else ""}"><b>{k}.</b> {html.escape(step.text)}</div>'
                    + (f'<div class="ss-chips">{chip(f"{money} {value:.2f} at stake", "bad", with_icon=False)}</div>' if value else ""))
            if slots and c2.button("Show", key=f"show-{k}", type="tertiary", icon=":material/center_focus_strong:",
                                   help="Show this to-do's slots on the photo and the plan"):
                st.session_state["focus"] = (k, slots)
                st.rerun()


def price_list(det, catalogue, price_issues, photo: Image.Image) -> None:
    if not det.labels:
        st.info("No price labels were read on this photo.", icon=":material/info:")
        return
    notes = label_notes(price_issues)
    with st.container(border=True, key="sheet-prices"):
        for i, lb in enumerate(det.labels):
            text, key = notes.get(lb.label_id, (f"{lb.price:.2f}" if lb.price is not None else "?", "label"))
            item = catalogue.get(lb.sku) if lb.sku else None
            with st.container(key=f"{'rowtop' if i == 0 else 'rowline'}-label-{lb.label_id}"):
                c0, c1, c2 = st.columns([1.1, 3, 1.4], vertical_alignment="center")
                with c0.container(key=f"thumb-label-{lb.label_id}"):
                    st.image(photo.crop(tuple(int(v) for v in lb.bbox)).convert("RGB"), width="stretch")
                read = (f'Label reads <span class="ss-code">{lb.price:.2f}</span>' if lb.price is not None
                        else "The label could not be read")
                where = (f'{html.escape(item.name)} in <span class="ss-code">{lb.slot_id}</span> · price list '
                         f'<span class="ss-code">{item.price:.2f}</span>' if item else "Not under a product in the plan")
                c1.html(f'<div class="ss-t2">{read}</div><div class="ss-meta">{where}</div>')
                verdict = {"bad": "Wrong price", "check": "Check by eye", "label": "Price is right"}[key]
                c2.html(chip(verdict, key))
    st.html('<div class="ss-help">A price is called wrong only when the text readers are confident; '
            "otherwise the label is marked to check by eye.</div>")


def shift_summary(report, cat_path: str) -> None:
    use_slm = st.toggle(
        "Written by a small language model (Qwen 2.5, 1.5B, on this laptop)",
        value=SLM_MODEL.exists() and os.environ.get("SHELFSENSE_SLM") != "0",  # tests set 0 to skip the model
        disabled=not SLM_MODEL.exists(),
    )
    with st.spinner("Writing the summary..."):
        briefing = cached_briefing(report.model_dump_json(), cat_path, use_slm)
    with st.container(border=True, key="sheet-summary"):
        st.html(f'<p class="ss-summary">{html.escape(briefing.summary)}</p>')
        if briefing.summary_source == "slm":
            st.html(chip(f"Written by the model in {briefing.seconds:.1f} s and checked: right slots, right problems, "
                         "no invented numbers", "ok"))
        elif briefing.rejected_summary:
            st.html(chip("The model got a detail wrong, so the checked template is shown", "check"))
            st.html(f'<div class="ss-help">The model wrote: “{html.escape(briefing.rejected_summary)}”</div>')
    st.html('<div class="ss-help">The to-do list is always built by rules. The model only writes these two lines, and a '
            "checker rejects them if they name a wrong slot, a wrong problem or a number that is not in the report.</div>")


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
    d1.download_button("Download report (JSON)", report_to_json(report), f"{det.image_id}_report.json", "application/json",
                       icon=":material/download:", width="stretch")
    d2.download_button("Download findings (CSV)", violations_to_csv(report), f"{det.image_id}_findings.csv", "text/csv",
                       icon=":material/download:", width="stretch")
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

view = st.session_state.get("view", "home")
if developer:
    app_bar("Developer inputs")
    view_shelf(None, (det_path, pg_path, cat_path))
elif view == "scan":
    app_bar("Check a photo")
    if st.button("All shelves", key="back", type="tertiary", icon=":material/arrow_back:"):
        go("home")
        st.rerun()
    runpy.run_path(str(ROOT / "app" / "views" / "availability.py"))
elif view == "shelf" and st.session_state.get("bay") in bays:
    app_bar("Today's shelf check")
    view_shelf(st.session_state["bay"])
else:
    app_bar("Today's shelf check")
    view_home(bays)
