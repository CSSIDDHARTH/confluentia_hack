---
name: shelfSense
description: One photo of a shelf. Every problem on it. What to fix first.
colors:
  ground: "#f6f8fc"
  sheet: "#ffffff"
  edge: "#e1e3e1"
  rule: "#eceef1"
  control-edge: "#c4c7c5"
  ink: "#1f1f1f"
  muted: "#444746"
  faint: "#5e5e5e"
  primary: "#0b57d0"
  primary-soft: "#d3e3fd"
  widget-fill: "#eef2f9"
  gauge-track: "#e3e3e3"
  status-ok: "#146c2e"
  status-ok-container: "#c4eed0"
  status-ok-ink: "#0a3818"
  status-low: "#a85300"
  status-low-container: "#ffdcbe"
  status-low-ink: "#5a2800"
  status-bad: "#b3261e"
  status-bad-container: "#f9dedc"
  status-bad-ink: "#8c1d18"
  status-check: "#f9ab00"
  status-check-container: "#ffefc4"
  status-check-ink: "#4d3800"
  status-label: "#0b57d0"
  status-label-container: "#d3e3fd"
  status-label-ink: "#041e49"
  status-unread-container: "#eef0f4"
typography:
  page-title:
    fontFamily: "Google Sans Flex, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: "-0.015em"
  lead:
    fontFamily: "Google Sans Flex, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 400
    lineHeight: 1.5
  section-title:
    fontFamily: "Google Sans Flex, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 500
  body:
    fontFamily: "Google Sans Flex, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  meta:
    fontFamily: "Google Sans Flex, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
  figure-lead:
    fontFamily: "Google Sans Flex, sans-serif"
    fontSize: "2rem"
    fontWeight: 500
  figure:
    fontFamily: "Google Sans Flex, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 500
  code:
    fontFamily: "Google Sans Code, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
rounded:
  photo: "4px"
  thumb: "8px"
  chip: "8px"
  tile: "10px"
  control: "12px"
  sheet: "16px"
  button: "999px"
spacing:
  xs: "4px"
  sm: "6px"
  md: "12px"
  lg: "16px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.sheet}"
    rounded: "{rounded.button}"
  button-secondary:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.button}"
  sheet:
    backgroundColor: "{colors.sheet}"
    rounded: "{rounded.sheet}"
    padding: "16px"
  chip-bad:
    backgroundColor: "{colors.status-bad-container}"
    textColor: "{colors.status-bad-ink}"
    rounded: "{rounded.chip}"
    padding: "3px 10px 3px 8px"
  plan-tile-bad:
    backgroundColor: "{colors.status-bad-container}"
    textColor: "{colors.status-bad-ink}"
    rounded: "{rounded.tile}"
    padding: "8px 10px 9px"
---

# Design System: shelfSense

## Overview

Google's product language, applied to one job: telling store staff what to fix on a shelf first. The shelf is the interface. The photo and the store's plan sit side by side as one object, the same slots in the same colours, and the to-do list sits beside them. It is a working screen (Operate): type, palette, density and one signature move come from the world; layout and controls are the platform's standard ones (Streamlit segmented controls, buttons, tabs, checkboxes).

Light, because the scene is a shop floor under bright store lighting and a laptop projected or screen-shared for judges. Nothing is decorative: every colour on the page is a status, the one main action, or the selection.

Source of truth: `.streamlit/config.toml` (theme, fonts), the `CSS` block in `app/dashboard.py`, and `STATUS` / `TONAL` in `src/shelfsense/vision/render.py`, which colour both the page and the tags drawn on the photos.

## Colors

### Primary
Google blue `#0b57d0`, only on the one main action of a screen (the first shelf's "Open", "Check the shelf now"), on the selected plan, on focus rings and text selection (`#d3e3fd`).

### Neutral
A cool ground `#f6f8fc` with white sheets edged `#e1e3e1`; row rules `#eceef1`; controls `#c4c7c5`. Ink `#1f1f1f`, secondary text `#444746`, faint `#5e5e5e`; every text tier holds 4.5:1 on white.

### Status
Google's hues, each as a solid (tags on the photo, the gauge arc) and a pale container with dark ink of the same hue (chips, plan tiles): ok green `#146c2e`, low stock orange `#a85300`, wrong or missing red `#b3261e`, check-by-eye yellow `#f9ab00` (dark text on it), price label blue `#0b57d0`, not yet checked grey.

### Named Rules
- **Colour carries a status or the action.** No colour on chrome, no tinted nav, no brand fill on every button.
- **Never colour alone.** Every status also has an icon and a word ("wrong product", "low stock", "check by eye").
- **One palette for page and photo.** The photo tags and pack outlines use the same status keys as the plan tiles and chips.

## Typography

Google Sans Flex for everything; Google Sans Code only for slot IDs and prices, where character-for-character reading matters. Both from Google Fonts, served from `app/static/fonts` so the demo works offline. Tabular figures (`tnum`) on every figure that changes; not inside running sentences.

### Hierarchy
Page title 28px/500 (an app title, never landing-page size); lead/finding line 17px; section titles 18px/500; body and to-dos 16px; meta and help 14px in `#444746`; the lead figure (money at stake) 32px/500; sibling figures 24px; plan tile text 13-15px; nothing under 12px.

### Named Rules
- **The answer figure leads.** On a shelf, money at stake is the largest figure; to-dos is its sibling at 24px.

## Layout

A 1480px-wide canvas on the ground, one gutter. Home: a title row (title and the finding sentence left; the plan switch and its explanation right), then one sheet listing every shelf, worst first, as rows (thumbnail, name and problem chips, score gauge, money, to-dos, open), then a sheet for "Check a photo". Shelf: back link; title row (aisle and store, plan switch, main action); then 55/45 columns, photo sheet left (photo capped at 70vh, centred), right the score sheet, the plan sheet and the to-do sheet; then tabs for Prices, Shift summary and Details. Below about 640px everything stacks in reading order.

## Elevation & Depth

Flat. Sheets are separated from the ground by a tone step and a 1px edge, never a shadow. Only what floats (the "How it works" popover, menus) carries Streamlit's own shadow.

## Shapes

Sheets 16px, plan tiles 10px, chips and thumbnails 8px, photos 4px (concentric inside the sheet's padding), widgets 12px, buttons full pills. The shelf edge under each row of plan tiles is a 4px bar, the plan's one drawn line.

## Components

### Buttons
Pills. Primary (blue fill, white label) once per screen; secondary (white, edged); tertiary (text only) for "All shelves", "Show" and closing a focus. Material Symbols Rounded icons from Streamlit's set.

### Chips
Pale status container, dark ink of the same hue, a filled Material Symbols icon, 8px radius, 14px/500 text. Used for problems, verdicts and the legend.

### Cards / Containers
"Sheets": white, 1px edge, 16px radius, no shadow. Rows inside a sheet are divided by a 1px rule, never boxed.

### Navigation
A plain top bar: logo, "shelfSense", where you are (after a 1px divider), "How it works" on the right. No sidebar in the main flow (developer inputs live in the collapsed sidebar).

### Score gauge (signature)
A Lighthouse-style ring: green from 90, orange from 50, red below, the number in the status colour at the centre, labelled "Shelf score" for screen readers. It sweeps once to its value when drawn (900ms, cubic-bezier(.2,0,0,1)); no motion when reduced motion is requested.

### Plan strip (signature)
The planogram drawn as the shelf: one row per shelf, top first, one tile per slot as wide as its planned facings, a 4px shelf edge under each row. A tile shows its slot ID (code face), product, planned and seen counts, and its status with icon and word; "done" after the to-do is ticked. With a to-do in focus, every other tile and the photo fade, and only that to-do's slots stay bright (the linked shelf).

## Do's and Don'ts

### Do:
- Put the screen's answer first: on home the shelf to fix first, on a shelf its score, money and to-dos next to the photo.
- Keep one main action per screen in blue; everything else secondary or tertiary.
- Give every status an icon and a word, and use the same status key on the page and on the photo.
- Use tabular figures for figures that change; Google Sans Code for slot IDs and prices.
- Escape `<` as `\3C` inside injected CSS: Streamlit's HTML sanitiser drops a style block with tag-like text.

### Don't:
- No KPI tile rows, no grid of identical cards as page structure, no eyebrows above headings.
- No shadows on resting sheets, no border plus shadow, no thick coloured side bars on tiles or cards.
- No gradients, glass or glows; no colour on chrome.
- No emoji or Unicode glyphs as icons (math signs such as ≠ in a price tag are content, not icons).
- No figure above 32px; no page title above 32px.
