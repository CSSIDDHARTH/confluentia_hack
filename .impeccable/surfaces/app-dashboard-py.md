---
version: 1
slug: "app-dashboard-py"
primary_target: "app/dashboard.py"
related_targets: ["app/views/availability.py"]
---

# Surface brief: shelfSense web app (app/dashboard.py, app/views/availability.py)

Scope: the whole web app (home, one shelf, check any photo). Visitor mode: Operate.
Audience: store staff on the shop floor first; managers; judges watching a projected laptop.
Job: know which shelf to fix first, then what to fix on it, and tick it off.
Constraints: every feature and number stays; Streamlit; offline; light theme (bright store lighting, projector).

## Direction contract

THESIS: The shelf is the interface. The photo and the store's plan sit side by side as one object, the same slots in the same colours, and the to-do list beside them answers "what do I fix first?". It refuses the category default of KPI tiles over a grid of cards and charts.

OWN-WORLD: Google's product language done precisely. Google Sans Flex for all text, Google Sans Code only for slot IDs and prices. A cool #f6f8fc ground, white sheets edged 1px #e1e3e1 at 16px radius, no resting shadows. Ink #1f1f1f, secondary #444746. Google blue #0b57d0 only on the one primary action and on selection. Status in Google's hues as pale containers with dark ink, always with a word. Material Symbols Rounded icons. A Lighthouse-style score gauge.

STORY: Staff see which shelf costs most, open it, press Check, watch each stage tick off, read the score and money, work down the to-dos (each one lights its slots on photo and plan), tick them off and watch the score climb.

FIRST VIEWPORT: Top bar: mark, shelfSense, where you are, How it works. Title row: aisle name at 28px with the store line; the plan switch; the primary "Check the shelf now" at the right. Below: left 55% the photo sheet (at most 70vh) with its tags; right 45% the score gauge, money at stake as the largest figure (32px), to-dos left; the plan strip (one row per shelf, slot tiles as wide as their facings); the first to-dos beneath. Home: title, one finding line, plan switch, "Check a photo"; one ranked list of shelves, worst first.

FORM: Google product screen (Gmail, Merchant Center, Lighthouse lineage), position 1 of 1, pinned by the brief. seed: none (direction pinned by the brief).

SIGNATURE: the linked shelf. Choosing a to-do lights its slots on the photo and the plan and fades the rest; the gauge sweeps once to the score when a check lands or a to-do is ticked.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
