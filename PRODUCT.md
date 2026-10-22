# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Store staff on the shop floor first: an associate or shift lead with a tablet, laptop or phone who checks a shelf bay, gets a short list of fixes and works through it. Store and area managers use the same screens to see which shelf needs attention first. For the hackathon, judges watch the product on a laptop that is screen-shared or projected (about 1080p), so type and contrast must hold at that distance.

## Product Purpose

shelfSense checks one photo of a shelf bay against the store's planogram and price list: right product, right place, right count, right price, nothing empty. It turns what it finds into a compliance score, the money at stake, and a short ordered to-do list for staff. Success: a person standing at the shelf knows within seconds what to fix first and can tick it off.

## Positioning

Everything comes from one photo and runs on the laptop itself (about 2 s with a GPU, under 5 s on four CPU cores, no cloud). Rules decide every finding; a small language model (Qwen 2.5, 1.5B) only writes a two-line summary, and a checker rejects it if it is wrong. Results are measured on photos the models never saw, and the limits are stated.

## Operating Context

Shop floor under bright store lighting; one shelf bay at a time; a phone or tablet photo taken straight on from 1-2 m. A store's planogram tells which product belongs in which slot and how many facings; the price list gives each product's price. Staff work through fixes in order and re-photograph the shelf to confirm. The demo runs offline from a laptop (`streamlit run app/dashboard.py`).

## Capabilities and Constraints

- Shelf check against a plan: product blocks found and named, aligned to the plan's slots, packs counted, empty space found, price labels read with two OCR engines; violations scored; money at stake estimated; violations merged into to-dos; live check with timings; "Try your own plan" edits the plan.
- Check any photo with no plan: empty space, row fill, refill list.
- Built in Streamlit (Python); the look is set by `.streamlit/config.toml`, injected CSS and HTML fragments; photos are annotated server-side with PIL.
- Every feature and every number must stay as it is (confirmed); layout, look and wording may change.
- Terminology: shelf (bay), shelf row (S1 = top), slot (S1-02), facing (one visible pack), planogram ("plan"), price list ("catalogue" in code), to-do (a merged fix).

## Brand Commitments

- Name: shelfSense. Tagline in use: "One photo of a shelf. Every problem on it. What to fix first." (may be reworded).
- The user asked for a premium Google style and Google Fonts: Google's own type and Material conventions.

## Evidence on Hand

- Three demo shelves (held-out Grozi photos, `data/demo_bays/`), each with a current plan and a planted "new plan from head office"; eight availability photos (`data/availability_demo/`).
- Measured results in `docs/WRITEUP.md` and `README.md` (median 53 over 35 held-out shelves; empty-space F1 0.64; label reading 29-30 of 31; timings).
- Absences that must not be invented: real store planograms, real prices (demo prices come from the labels), customers, sales rates.

## Product Principles

1. The shelf is the interface: what the photo shows and what the plan expects are read as one object.
2. Answer "what do I fix first" before anything else; detail is one step away, never in the way.
3. Never claim more than was measured: unsure reads say "check by eye", estimates are labelled.
4. Fast and offline at the shelf; nothing waits on a network.

## Accessibility & Inclusion

Readable on a projector and under store lighting: text contrast at least 4.5:1, nothing under 12 px, colour never the only signal (every status also has a word).
