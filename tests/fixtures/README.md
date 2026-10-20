# tests/fixtures/

Hand-edited detection outputs for scenarios the canonical
`contracts/detection_output.example.json` doesn't cover. These are not
part of the shared data contracts (that's `contracts/`); they exist only
to exercise `shelfsense.report`/`shelfsense.pricing` logic and the
dashboard, so Person B can edit them freely.

They still reuse `planogram.example.json` and `catalogue.example.csv`
(the SKUs, slot IDs and image dimensions must stay consistent with
those, or `test_contracts`-style checks on them will fail).

- `detection_output_lowstock.json` — same as the canonical example
  (same planted stock-out, misplaced item, unrecognised item and price
  mismatch), but with slot S1-01's matching facings dropped from 3 to 1
  (removes items i02 and i03), which is below its `min_facings` of 2.
  The canonical example's own partially-stocked slot (S1-03) lands
  exactly at `min_facings`, not below it, so it never triggers
  low-stock under assumption 5's strict `<` rule — this fixture is the
  one place that category is actually exercised end-to-end.
