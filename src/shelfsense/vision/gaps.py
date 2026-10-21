"""Phase 4 vision stage: empty shelf space and low stock from pack-level boxes.

Given one box per visible pack, this finds shelf rows and then looks for stretches of
a row that no pack covers but that are wide enough to hold one: those are empty
spaces. The number of packs that would fit estimates the missing facings, and a row's
fill ratio (share of the bay width covered by packs) flags low stock.

Everything here is geometry on boxes, so it is independent of which detector made them.
"""

from dataclasses import dataclass
from statistics import median

import numpy as np

# A stretch is an empty space when it is at least this many median pack widths wide.
MIN_GAP_WIDTHS = 0.8
# Rows whose fill ratio is below this are low stock.
LOW_FILL_RATIO = 0.6
# A height is a shelf edge when pack coverage there falls below this share of the typical row coverage.
ROW_VALLEY = 0.3
# Rows with fewer packs than this are too sparse to judge gaps from.
MIN_PACKS_PER_ROW = 2
# A "row" of at most this many boxes, each under STRIP_HEIGHT_RATIO of the bay's typical pack height, is a
# strip of price labels or a pack cut off by the photo's edge, not a shelf row. Seen on two of 24 unseen
# phone photos (a price-label strip under the products read as a row 38% and 47% stocked).
STRIP_MAX_BOXES = 3
STRIP_HEIGHT_RATIO = 0.5


@dataclass
class Row:
    index: int  # 0 = top row
    boxes: list[list[float]]  # packs, left to right
    y1: float
    y2: float
    fill_ratio: float = 1.0


@dataclass
class GapCandidate:
    row: int
    bbox: list[float]
    confidence: float
    estimated_missing: int


def cluster_rows(boxes: list[list[float]], bins: int = 400, valley: float = ROW_VALLEY) -> list[Row]:
    """Group pack boxes into shelf rows, top to bottom, each sorted left to right.

    Rows are found from a vertical profile: how much pack width covers each height. Shelf
    edges show up as dips in that profile, so a row is a stretch between dips. This copes
    with stacked packs and mixed heights, which fool grouping by individual box position.
    A dip is any height where coverage falls below `valley` times the typical row coverage.
    """
    if not boxes:
        return []
    y_min = min(b[1] for b in boxes)
    span = max(max(b[3] for b in boxes) - y_min, 1e-6)
    profile = np.zeros(bins)
    for b in boxes:
        lo = min(bins - 1, int((b[1] - y_min) / span * bins))
        hi = min(bins, int((b[3] - y_min) / span * bins) + 1)
        profile[lo:hi] += b[2] - b[0]
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    typical = float(np.percentile(smooth[smooth > 0], 75))
    high = smooth >= valley * typical

    runs: list[list[int]] = []  # [start, end] bins of each row band
    for i, h in enumerate(high):
        if h and runs and i - runs[-1][1] <= 2:  # bridge dips shorter than 3 bins
            runs[-1][1] = i
        elif h:
            runs.append([i, i])
    runs = [r for r in runs if r[1] - r[0] >= 1] or [[0, bins - 1]]

    groups: list[list[int]] = [[] for _ in runs]
    for idx, b in enumerate(boxes):
        c = min(bins - 1, int(((b[1] + b[3]) / 2 - y_min) / span * bins))
        k = min(range(len(runs)), key=lambda k: 0 if runs[k][0] <= c <= runs[k][1] else min(abs(c - runs[k][0]), abs(c - runs[k][1])))
        groups[k].append(idx)

    rows = []
    for g in groups:
        if not g:
            continue
        row_boxes = sorted((boxes[i] for i in g), key=lambda b: b[0])
        rows.append(Row(len(rows), row_boxes, median(b[1] for b in row_boxes), median(b[3] for b in row_boxes)))
    return rows


def drop_label_strips(rows: list[Row]) -> list[Row]:
    """Remove rows that are a few flat boxes (price labels, cut-off packs) under much taller packs."""
    heights = [b[3] - b[1] for r in rows for b in r.boxes]
    if len(rows) < 2 or not heights:
        return rows
    typical = median(heights)
    kept = [
        r for r in rows
        if not (len(r.boxes) <= STRIP_MAX_BOXES and median(b[3] - b[1] for b in r.boxes) < STRIP_HEIGHT_RATIO * typical)
    ]
    if not kept:
        return rows
    for k, r in enumerate(kept):
        r.index = k
    return kept


def find_gaps(
    boxes: list[list[float]], min_gap_widths: float = MIN_GAP_WIDTHS, valley: float = ROW_VALLEY
) -> tuple[list[Row], list[GapCandidate]]:
    """Return the shelf rows (with fill ratios set) and the empty spaces found in them.

    The bay's left and right edges are the outermost pack edges over all rows, so a row
    that stops short of them has a gap at its end.
    """
    rows = drop_label_strips(cluster_rows(boxes, valley=valley))
    if not rows:
        return [], []
    bay_left = min(r.boxes[0][0] for r in rows)
    bay_right = max(max(b[2] for b in r.boxes) for r in rows)
    bay_width = max(bay_right - bay_left, 1e-6)

    gaps: list[GapCandidate] = []
    for row in rows:
        w_med = median(b[2] - b[0] for b in row.boxes)
        covered, cursor = 0.0, bay_left
        spans = []  # (start, end) of uncovered stretches
        for b in row.boxes:
            if b[0] > cursor:
                spans.append((cursor, b[0]))
            covered += max(0.0, b[2] - max(b[0], cursor))
            cursor = max(cursor, b[2])
        if bay_right > cursor:
            spans.append((cursor, bay_right))
        row.fill_ratio = min(1.0, covered / bay_width)
        for start, end in spans:
            width = end - start
            if len(row.boxes) >= MIN_PACKS_PER_ROW and width >= min_gap_widths * w_med:
                gaps.append(
                    GapCandidate(
                        row=row.index,
                        bbox=[start, row.y1, end, row.y2],
                        confidence=round(min(0.95, 0.5 + 0.25 * (width / w_med - min_gap_widths)), 3),
                        estimated_missing=max(1, round(width / w_med)),
                    )
                )
    return rows, gaps


def low_stock_rows(rows: list[Row], threshold: float = LOW_FILL_RATIO) -> list[Row]:
    return [r for r in rows if r.fill_ratio < threshold]
