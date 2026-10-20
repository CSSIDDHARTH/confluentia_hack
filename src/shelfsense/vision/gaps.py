"""Phase 4 vision stage: empty shelf space and low stock from pack-level boxes.

Given one box per visible pack, this finds shelf rows and then looks for stretches of
a row that no pack covers but that are wide enough to hold one: those are empty
spaces. The number of packs that would fit estimates the missing facings, and a row's
fill ratio (share of the bay width covered by packs) flags low stock.

Everything here is geometry on boxes, so it is independent of which detector made them.
"""

from dataclasses import dataclass
from statistics import median

# A stretch is an empty space when it is at least this many median pack widths wide.
MIN_GAP_WIDTHS = 0.8
# Rows whose fill ratio is below this are low stock.
LOW_FILL_RATIO = 0.6


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


def cluster_rows(boxes: list[list[float]]) -> list[Row]:
    """Group pack boxes into shelf rows, top to bottom, each sorted left to right.

    Boxes are sorted by vertical centre; a box starts a new row when its centre is more
    than half a median pack height below the current row's median centre.
    """
    if not boxes:
        return []
    med_h = median(b[3] - b[1] for b in boxes)
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    groups: list[list[int]] = []
    for i in order:
        cy = (boxes[i][1] + boxes[i][3]) / 2
        if groups and cy - median((boxes[j][1] + boxes[j][3]) / 2 for j in groups[-1]) <= 0.5 * med_h:
            groups[-1].append(i)
        else:
            groups.append([i])
    rows = []
    for k, g in enumerate(groups):
        row_boxes = sorted((boxes[i] for i in g), key=lambda b: b[0])
        rows.append(Row(k, row_boxes, median(b[1] for b in row_boxes), median(b[3] for b in row_boxes)))
    return rows


def find_gaps(boxes: list[list[float]], min_gap_widths: float = MIN_GAP_WIDTHS) -> tuple[list[Row], list[GapCandidate]]:
    """Return the shelf rows (with fill ratios set) and the empty spaces found in them.

    The bay's left and right edges are the outermost pack edges over all rows, so a row
    that stops short of them has a gap at its end.
    """
    rows = cluster_rows(boxes)
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
            if width >= min_gap_widths * w_med:
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
