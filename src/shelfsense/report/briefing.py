"""Shift briefing: the compliance report turned into a walk-the-aisle task list
plus a short summary for the store manager.

The task list is built in code. Steps are grouped by shelf, top shelf first,
and within a shelf the most costly problems come first (an empty slot loses
sales, a wrong price is a legal and trust issue, an unreadable label only needs
a look). Every step is a violation's own recommended_action, unchanged.

The summary is written by a small language model (Qwen2.5-1.5B-Instruct, 4-bit
GGUF, about 1 GB, run on CPU through llama.cpp). A model that small rewords
facts wrongly when allowed to restate them: asked to rewrite "shows 2.97,
catalogue price is 2.70", it wrote "shows 2.70". So the model never writes
the steps, and its summary is checked before use: any number in it other than
the score, a slot ID, a label ID or a product name from the report rejects it.
The template summary is used when the model is rejected, missing, or the
`llama-cpp-python` package is not installed (`pip install -e ".[slm]"`).

    briefing = write_briefing(report, catalogue)
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from pydantic import BaseModel

from shelfsense.contracts import CatalogueItem
from shelfsense.report.compliance import ComplianceReport, Violation, ViolationKind

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "slm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Lower comes first within a shelf.
PRIORITY: dict[ViolationKind, int] = {
    ViolationKind.STOCK_OUT: 0,
    ViolationKind.PRICE_MISMATCH: 1,
    ViolationKind.MISPLACED: 2,
    ViolationKind.LOW_STOCK: 3,
    ViolationKind.UNKNOWN_PRODUCT: 4,
    ViolationKind.UNRECOGNISED_ITEM: 5,
    ViolationKind.UNREADABLE_PRICE: 6,
}

# (singular, plural)
KIND_TEXT: dict[ViolationKind, tuple[str, str]] = {
    ViolationKind.STOCK_OUT: ("stock-out", "stock-outs"),
    ViolationKind.LOW_STOCK: ("low-stock slot", "low-stock slots"),
    ViolationKind.MISPLACED: ("misplaced product", "misplaced products"),
    ViolationKind.UNRECOGNISED_ITEM: ("unrecognised product", "unrecognised products"),
    ViolationKind.PRICE_MISMATCH: ("price mismatch", "price mismatches"),
    ViolationKind.UNREADABLE_PRICE: ("unreadable label", "unreadable labels"),
    ViolationKind.UNKNOWN_PRODUCT: ("label for an unknown product", "labels for unknown products"),
}


class ShelfTasks(BaseModel):
    shelf_id: str | None
    actions: list[str]


class Briefing(BaseModel):
    summary: str
    summary_source: str  # "slm" or "template"
    rejected_summary: str | None  # the model's summary when the check threw it out
    seconds: float  # time spent on the model, 0 for the template
    tasks: list[ShelfTasks]


def _shelf_key(shelf_id: str | None) -> tuple[int, str]:
    # "S1" < "S2" < ... < "S10"; violations with no shelf go last
    if shelf_id is None:
        return (1 << 30, "")
    digits = re.findall(r"\d+", shelf_id)
    return (int(digits[0]) if digits else 0, shelf_id)


def group_tasks(report: ComplianceReport) -> list[ShelfTasks]:
    by_shelf: dict[str | None, list[Violation]] = {}
    for v in report.violations:
        by_shelf.setdefault(v.shelf_id, []).append(v)
    return [
        ShelfTasks(
            shelf_id=shelf_id,
            actions=[v.recommended_action for v in sorted(vs, key=lambda v: (PRIORITY[v.kind], v.slot_id or ""))],
        )
        for shelf_id, vs in sorted(by_shelf.items(), key=lambda kv: _shelf_key(kv[0]))
    ]


def template_summary(report: ComplianceReport) -> str:
    if not report.violations:
        return f"Compliance score {report.score:.0f}/100. The shelf matches its planogram; no action needed."
    parts = [
        f"{n} {KIND_TEXT[ViolationKind(kind)][n > 1]}"
        for kind, n in sorted(report.counts.items(), key=lambda kv: PRIORITY[ViolationKind(kv[0])])
    ]
    first = min(report.violations, key=lambda v: PRIORITY[v.kind])
    return (
        f"Compliance score {report.score:.0f}/100 with {', '.join(parts)}. "
        f"Start with: {first.recommended_action}"
    )


TOP_N = 3


def most_urgent(report: ComplianceReport) -> list[Violation]:
    return sorted(report.violations, key=lambda v: (PRIORITY[v.kind], _shelf_key(v.shelf_id), v.slot_id or ""))[:TOP_N]


def _facts(report: ComplianceReport, catalogue: dict[str, CatalogueItem]) -> str:
    others = len(report.violations) - TOP_N
    lines = [f"Compliance score: {report.score:.0f}/100.", "Most urgent problems, most urgent first:"]
    for i, v in enumerate(most_urgent(report), 1):
        product = catalogue[v.sku].name if v.sku in catalogue else "an unidentified product"
        where = v.slot_id or v.shelf_id or "an unplaced label"
        lines.append(f"{i}. {KIND_TEXT[v.kind][0]} at {where} ({product})")
    if others > 0:
        lines.append(f"Plus {'one smaller problem' if others == 1 else 'several smaller problems'} elsewhere.")
    return "\n".join(lines)


def _allowed_tokens(report: ComplianceReport, catalogue: dict[str, CatalogueItem]) -> list[str]:
    """Strings the summary may contain digits inside of."""
    allowed = [f"{report.score:.0f}/100", f"{report.score:.0f}", "100"]
    for v in report.violations:
        allowed += [s for s in (v.slot_id, v.shelf_id, v.sku) if s]
        allowed += re.findall(r"label_\d+", v.recommended_action)
        if v.sku in catalogue:
            allowed.append(catalogue[v.sku].name)
    return sorted(set(allowed), key=len, reverse=True)


# How the model words each kind of problem. A sentence claiming one of these must be
# about a slot that really has it ("S2-02 is running low" when S2-02 is a stock-out fails).
KIND_WORDS: dict[ViolationKind, re.Pattern] = {
    ViolationKind.STOCK_OUT: re.compile(r"out of stock|stock-?outs?\b|\bempty\b", re.I),
    ViolationKind.LOW_STOCK: re.compile(r"low stock|running low|\blow on\b|understock", re.I),
    ViolationKind.PRICE_MISMATCH: re.compile(r"price mismatch|(wrong|incorrect) price|price is (wrong|incorrect)", re.I),
    ViolationKind.MISPLACED: re.compile(r"misplac|wrong (place|slot|shelf)", re.I),
    ViolationKind.UNREADABLE_PRICE: re.compile(r"unreadable|could not be read", re.I),
    ViolationKind.UNRECOGNISED_ITEM: re.compile(r"unrecogni[sz]ed|unidentified", re.I),
    ViolationKind.UNKNOWN_PRODUCT: re.compile(r"unknown product", re.I),
}
_SLOT_RE = re.compile(r"\bS\d+(?:-\d+)?\b")


def _kinds_consistent(text: str, urgent: list[Violation]) -> bool:
    for sentence in re.split(r"(?<=[.!])\s+", text):
        claimed = {k for k, pattern in KIND_WORDS.items() if pattern.search(sentence)}
        slots = set(_SLOT_RE.findall(sentence))
        true_kinds = {v.kind for v in urgent if not slots or {v.slot_id, v.shelf_id} & slots}
        if claimed - true_kinds:
            return False
    return True


def check_summary(text: str, report: ComplianceReport, catalogue: dict[str, CatalogueItem]) -> bool:
    """True when the summary is complete, names only the most urgent slots with the problems they
    really have, calls no problem slot compliant, and every digit in it belongs to the score,
    an ID, or a product name from the report."""
    if not text.strip() or text.rstrip()[-1] not in ".!":
        return False  # empty, or cut off by the token limit
    urgent = most_urgent(report)
    urgent_slots = {s for v in urgent for s in (v.slot_id, v.shelf_id) if s}
    if set(_SLOT_RE.findall(text)) - urgent_slots:
        return False
    if re.search(r"\bcompliant\b", text, re.I) or not _kinds_consistent(text, urgent):
        return False
    rest = text
    for token in _allowed_tokens(report, catalogue):
        rest = re.sub(re.escape(token), " ", rest, flags=re.I)
    return not re.search(r"\d", rest)


_llms: dict[Path, object] = {}


def _load(model: Path):
    if model not in _llms:
        from llama_cpp import Llama  # optional dependency, only when the model is used

        _llms[model] = Llama(str(model), n_ctx=2048, n_threads=6, verbose=False)
    return _llms[model]


def slm_summary(report: ComplianceReport, catalogue: dict[str, CatalogueItem], model: Path = DEFAULT_MODEL) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You brief a supermarket store manager on one shelf audit. Write at most 2 short, plain "
                "sentences: how the shelf is doing, then what to fix first, keeping the order given. "
                "Use only the facts given. Do not write any prices, counts or other numbers except "
                "the score and slot IDs."
            ),
        },
        {"role": "user", "content": _facts(report, catalogue)},
    ]
    out = _load(model).create_chat_completion(messages, temperature=0.2, max_tokens=90)
    return out["choices"][0]["message"]["content"].strip()


def write_briefing(
    report: ComplianceReport,
    catalogue: dict[str, CatalogueItem],
    model: Path | None = DEFAULT_MODEL,
) -> Briefing:
    """Task list plus summary. model=None, a missing model file or a missing
    llama-cpp-python all give the template summary."""
    tasks = group_tasks(report)
    if model is None or not Path(model).exists() or not report.violations:
        return Briefing(summary=template_summary(report), summary_source="template",
                        rejected_summary=None, seconds=0.0, tasks=tasks)
    try:
        start = time.perf_counter()
        text = slm_summary(report, catalogue, Path(model))
        seconds = round(time.perf_counter() - start, 2)
    except ImportError:
        return Briefing(summary=template_summary(report), summary_source="template",
                        rejected_summary=None, seconds=0.0, tasks=tasks)
    if check_summary(text, report, catalogue):
        return Briefing(summary=text, summary_source="slm", rejected_summary=None, seconds=seconds, tasks=tasks)
    return Briefing(summary=template_summary(report), summary_source="template",
                    rejected_summary=text, seconds=seconds, tasks=tasks)
