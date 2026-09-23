from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_LINE_RE = re.compile(r"\bline[\s_-]?(\d{1,3}[A-Za-z]?)\b", re.IGNORECASE)
_RS_RE = re.compile(r"\bRS[\s_-]?(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)
_REV_RE = re.compile(r"\b(?:rev(?:ision)?|version|ver\.?)\s*[:._-]?\s*([A-Za-z0-9.]+)", re.IGNORECASE)


def _top(values: Iterable[str], limit: int = 8) -> list[str]:
    counts = Counter(v for v in values if v)
    return [value for value, _count in counts.most_common(limit)]


def infer_operational_profile(title: str, filename: str, samples: list[str]) -> dict:
    """Infer only high-signal document identity fields from existing extracted content."""
    identity_text = "\n".join([title or "", filename or ""])
    joined = "\n".join([identity_text, *samples[:20]])
    primary_lines = [f"Line {m.group(1)}" for m in _LINE_RE.finditer(identity_text)]
    primary_stocks = [f"RS-{m.group(1)}" for m in _RS_RE.finditer(identity_text)]
    lines = [f"Line {m.group(1)}" for m in _LINE_RE.finditer(joined)]
    stocks = [f"RS-{m.group(1)}" for m in _RS_RE.finditer(joined)]
    revisions = [m.group(1) for m in _REV_RE.finditer(joined)]
    lowered = joined.casefold()
    doc_type = "unknown"
    for label, needles in (
        ("troubleshooting", ("troubleshooting", "fault finding", "fault isolation")),
        ("emergency", ("emergency procedure", "emergency response")),
        ("operating_manual", ("operating manual", "operation manual", "train operator")),
        ("rules", ("general rules", "metro railway general rules", "mrgr")),
        ("maintenance", ("maintenance manual", "maintenance instruction")),
        ("training_reference", ("training manual", "handbook")),
    ):
        if any(needle in lowered for needle in needles):
            doc_type = label
            break
    confidence = 0.35
    if lines or stocks:
        confidence = 0.75
    if (title and title.casefold() in lowered) or "mrgr" in lowered:
        confidence = max(confidence, 0.8)
    return {
        "primary_line_codes": _top(primary_lines),
        "primary_rolling_stock": _top(primary_stocks),
        "line_codes": _top(lines),
        "rolling_stock": _top(stocks),
        "document_type": doc_type,
        "inferred_revision": revisions[0] if revisions else None,
        "classification_confidence": confidence,
        "source": "existing_title_filename_chunks",
    }
