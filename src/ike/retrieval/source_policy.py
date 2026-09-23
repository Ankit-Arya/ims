from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from uuid import UUID


def split_patterns(value: str | None) -> list[str]:
    """Parse administrator-configured source-family/title patterns.

    Patterns are configuration, not domain logic.  Deployments can prioritize a general rule
    book, a policy family, a standards manual, or leave the setting empty.
    """

    result: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[,;\n]+", value or ""):
        item = re.sub(r"\s+", " ", raw).strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def source_name_matches(title: str | None, filename: str | None, pattern: str) -> bool:
    needle = re.sub(r"\s+", "", pattern or "").casefold()
    if not needle:
        return False
    haystack = re.sub(r"\s+", "", f"{title or ''} {filename or ''}").casefold()
    return needle in haystack


@dataclass(slots=True)
class SourcePolicy:
    priority_document_ids: list[UUID] = field(default_factory=list)
    priority_patterns: list[str] = field(default_factory=list)
    hard_scope: bool = False
    reason: str = "corpus_default"

    @property
    def has_priority_stage(self) -> bool:
        return bool(self.priority_document_ids)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["priority_document_ids"] = [str(item) for item in self.priority_document_ids]
        payload["has_priority_stage"] = self.has_priority_stage
        return payload
