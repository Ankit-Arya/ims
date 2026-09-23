from __future__ import annotations

import re


def overview_name_matches(title: str | None, filename: str | None, pattern: str) -> bool:
    """Configurable overview-document identity; default pattern is MRGR.

    The special role is not hard-coded to a database UUID or filename. Administrators can
    change OVERVIEW_DOCUMENT_PATTERN without code changes.
    """
    needle = re.sub(r"\s+", "", pattern or "").casefold()
    if not needle:
        return False
    haystack = re.sub(r"\s+", "", f"{title or ''} {filename or ''}").casefold()
    return needle in haystack
