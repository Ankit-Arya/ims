from __future__ import annotations

import re


def overview_name_matches(title: str | None, filename: str | None, pattern: str) -> bool:
    """Configurable optional overview-document identity.

    No source is privileged by default. Administrators can opt into a pattern without
    changing application code.
    """
    needle = re.sub(r"\s+", "", pattern or "").casefold()
    if not needle:
        return False
    haystack = re.sub(r"\s+", "", f"{title or ''} {filename or ''}").casefold()
    return needle in haystack
