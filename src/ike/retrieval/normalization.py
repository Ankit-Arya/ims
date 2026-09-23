from __future__ import annotations

import re


_QUOTE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', " ": " ",
})
_QUERY_SPELLING_EQUIVALENTS = (
    (re.compile(r"\bauthorized\b", re.IGNORECASE), "authorised"),
    (re.compile(r"\bauthorization\b", re.IGNORECASE), "authorisation"),
    (re.compile(r"\borganisation\b", re.IGNORECASE), "organization"),
)


def canonical_query_text(text: str) -> str:
    """Normalize harmless user-surface variation without changing technical identifiers."""
    value = (text or "").translate(_QUOTE_TRANSLATION)
    value = re.sub(r"^\s*\d{1,3}[.)]\s+(?=[A-Za-z])", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    for pattern, replacement in _QUERY_SPELLING_EQUIVALENTS:
        value = pattern.sub(replacement, value)
    return value

_IDENTIFIER_RE = re.compile(r"\b([A-Za-z]{1,10})[\s_-]?(\d{1,5}[A-Za-z]?)\b")
_IDENTIFIER_ONLY_RE = re.compile(r"^\s*([A-Za-z]{1,10})[\s_-]*0*(\d{1,5})([A-Za-z]?)\s*$")
_NON_IDENTIFIER_PREFIXES = {"if", "of", "in", "to", "on", "by", "for", "and", "or", "at", "as", "is"}



def canonical_identifier(text: str) -> str | None:
    """Canonicalise a standalone technical identifier without guessing its meaning.

    Short prefixes use a two-digit numeric form so common document identifiers such as
    ``SC 7A`` and ``SC-07A`` share a stable lookup key. The original spelling is always
    retained separately by ``technical_identifier_variants``.
    """
    match = _IDENTIFIER_ONLY_RE.fullmatch(text or "")
    if not match:
        return None
    prefix, digits, suffix = match.groups()
    numeric = digits.zfill(2) if len(prefix) <= 3 and len(digits) == 1 else digits
    return f"{prefix.upper()}-{numeric}{suffix.upper()}"


def technical_identifier_variants(text: str, *, max_variants: int = 8) -> list[str]:
    """Return separator-tolerant technical-name variants without replacing the original."""
    original = re.sub(r"\s+", " ", text.strip())
    if not original:
        return []
    variants = [original]
    seen = {original.casefold()}
    matches = list(_IDENTIFIER_RE.finditer(original))[:4]
    for match in matches:
        prefix, number = match.group(1), match.group(2)
        if prefix.casefold() in _NON_IDENTIFIER_PREFIXES:
            continue
        replacements = [f"{prefix}{number}", f"{prefix}-{number}", f"{prefix} {number}"]
        digits_match = re.fullmatch(r"0*(\d{1,5})([A-Za-z]?)", number)
        if digits_match and len(prefix) <= 3:
            digits, suffix = digits_match.groups()
            canonical_number = digits.zfill(2) if len(digits) == 1 else digits
            upper_prefix = prefix.upper()
            upper_suffix = suffix.upper()
            replacements.extend([
                f"{upper_prefix}{canonical_number}{upper_suffix}",
                f"{upper_prefix}-{canonical_number}{upper_suffix}",
                f"{upper_prefix} {canonical_number}{upper_suffix}",
            ])
        for replacement in replacements:
            candidate = original[: match.start()] + replacement + original[match.end() :]
            candidate = re.sub(r"\s+", " ", candidate).strip()
            key = candidate.casefold()
            if key not in seen:
                variants.append(candidate)
                seen.add(key)
                if len(variants) >= max_variants:
                    return variants
    return variants


def normalized_identifier_tokens(text: str) -> set[str]:
    """Canonical identifier tokens used for matching metadata and text."""
    tokens: set[str] = set()
    for match in _IDENTIFIER_RE.finditer(text or ""):
        tokens.add(f"{match.group(1)}{match.group(2)}".casefold())
    return tokens


def _singularize_word(word: str) -> str | None:
    bare = re.sub(r"[^A-Za-z]", "", word)
    lowered = bare.casefold()
    if len(bare) <= 3:
        return None
    if len(bare) > 4 and lowered.endswith("ies"):
        return re.sub(r"ies(?=[^A-Za-z]*$)", "y", word, flags=re.IGNORECASE)
    if len(bare) > 4 and lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
        return re.sub(r"s(?=[^A-Za-z]*$)", "", word, flags=re.IGNORECASE)
    return None


def _pluralize_word(word: str) -> str | None:
    bare = re.sub(r"[^A-Za-z]", "", word)
    lowered = bare.casefold()
    if len(bare) <= 3 or not bare or bare.isupper():
        return None
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        if lowered.endswith("s"):
            return None
        return re.sub(r"(?=[^A-Za-z]*$)", "es", word, count=1)
    if len(bare) > 3 and lowered.endswith("y") and len(bare) >= 2 and bare[-2].casefold() not in "aeiou":
        return re.sub(r"y(?=[^A-Za-z]*$)", "ies", word, flags=re.IGNORECASE)
    return re.sub(r"(?=[^A-Za-z]*$)", "s", word, count=1)


def lexical_form_variants(text: str, *, max_variants: int = 6) -> list[str]:
    """Conservative singular/plural surface variants for PostgreSQL `simple` FTS.

    `simple` deliberately avoids linguistic stemming for technical corpora. This helper
    therefore supplies bounded bidirectional English number variants while preserving the
    original query and avoiding acronyms/technical identifiers.
    """
    original = re.sub(r"\s+", " ", text.strip())
    if not original:
        return []

    words = original.split(" ")
    variants = [original]
    seen = {original.casefold()}

    singular_words = list(words)
    singular_changed = False
    for index, word in enumerate(words):
        replacement = _singularize_word(word)
        if replacement and replacement != word:
            singular_words[index] = replacement
            singular_changed = True
    if singular_changed:
        candidate = " ".join(singular_words)
        if candidate.casefold() not in seen:
            variants.append(candidate)
            seen.add(candidate.casefold())

    # Add a plural variant for the terminal lexical head only. Pluralising every word
    # produced harmful forms such as ``Crews Control`` or ``Automatic Trains Operation``.
    # The final token captures the useful `simple`-FTS bridge for technical noun phrases:
    # ``Crew Control`` -> ``Crew Controls`` and ``Automatic Train Operation`` ->
    # ``Automatic Train Operations``.
    if words:
        replacement = _pluralize_word(words[-1])
        if replacement and replacement != words[-1]:
            plural_words = list(words)
            plural_words[-1] = replacement
            candidate = " ".join(plural_words)
            key = candidate.casefold()
            if key not in seen:
                variants.append(candidate)
                seen.add(key)

    return variants[:max_variants]
