import re

COMPLEX_QUESTION_RE = re.compile(
    r"\b(compare|comparison|across|all\b|every\b|analyse|analyze|summari[sz]e|"
    r"procedure|steps|requirements|exceptions?|differences?|relationship|why\b|how\b|"
    r"research|review|evaluate|assess|implications?|contradictions?|duties|responsibilities|"
    r"functions?|obligations?|roles?)\b",
    re.IGNORECASE,
)

COVERAGE_QUESTION_RE = re.compile(
    r"\b(procedure|procedures|steps|requirements?|instructions?|process|processes|method|methods|"
    r"handling|actions?\s+required|what\s+action|what\s+to\s+do|how\s+to)\b",
    re.IGNORECASE,
)

ROLE_COVERAGE_CUE_RE = re.compile(
    r"\b(dut(?:y|ies)|responsibilit(?:y|ies)|functions?|obligations?|role(?:s)?)\b",
    re.IGNORECASE,
)

VERIFY_QUESTION_RE = re.compile(
    r"\b(all\b|every\b|procedure|steps|compare|difference|requirements|exceptions?|"
    r"limit|repayment|duration|how many|must\b|shall\b|contradictions?|duties|responsibilities|"
    r"functions?|obligations?|roles?|locations?|located|where\b|contacts?)\b",
    re.IGNORECASE,
)

_DEFINITION_CUE_RE = re.compile(
    r"\b(?:what\s+(?:is|does)|meaning\s+of|full\s+form\s+of|expand|define|definition\s+of|definition\s*[:\-])\b",
    re.IGNORECASE,
)
_ATTRIBUTE_CUE_RE = re.compile(
    r"\b(?:where|locat(?:e|ed|ion|ions)|address(?:es)?|contact(?:s)?|phone|mobile|extension(?:s)?|"
    r"limit(?:s)?|repayment|duration|validity|frequency|how\s+many|line[-\s]?wise|by\s+line)\b",
    re.IGNORECASE,
)
_ENUMERATION_CUE_RE = re.compile(
    r"\b(?:all|every|list|enumerate|types?\s+of|different\s+types?|various|number\s+and\s+names?|"
    r"items?|members?|classification|categories|line[-\s]?wise|by\s+line)\b|"
    r"^\s*(?:show|provide|give)\b",
    re.IGNORECASE,
)


_SOURCE_SCOPE_TAIL_RE = re.compile(
    r"\s+\b(?:as\s+per|according\s+to|under|within)\s+(?:the\s+)?"
    r"(?P<source>[A-Za-z0-9][A-Za-z0-9 _./&()-]{1,80})$",
    re.IGNORECASE,
)
_FROM_SOURCE_SCOPE_TAIL_RE = re.compile(
    r"\s+\bfrom\s+(?:the\s+)?(?P<source>[A-Za-z0-9][A-Za-z0-9 _./&()-]{1,80})$",
    re.IGNORECASE,
)
_SOURCE_LIKE_RE = re.compile(
    r"\b(?:manual|handbook|rule|rules|policy|circular|instruction|sop|otm|swo|mrgr|adm|"
    r"compendium|document|file|source|guideline|standard|act|regulation|code)\b|"
    r"^[A-Z][A-Z0-9./_-]{1,15}$",
    re.IGNORECASE,
)

def extract_source_scope(question: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", question.strip().strip("?.! :"))
    match = _SOURCE_SCOPE_TAIL_RE.search(cleaned)
    if match:
        source = re.sub(r"\s+", " ", match.group("source")).strip(" ,;:-")
        # "as per pay scale/designation" and similar qualifiers are applicability clauses,
        # not document names. Only promote a tail to source scope when it looks like a named
        # documentary source (MRGR, ADM, handbook, policy, business rule, etc.).
        if not (1 <= len(source.split()) <= 10 and _SOURCE_LIKE_RE.search(source)):
            return None
        return source

    # Plain "from X" is often ordinary syntax ("from commissioned to non-commissioned",
    # "rescued from train"). Treat it as document scope only when X itself looks like a
    # named documentary source.
    match = _FROM_SOURCE_SCOPE_TAIL_RE.search(cleaned)
    if not match:
        return None
    source = re.sub(r"\s+", " ", match.group("source")).strip(" ,;:-")
    if not (1 <= len(source.split()) <= 10 and _SOURCE_LIKE_RE.search(source)):
        return None
    return source

_SINGLE_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9_./-]{1,11}$")
_TECHNICAL_IDENTIFIER_ONLY_RE = re.compile(r"^[A-Za-z]{1,3}[\s_-]*0*\d{1,5}[A-Za-z]?$", re.IGNORECASE)
_GENERIC_LOOKUP_HEAD_RE = re.compile(
    r"^(?:the\s+)?(?:procedure|procedures|process|processes|steps?|requirements?|instructions?|method|methods)\b",
    re.IGNORECASE,
)

_ROLE_PATTERNS = (
    re.compile(
        r"(?:\b(?:all|every|complete|comprehensive)\b\s+)?(?:the\s+)?"
        r"(?:dut(?:y|ies)|responsibilit(?:y|ies)|functions?|obligations?|role(?:s)?)\s+"
        r"(?:of|for)\s+(?:the\s+)?(?P<subject>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:are|is)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:dut(?:y|ies)|responsibilit(?:y|ies)|functions?|obligations?|role(?:s)?)\s+"
        r"(?:of|for)\s+(?:the\s+)?(?P<subject>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:list|give|provide|show|enumerate)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:dut(?:y|ies)|responsibilit(?:y|ies)|functions?|obligations?|role(?:s)?)\s+"
        r"(?:of|for)\s+(?:the\s+)?(?P<subject>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:all\s+)?(?:does|do)\s+(?P<subject>.+?)\s+(?:do|handle|perform)\??$",
        re.IGNORECASE,
    ),
)

_ROLE_SUBJECT_STOP_RE = re.compile(
    r"\s+\b(?:during|when|while|under|according\s+to|across|within|in\s+case\s+of|for\s+the\s+purpose\s+of)\b.*$",
    re.IGNORECASE,
)

_ENTITY_TAIL_RE = re.compile(
    r"(?:\s*[,;]?)\s+(?:and|but|or)\s+(?=(?:what|where|when|which|who|how|its\b|it['’]?s\b|what['’]?s\b)).*$",
    re.IGNORECASE,
)
_ENTITY_ENUM_TAIL_RE = re.compile(
    r"\s+\b(?:line[-\s]?wise|by\s+line|across\s+(?:all\s+)?documents?|in\s+(?:all\s+)?documents?)\b.*$",
    re.IGNORECASE,
)


def _clean_entity_candidate(value: str, *, normalize_single_acronym: bool = False) -> str | None:
    value = re.sub(r"\s+", " ", value.strip().strip("?.! :;,-\'\""))
    value = _ENTITY_TAIL_RE.sub("", value).strip("?.! :;,-")
    value = re.sub(
        r"\s*[,;:]\s*(?:what|which|where|when|who|why|how)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip("?.! :;,-")
    value = _ENTITY_ENUM_TAIL_RE.sub("", value).strip("?.! :;,-")
    # Common where-formulations put the location verb after the entity. Keep that
    # predicate out of the entity phrase ("where are fire extinguishers located?").
    value = re.sub(r"\s+\b(?:located|situated)\b$", "", value, flags=re.IGNORECASE).strip("?.! :;,-")
    value = re.sub(r"^(?:a|an|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:please|kindly)$", "", value, flags=re.IGNORECASE)
    if not value or _GENERIC_LOOKUP_HEAD_RE.search(value):
        return None
    # Interrogative/request tails such as ``which applies`` are predicates, not entities.
    # Rejecting them is safer than turning a conflict question into a fake entity query.
    if re.match(r"^(?:what|which|where|when|who|why|how)\b", value, re.IGNORECASE):
        return None
    tokens = re.findall(r"[A-Za-z0-9_./-]+", value)
    if not tokens or len(tokens) > 10 or len(value) > 100:
        return None
    # Definition cues retain the legacy convenience where a short one-token lookup such
    # as ``what is bic`` becomes ``BIC``. Enumeration/category extraction keeps ordinary
    # one-word nouns in user casing (``depots`` must not become acronym-like ``DEPOTS``).
    if normalize_single_acronym and len(tokens) == 1 and len(value) <= 12 and re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_./-]{1,11}", value
    ):
        return value.upper()
    return value


def extract_lookup_term(question: str) -> str | None:
    """Return the entity/identifier targeted by a definition-style question.

    Unlike the 0.6.0 implementation, this preserves bounded multi-word entities such as
    "Crew Control" or "Automatic Train Operation" rather than truncating them to the
    first token. A bare single-token path remains conservative and acronym-only.
    """

    cleaned = re.sub(r"\s+", " ", question.strip().strip("?.! :"))
    if _TECHNICAL_IDENTIFIER_ONLY_RE.fullmatch(cleaned):
        return cleaned.upper()
    if _SINGLE_ACRONYM_RE.fullmatch(cleaned):
        return cleaned
    patterns = (
        # ``what does X mean/stand for`` needs a right boundary; otherwise a multi-word
        # extractor would incorrectly make "X mean" part of the entity.
        re.compile(
            r"^what\s+does\s+(?P<term>.+?)\s+(?:mean|means|stand\s+for)(?:\s+(?:here|in\s+(?:this|that)\s+context))?\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:what\s+is|meaning\s+of|full\s+form\s+of|expand|define|definition\s+of|definition\s*[:\-])\s*(?P<term>.+)$",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(cleaned)
        if match:
            term = _SOURCE_SCOPE_TAIL_RE.sub("", match.group("term")).strip()
            return _clean_entity_candidate(term, normalize_single_acronym=True)
    return None


def _looks_like_independent_set_member(value: str) -> bool:
    """Conservative signal that one coordinated phrase can stand as its own set/category.

    We split ``depots and crew controls`` for list/enumeration questions, but avoid
    blindly splitting compound names such as ``safety and security equipment``. The
    heuristic is intentionally grammatical rather than corpus/domain specific.
    """

    tokens = re.findall(r"[A-Za-z0-9_./-]+", value)
    if not tokens or len(tokens) > 6:
        return False
    head = tokens[-1]
    if re.fullmatch(r"[A-Z][A-Z0-9_./-]{1,11}", head):
        return True
    lowered = re.sub(r"[^A-Za-z]", "", head).casefold()
    return len(lowered) > 3 and lowered.endswith("s") and not lowered.endswith(("ss", "us", "is"))


def _shared_suffix_slash_entities(value: str) -> list[str]:
    """Expand compact coordinated phrases such as 'red/general alert' generically."""
    match = re.fullmatch(
        r"\s*([A-Za-z][A-Za-z0-9_-]*)\s*/\s*([A-Za-z][A-Za-z0-9_-]*)\s+([A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,2})\s*",
        value,
    )
    if not match:
        return []
    left, right, suffix = match.groups()
    return [f"{left} {suffix}", f"{right} {suffix}"]


def _split_coordinated_entities(value: str, *, explicit_enumeration: bool) -> list[str]:
    """Split a bounded list target into independent entity/category phrases when safe."""

    if not explicit_enumeration:
        cleaned = _clean_entity_candidate(value)
        return [cleaned] if cleaned else []

    has_punctuation = bool(re.search(r"[,;]", value))
    has_conjunction = bool(re.search(r"\s+(?:and|&)\s+", value, re.IGNORECASE))
    if not (has_punctuation or has_conjunction):
        cleaned = _clean_entity_candidate(value)
        return [cleaned] if cleaned else []

    parts = [
        part.strip()
        for part in re.split(r"\s*(?:[,;]|\band\b|&)\s*", value, flags=re.IGNORECASE)
        if part.strip()
    ]
    if not 2 <= len(parts) <= 5:
        cleaned = _clean_entity_candidate(value)
        return [cleaned] if cleaned else []

    cleaned_parts = [_clean_entity_candidate(part) for part in parts]
    if any(part is None for part in cleaned_parts):
        cleaned = _clean_entity_candidate(value)
        return [cleaned] if cleaned else []
    result = [part for part in cleaned_parts if part]

    # Commas/semicolons are an explicit list boundary. For a conjunction-only phrase,
    # require each member to look independently plural/identifier-like before splitting.
    # This fixes multi-category requests without turning arbitrary noun compounds into
    # unrelated searches.
    if has_punctuation or all(_looks_like_independent_set_member(part) for part in result):
        return result

    cleaned = _clean_entity_candidate(value)
    return [cleaned] if cleaned else []


def extract_entity_terms(question: str) -> list[str]:
    """Extract one or more bounded entity/category phrases from a user question.

    Multi-category enumeration is represented as separate search targets. This prevents
    ``List of depots and crew controls`` from becoming one mandatory phrase query
    (``depots AND crew AND controls``), which was the 0.6.1 regression.
    """

    lookup = extract_lookup_term(question)
    if lookup:
        slash_terms = _shared_suffix_slash_entities(lookup)
        return slash_terms or [lookup]

    cleaned = re.sub(r"\s+", " ", question.strip().strip("?.! :"))
    patterns = (
        re.compile(
            r"\b(?:locations?|addresses?|contacts?|limits?|durations?|validity|frequency|repayment(?:\s+time)?)\s+"
            r"(?:of|for)\s+(?:the\s+)?(?P<term>.+)$",
            re.IGNORECASE,
        ),
        re.compile(r"\bwhere\s+(?:is|are)\s+(?:the\s+)?(?P<term>.+)$", re.IGNORECASE),
        re.compile(
            r"\b(?:list(?:\s+of)?|show|give|provide|enumerate)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?P<term>.+)$",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        raw = match.group("term")
        terms = _split_coordinated_entities(
            raw, explicit_enumeration=bool(_ENUMERATION_CUE_RE.search(question))
        )
        if terms:
            # Stable case-insensitive de-duplication.
            unique: list[str] = []
            seen: set[str] = set()
            for term in terms:
                key = term.casefold()
                if key not in seen:
                    unique.append(term)
                    seen.add(key)
            return unique
    return []


def extract_entity_term(question: str) -> str | None:
    """Return the single entity target, or ``None`` for true multi-entity requests."""

    terms = extract_entity_terms(question)
    return terms[0] if len(terms) == 1 else None


def query_facets(question: str) -> list[str]:
    """Return observable information facets requested by the user."""

    facets: list[str] = []
    lowered = question.casefold()
    if _DEFINITION_CUE_RE.search(question):
        facets.append("definition")
    if re.search(r"\b(?:where|locations?|located|situated|address(?:es)?)\b", lowered):
        facets.append("location")
    if re.search(r"\b(?:contacts?|phone|mobile|extension(?:s)?)\b", lowered):
        facets.append("contact")
    if re.search(
        r"\b(?:limits?|repayment|duration|validity|frequency|how\s+many|how\s+much|"
        r"how\s+long|after\s+how\s+much\s+time|speed|rate|percentage|percent)\b",
        lowered,
    ):
        facets.append("constraint")
    if _ENUMERATION_CUE_RE.search(question):
        facets.append("enumeration")
    if re.search(r"\b(?:who\s+(?:impose|imposes|imposed|authori[sz]es|approves)|imposing\s+authority|who\s+is\s+responsible)\b", lowered):
        facets.append("authority")
    return list(dict.fromkeys(facets))


def is_entity_attribute_coverage_question(question: str) -> bool:
    """True when an entity's attributes/set membership need broader evidence coverage."""

    entities = extract_entity_terms(question)
    if not entities:
        return False
    facets = query_facets(question)
    return bool(
        any(facet in {"location", "contact", "constraint", "enumeration"} for facet in facets)
    )


def is_lookup_question(question: str) -> bool:
    return extract_lookup_term(question) is not None


def is_complex_question(question: str) -> bool:
    facets = query_facets(question)
    return (
        len(question) > 220
        or bool(COMPLEX_QUESTION_RE.search(question))
        or len(facets) >= 2
        or ("enumeration" in facets and is_entity_attribute_coverage_question(question))
    )


def is_coverage_question(question: str) -> bool:
    """Procedure/process queries where source/document variants must be preserved."""
    return bool(COVERAGE_QUESTION_RE.search(question))


def extract_role_subject(question: str) -> str | None:
    """Extract the role/entity targeted by a broad duties/responsibilities question."""

    cleaned = re.sub(r"\s+", " ", question.strip().strip("?.! :"))
    for pattern in _ROLE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        subject = match.group("subject").strip("?.! :;,-")
        subject = _ROLE_SUBJECT_STOP_RE.sub("", subject).strip("?.! :;,-")
        subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.IGNORECASE)
        if 1 < len(subject) <= 100:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_./-]{1,7}", subject):
                subject = subject.upper()
            return subject
    return None


def is_role_coverage_question(question: str) -> bool:
    return bool(ROLE_COVERAGE_CUE_RE.search(question)) and extract_role_subject(question) is not None


def should_verify_question(question: str) -> bool:
    return bool(VERIFY_QUESTION_RE.search(question)) or is_entity_attribute_coverage_question(question)


_COVERAGE_STOP_WORDS = {
    "a", "an", "the", "what", "which", "how", "is", "are", "was", "were",
    "for", "to", "of", "in", "on", "from", "about", "across", "all", "every",
    "document", "documents", "manual", "manuals", "procedure", "procedures",
    "process", "processes", "steps", "step", "requirements", "requirement",
    "instruction", "instructions", "method", "methods", "please", "give", "provide",
    "where", "location", "locations", "located", "list", "show", "enumerate",
    "its", "it", "s", "linewise", "wise",
}


def coverage_search_query(question: str) -> str:
    """Create a deterministic topic query for corpus coverage discovery."""
    tokens = re.findall(r"[A-Za-z0-9_./-]+", question)
    kept = [token for token in tokens if token.lower() not in _COVERAGE_STOP_WORDS]
    return " ".join(kept[:16]) or question.strip()


def role_search_queries(subject: str, aliases: list[str] | tuple[str, ...]) -> list[str]:
    """Build deterministic, bounded search formulations for role coverage."""

    ordered: list[str] = []
    seen: set[str] = set()
    names = [subject, *aliases]
    for name in names:
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            continue
        for query in (
            f"Responsibilities of {name}",
            f"Duties of {name}",
            f"{name} shall",
            f"{name} responsibilities duties",
        ):
            key = query.casefold()
            if key not in seen:
                ordered.append(query)
                seen.add(key)
    return ordered[:16]
