import math

from ike.retrieval.corpus_intelligence import (
    INDEX_VERSION,
    _high_signal_concepts,
    _normalized_centroid,
)


def test_index_version_is_091():
    assert INDEX_VERSION == "0.9.1"


def test_centroid_reuses_existing_vectors_and_normalizes():
    result = _normalized_centroid([[1.0, 0.0], [0.0, 1.0]])
    expected = 1.0 / math.sqrt(2.0)
    assert len(result) == 2
    assert abs(result[0] - expected) < 1e-9
    assert abs(result[1] - expected) < 1e-9
    assert abs(sum(value * value for value in result) - 1.0) < 1e-9


def test_terms_prefer_heading_definition_and_acronym_pairs_over_uppercase_noise():
    terms = _high_signal_concepts(
        "6.10 Emergency Evacuation > Emergency Egress Device (EED)",
        "The Emergency Access Device (EAD) is provided for access. "
        "The controller display contains RED BLACK LOW HIGH FORWARD REVERSE OFF. "
        "Passenger Interface Controller means the equipment used for passenger communication.",
        limit=12,
    )
    folded = {term.casefold() for term in terms}
    assert "6.10 emergency evacuation" in folded
    assert "emergency egress device (eed)" in folded
    assert "eed" in folded
    assert "emergency access device" in folded
    assert "ead" in folded
    assert "passenger interface controller" in folded
    assert "red" not in folded
    assert "black" not in folded
    assert "low" not in folded
    assert "high" not in folded
    assert "forward" not in folded
    assert "reverse" not in folded
    assert "off" not in folded


def test_unsectioned_is_not_promoted_as_terminology():
    assert _high_signal_concepts("Unsectioned", "No explicit definition is present.", limit=12) == []
