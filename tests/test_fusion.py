from uuid import uuid4

from ike.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_multi_source_hits():
    a, b, c = uuid4(), uuid4(), uuid4()
    result = reciprocal_rank_fusion(
        [
            ("dense", [a, b], 1.0),
            ("lexical", [a, c], 1.0),
        ],
        k=60,
    )
    assert result[a][0] > result[b][0]
    assert result[a][0] > result[c][0]
    assert result[a][1] == {"dense", "lexical"}
