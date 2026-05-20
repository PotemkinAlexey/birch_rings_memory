"""VectorIndex — numpy-backed cosine search."""
from birch.vector_index import VectorIndex


def test_add_and_search_orders_by_similarity():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    idx.add("c", [0.9, 0.1, 0.0])

    top = idx.search([1.0, 0.0, 0.0], top_k=3)
    ids = [fid for fid, _ in top]
    assert ids[0] == "a"
    assert ids[1] == "c", "vector closest in angle should rank second"
    assert ids[2] == "b"
    assert top[0][1] > top[1][1] > top[2][1]


def test_remove_drops_row_and_rebuilds_index():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0])
    idx.add("b", [0.0, 1.0])
    idx.add("c", [1.0, 1.0])
    idx.remove("b")

    assert "b" not in idx
    assert len(idx) == 2

    sims = idx.all_similarities([1.0, 0.0])
    assert set(sims.keys()) == {"a", "c"}
    assert sims["a"] > sims["c"]


def test_replace_updates_existing_row():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0])
    idx.add("a", [0.0, 1.0])

    assert len(idx) == 1
    sims = idx.all_similarities([1.0, 0.0])
    assert sims["a"] < 0.01, "after replacement, sim should reflect the new vector"


def test_mismatched_dim_is_ignored():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("bad", [1.0, 0.0])

    assert "bad" not in idx
    assert len(idx) == 1


def test_empty_vector_is_noop():
    idx = VectorIndex()
    idx.add("a", [])
    assert len(idx) == 0
    assert idx.search([1.0, 0.0]) == []


def test_search_threshold_filters():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0])
    idx.add("b", [-1.0, 0.0])

    top = idx.search([1.0, 0.0], top_k=5, threshold=0.5)
    assert [fid for fid, _ in top] == ["a"]


def test_static_similarity_pure():
    s = VectorIndex.similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(s - 1.0) < 1e-6
    s2 = VectorIndex.similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(s2) < 1e-6
    s3 = VectorIndex.similarity([1.0, 0.0], [-1.0, 0.0])
    assert abs(s3 + 1.0) < 1e-6
