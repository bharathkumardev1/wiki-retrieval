import numpy as np
import pytest

from wikiret.metrics import bootstrap_ci, paired_diff_ci, per_query


def test_known_ranking():
    m = per_query([["a", "b", "c"]], [{"c"}], k=3)
    assert m["hit@3"][0] == 1 and m["mrr@3"][0] == pytest.approx(1 / 3)
    assert m["ndcg@3"][0] == pytest.approx(1 / np.log2(4))
    assert m["recall@3"][0] == 1


def test_perfect_and_miss():
    m = per_query([["a", "b"], ["x", "y"]], [{"a"}, {"z"}], k=2)
    assert list(m["ndcg@2"]) == [1.0, 0.0]


def test_ndcg_ideal_uses_number_of_relevant_docs():
    m = per_query([["a", "b"]], [{"a", "b", "c"}], k=2)
    assert m["ndcg@2"][0] == pytest.approx(1.0)
    assert m["recall@2"][0] == pytest.approx(2 / 3)


def test_query_without_relevant_docs_raises():
    with pytest.raises(ValueError):
        per_query([["a"]], [set()])


def test_bootstrap_is_seeded_and_brackets_mean():
    v = np.random.default_rng(1).random(200)
    assert bootstrap_ci(v, seed=3) == bootstrap_ci(v, seed=3)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi


def test_paired_diff_excludes_zero_for_a_real_gap():
    a = np.full(300, 0.6) + np.random.default_rng(0).normal(0, 0.05, 300)
    b = np.full(300, 0.5) + np.random.default_rng(1).normal(0, 0.05, 300)
    mean, lo, hi = paired_diff_ci(a, b)
    assert lo > 0 and mean > 0
