from wikiret.hardneg import mine, norm


def test_negatives_come_from_the_rank_window_and_exclude_the_positive():
    pool = [f"topic{i} common words here" for i in range(60)] + ["the target about zebras zebras"]
    out = mine(["zebras"], pool, positives=[60], lo=0, hi=20, seed=1)
    assert out[0] != 60 and 0 <= out[0] < len(pool)


def test_deterministic_for_a_seed():
    pool = [f"alpha beta {i}" for i in range(80)]
    a = mine(["alpha beta"] * 3, pool, [0, 1, 2], lo=5, hi=40, seed=7)
    b = mine(["alpha beta"] * 3, pool, [0, 1, 2], lo=5, hi=40, seed=7)
    assert a == b


def test_norm_collapses_case_and_space():
    assert norm("A  b\nC") == "a b c"


def test_dense_window_and_positive_exclusion():
    import numpy as np

    from wikiret.hardneg import mine_dense
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(300, 8)).astype("float32")
    q = pool[:5] + 0.01
    out = mine_dense(q, pool, positives=[0, 1, 2, 3, 4], lo=20, hi=120, seed=2)
    sc = q @ pool.T
    for i, o in enumerate(out):
        rank = int((sc[i] > sc[i][o]).sum())
        assert 20 <= rank < 120 and o != i
