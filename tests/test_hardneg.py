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
