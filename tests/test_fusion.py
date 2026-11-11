import numpy as np

from wikiret.fusion import rrf


def test_agreement_beats_single_list_strength():
    fused = rrf([np.array([1, 2, 3]), np.array([2, 4, 1])])
    assert list(fused[:2]) == [2, 1]


def test_depth_cuts_lists():
    assert set(rrf([np.array([1, 2, 3])], depth=2)) == {1, 2}
