"""Kinship-estimator extension tests (IBS, windowed; LOCO is in
test_review_fixes.py)."""

import numpy as np
import pytest

import phensim
from phensim.kinship import grm, ibs_kinship, windowed_kinships


def _sim_G(n=80, m=600, seed=0):
    return phensim.simulate_independent(n, m, maf=0.3, freq_dist="uniform", seed=seed)


def test_ibs_kinship_orders_relatedness():
    G = _sim_G()
    # duplicate one individual: an IBS "clone"
    G[1] = G[0]
    K = ibs_kinship(G, scale=False)
    assert K[0, 1] == 1.0  # every called genotype matches
    off = K[np.triu_indices_from(K, 1)]
    assert off.mean() < 0.8  # unrelated pairs share far less


def test_ibs_kinship_skips_missing():
    G = _sim_G().astype(float)
    G[0, :100] = -1  # fully missing for person 0 on 100 variants
    K = ibs_kinship(G, scale=False)
    assert np.isfinite(K).all()
    # person 0 vs 1 is scored only on the jointly called variants
    assert 0 <= K[0, 1] <= 1


def test_windowed_kinships_partition():
    G = _sim_G(n=60, m=400)
    windows = list(windowed_kinships(G, 100, 100, scale=False))
    assert len(windows) == 4
    # local window equals the unscaled GRM of those columns
    for wi, (i, Kloc, Krest) in enumerate(windows):
        cols = G[:, i * 100:(i + 1) * 100]
        np.testing.assert_allclose(Kloc, grm(cols, scale=False), atol=1e-10)
        assert Krest.shape == Kloc.shape
    with np.testing.assert_raises(ValueError):
        list(windowed_kinships(G, 0, 50))


def test_constant_fractional_columns_contribute_no_kinship():
    from phensim.kinship import _called_standardized
    G = np.column_stack([np.full(30, .1), np.full(30, .1), np.full(30, np.nan),
                         np.linspace(.1, .1 + 1e-12, 30)])
    G[:4, 1] = -1
    Z, counts = _called_standardized(G, block=2)
    np.testing.assert_array_equal(Z[:, :3], 0.)
    np.testing.assert_array_equal(counts, [30, 26, 0, 30])
    assert Z[:, 3].std() > .99  # tiny real variation is retained
    expected = np.outer(Z[:, 3], Z[:, 3]) / 4
    np.testing.assert_allclose(phensim.grm(G, scale=False), expected)
    np.testing.assert_allclose(phensim.loco_kinships(G, [1, 2, 2, 2], scale=False)[1], expected * 4/3)
    _, local, _ = next(phensim.windowed_kinships(G, 3, 3, scale=False))
    np.testing.assert_array_equal(local, 0.)
    np.testing.assert_array_equal(phensim.grm(np.full((30, 5), .1), scale=False), 0.)
    with pytest.raises(ValueError, match="no polymorphic"):
        phensim.simulate_trait(np.full((30, 5), .1), architecture="infinitesimal", n_causal=0)
