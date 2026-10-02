"""Kinship-estimator extension tests (IBS, LOCO, windowed)."""

import numpy as np

import phensim
from phensim.kinship import grm, ibs_kinship, loco_kinships, windowed_kinships


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


def test_loco_matches_single_chromosome_grm():
    G = _sim_G()
    chrom = np.repeat([1, 2, 3], 200)
    out = loco_kinships(G, chrom, scale=False)
    for c in (1, 2, 3):
        other = G[:, chrom != c]
        np.testing.assert_allclose(out[c], grm(other, scale=False), atol=1e-10)
    with np.testing.assert_raises(ValueError):
        loco_kinships(G, np.repeat([1], 100))  # wrong variant count


def test_loco_scaled_recovers_grm_diagonal_level():
    G = _sim_G()
    chrom = np.repeat([1, 2], 300)
    out = loco_kinships(G, chrom, scale=True)
    for c in (1, 2):
        assert abs(np.trace(out[c]) / G.shape[0] - 1.0) < 0.15


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
