"""Kinship-estimator tests for IBS, windowed and LOCO kinships."""

import types

import numpy as np
import pytest

import phensim
from phensim.kinship import grm, ibs_kinship, windowed_kinships, iter_loco_kinships, loco_kinships


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


@pytest.mark.parametrize("constant", [1., .1, np.nan])
def test_monomorphic_complements_have_exactly_zero_information(constant):
    # The full and subset GEMMs need not round identically. Their difference
    # must not turn a constant/missing complement into a normalized kinship.
    G = np.array([[2, 1, 2, constant], [2, 1, 2, constant],
                  [2, 0, 0, constant], [0, 0, 2, constant]])
    expected = grm(G[:, 3:], scale=False)
    np.testing.assert_array_equal(expected, 0.)
    loco = phensim.loco_kinships(G, [1, 1, 1, 2], scale=False)
    np.testing.assert_array_equal(loco[1], expected)
    _, _, global_kinship = next(windowed_kinships(G, 3, 3, scale=False))
    np.testing.assert_array_equal(global_kinship, expected)
    with pytest.raises(ValueError, match="no positive diagonal scale"):
        phensim.loco_kinships(G, [1, 1, 1, 2])
    with pytest.raises(ValueError, match="no positive diagonal scale"):
        next(windowed_kinships(G, 3, 3))


def test_complements_retain_tiny_real_genotype_variation():
    G = np.array([[2, 1, 2, .1], [2, 1, 2, .1 + 1e-12],
                  [2, 0, 0, .1 + 2e-12], [0, 0, 2, .1 + 3e-12]])
    for scale in (False, True):
        expected = grm(G[:, 3:], scale=scale)
        loco = phensim.loco_kinships(G, [1, 1, 1, 2], scale=scale)
        np.testing.assert_allclose(loco[1], expected, rtol=1e-12, atol=1e-12)
        _, _, global_kinship = next(windowed_kinships(G, 3, 3, scale=scale))
        np.testing.assert_allclose(global_kinship, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("width,jump", [(20, 10), (10, 20), (20, 20)])
@pytest.mark.parametrize("scale", [False, True])
def test_window_complements_match_direct_grms(width, jump, scale):
    G = phensim.simulate_independent(20, 67, seed=43)
    G[:5, :8] = -1
    for wi, local, rest in phensim.windowed_kinships(G, width, jump, scale=scale):
        start, stop = wi * jump, min(wi * jump + width, G.shape[1])
        np.testing.assert_allclose(local, phensim.grm(G[:, start:stop], scale=scale), atol=1e-12)
        other = np.concatenate([G[:, :start], G[:, stop:]], axis=1)
        np.testing.assert_allclose(rest, phensim.grm(other, scale=scale), atol=1e-12)


def test_whole_genome_window_rejects_empty_complement():
    with pytest.raises(ValueError, match="outside"):
        list(phensim.windowed_kinships(np.ones((5, 10)), 10, 1))


def test_grm_scale_guards():
    with pytest.raises(ValueError, match="two samples"):
        grm(np.array([[0, 1, 2]]), scale=True)  # scaled singleton
    K = grm(np.array([[0, 1, 2]]), scale=False)  # unscaled singleton: finite
    assert K.shape == (1, 1) and np.isfinite(K).all()
    with pytest.raises(ValueError, match="diagonal scale"):
        grm(np.ones((5, 4)), scale=True)  # all monomorphic -> no scale
    np.testing.assert_array_equal(grm(np.ones((5, 4)), scale=False), 0)
    with pytest.raises(ValueError, match="finite"):
        grm(np.array([[np.inf, 0], [1, 2]]))
    with pytest.raises(ValueError, match="genotype matrix"):
        grm(np.ones((4, 0)))
    with pytest.raises(ValueError, match="genotype matrix"):
        ibs_kinship(np.ones(5))


def test_loco_requires_two_chromosomes_and_1d_labels():
    G = phensim.simulate_independent(20, 100, seed=2)
    with pytest.raises(ValueError, match="two chromosomes"):
        loco_kinships(G, np.ones(100, dtype=int))
    with pytest.raises(ValueError, match="one entry per variant"):
        loco_kinships(G, np.repeat([1, 2], 40))
    with pytest.raises(ValueError, match="1-D"):
        loco_kinships(G, np.ones((100, 1), dtype=int))


@pytest.mark.parametrize("scale", [False, True])
def test_loco_equals_leave_chromosome_out_grm(scale):
    G = phensim.simulate_independent(30, 200, seed=3).astype(float)
    G[:5, :20] = np.nan
    G[10:, 50:60] = -1
    chrom = np.repeat([1, 2, 3, 4], 50)
    out = loco_kinships(G, chrom, scale=scale)
    assert set(out) == {1, 2, 3, 4}
    for c in out:
        np.testing.assert_allclose(
            out[c], grm(G[:, chrom != c], scale=scale), atol=1e-10)


def test_loco_rejects_nonfinite_chromosome_labels():
    G = phensim.simulate_independent(20, 100, seed=2)
    chrom = np.repeat([1.0, 2.0], 50)
    chrom[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        loco_kinships(G, chrom)
    chrom = np.repeat([1.0, 2.0], 50)
    chrom[10] = np.inf
    with pytest.raises(ValueError, match="finite"):
        loco_kinships(G, chrom)


def test_iter_loco_kinships_lazy_and_equivalent():
    G = phensim.simulate_independent(30, 200, seed=4)
    chrom = np.repeat([1, 2, 3, 4], 50)
    gen = iter_loco_kinships(G, chrom, scale=False)
    assert isinstance(gen, types.GeneratorType)
    out = dict(gen)
    reference = loco_kinships(G, chrom, scale=False)
    assert out.keys() == reference.keys()
    for c in out:
        np.testing.assert_array_equal(out[c], reference[c])
    # validation is deferred to the first iteration, like any generator
    bad = iter_loco_kinships(G, np.ones(200, dtype=int))
    with pytest.raises(ValueError, match="two chromosomes"):
        next(bad)
