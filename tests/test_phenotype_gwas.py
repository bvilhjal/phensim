"""Phenotype truth and GWAS scan regression oracles."""

import math

import numpy as np
import pytest

import phensim


def _standardize(G):
    G = np.asarray(G, dtype=float)
    return (G - G.mean(0)) / G.std(0)


@pytest.mark.parametrize("architecture,h2", [("qtl", 1), ("qtl", 0.4), ("mixed", 0.6)])
def test_reported_effects_reconstruct_liability_components(architecture, h2):
    G = phensim.simulate_independent(100, 150, seed=31)
    tr = phensim.simulate_trait(G, h2=h2, architecture=architecture, n_causal=8, seed=32)
    q = _standardize(G[:, tr["causal"]]) @ tr["effects"]
    np.testing.assert_allclose(q, tr["q"], atol=1e-14)
    np.testing.assert_allclose(q.var(), h2 if architecture == "qtl" else h2 / 2)
    np.testing.assert_allclose(tr["u"] + q + tr["e"], tr["liability"], atol=1e-14)


@pytest.mark.parametrize("rg", [-1, 1])
def test_total_genetic_correlation_endpoints(rg):
    G = phensim.simulate_independent(100, 150, seed=33)
    tr = phensim.simulate_correlated_traits(G, h2_a=0.4, h2_b=0.8, rg=rg, seed=34)
    np.testing.assert_allclose(tr["g_b"], rg * np.sqrt(2) * tr["g_a"], atol=1e-14)
    fully_genetic = phensim.simulate_correlated_traits(G, h2_a=1, h2_b=1, rg=rg, seed=34)
    np.testing.assert_allclose(fully_genetic["y_b"], rg * fully_genetic["y_a"], atol=1e-14)


def test_total_genetic_correlation_interior():
    G = phensim.simulate_independent(80, 200, seed=35)
    sums = np.zeros(3)
    for seed in range(60):
        tr = phensim.simulate_correlated_traits(G, h2_a=1, h2_b=1, rg=0.6, seed=seed)
        a, b = tr["g_a"] - tr["g_a"].mean(), tr["g_b"] - tr["g_b"].mean()
        sums += [a @ a, b @ b, a @ b]
    assert abs(sums[2] / np.sqrt(sums[0] * sums[1]) - 0.6) < 0.06


@pytest.mark.parametrize("h2,interaction", [(1, 0.2), (0.5, 0.2), (0.5, 0)])
def test_gxe_truth_and_residual_variance(h2, interaction):
    G = phensim.simulate_independent(200, 100, seed=36)
    tr = phensim.simulate_gxe_trait(G, h2=h2, interaction_h2=interaction, seed=37)
    Z = _standardize(G[:, tr["causal"]])
    q = Z @ tr["effects"]
    gx = (Z * tr["environment"][:, None]) @ tr["interaction_effects"]
    np.testing.assert_allclose(gx, tr["interaction"], atol=1e-14)
    np.testing.assert_allclose(gx.var(), interaction, atol=1e-14)
    np.testing.assert_allclose(tr["liability"], tr["u"] + q + gx + tr["e"], atol=1e-14)
    if h2 == 1:
        np.testing.assert_array_equal(tr["e"], 0)
    else:
        assert abs(tr["e"].var() - (1-h2)) < 0.1


def test_gxe_environment_and_residual_do_not_reuse_innovations():
    G = np.zeros((1000, 1))
    tr = phensim.simulate_gxe_trait(G, h2=0, interaction_h2=0, n_causal=0, seed=38)
    assert abs(np.corrcoef(tr["environment"], tr["e"])[0, 1]) < 0.1


def test_confounded_effects_remain_on_liability_scale():
    G = phensim.simulate_independent(80, 100, seed=39)
    tr = phensim.simulate_confounded_trait(G, seed=40)
    q = _standardize(G[:, tr["causal"]]) @ tr["effects"]
    np.testing.assert_allclose(q, tr["q"], atol=1e-14)
    np.testing.assert_allclose(tr["liability"], tr["structure"] + tr["u"] + q + tr["e"], atol=1e-14)


def test_gwas_missingness_matches_direct_ols():
    rng = np.random.default_rng(41)
    G = rng.binomial(2, 0.4, (70, 5)).astype(float)
    G[:20, 0] = -1
    G[::2, 1] = np.nan
    G[20:, 2] = -1
    y = rng.normal(size=70) + np.arange(70)/30
    out = phensim.gwas_scan(G, y)
    for j in range(G.shape[1]):
        called = np.isfinite(G[:, j]) & (G[:, j] >= 0)
        x, target = _standardize(G[called, j]), _standardize(y[called])
        design = np.column_stack([np.ones(x.size), x])
        fit = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ fit
        se = np.sqrt((residual @ residual) / (x.size-2) * np.linalg.inv(design.T @ design)[1, 1])
        np.testing.assert_allclose([out["beta"][j], out["se"][j], out["z"][j]],
                                   [fit[1], se, fit[1]/se], atol=1e-12)
        np.testing.assert_allclose(out["p"][j], math.erfc(abs(fit[1]/se)/np.sqrt(2)))


def test_gwas_perfect_association_and_untestable_variants():
    G = np.array([[0, 0, -1, 0], [2, 0, -1, 2], [0, 0, -1, -1], [2, 0, -1, -1.]])
    for sign in [-1, 1]:
        out = phensim.gwas_scan(G, sign * G[:, 0])
        assert out["z"][0] == sign * np.inf
        assert out["se"][0] == 0 and out["p"][0] == 0
        for value in out.values():
            assert np.isnan(value[1:]).all()


def test_gwas_missing_null_is_calibrated():
    rng = np.random.default_rng(42)
    G = rng.binomial(2, 0.3, (800, 1500)).astype(float)
    G[rng.random(G.shape) < 0.75] = -1
    out = phensim.gwas_scan(G, rng.normal(size=800))
    assert 0.85 < np.mean(out["z"]**2) < 1.2


def test_gwas_constant_called_values_are_untestable():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    G = np.tile(rng.uniform(0.01, 1.99, 20), (300, 1))   # monomorphic dosages
    G[rng.random(G.shape) < 0.1] = np.nan
    G[:, 0] = rng.integers(0, 3, 300)                    # one real variant
    out = phensim.gwas_scan(G, y)
    assert np.isfinite(out["z"][0]) and np.isnan(out["z"][1:]).all()
    yc = np.r_[np.full(150, 0.3), rng.normal(size=150)]  # constant where called
    Gc = rng.integers(0, 3, (300, 20)).astype(float)
    Gc[150:] = np.nan
    assert np.isnan(phensim.gwas_scan(Gc, yc)["p"]).all()
