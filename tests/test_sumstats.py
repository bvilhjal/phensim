"""Summary-statistic simulator tests."""

import numpy as np
import pytest

import phensim
from phensim.sumstats import simulate_effects, simulate_sumstats, \
    simulate_sumstats_pair, gwas_scan, shake_ld


@pytest.fixture(scope="module")
def blocks():
    """Two AR(1) LD blocks with known correlation structure."""
    rng = np.random.default_rng(0)
    sizes = [60, 90]
    m = sum(sizes)
    maf = rng.uniform(0.1, 0.5, m)
    G, _ = phensim.simulate_ar1_blocks(2000, sizes, maf=maf, rho=0.8, seed=0)
    Z = G.astype(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    out, col = [], 0
    for k in sizes:
        Zb = Z[:, col:col + k]
        out.append((Zb.T @ Zb / Z.shape[0], np.arange(col, col + k)))
        col += k
    return out, maf


def _var(beta, blocks):
    return sum(beta[ix] @ (R @ beta[ix]) for R, ix in blocks)


def test_effects_hit_h2_for_every_architecture(blocks):
    blocks, maf = blocks
    kwargs = [
        dict(architecture="sparse", n_causal=10),
        dict(architecture="polygenic"),
        dict(architecture="maf", maf=maf),
        dict(architecture="equal", n_causal=15),
    ]
    for kw in kwargs:
        beta = simulate_effects(blocks, h2=0.3, seed=7, **kw)
        np.testing.assert_allclose(_var(beta, blocks), 0.3, rtol=1e-10)
    beta = simulate_effects(blocks, h2=0.3, n_causal=10, seed=7)
    assert np.count_nonzero(beta) == 10


def test_effects_validation(blocks):
    blocks, maf = blocks
    with pytest.raises(ValueError, match="architecture"):
        simulate_effects(blocks, architecture="monogenic", n_causal=3)
    with pytest.raises(ValueError, match="n_causal"):
        simulate_effects(blocks, architecture="sparse")
    with pytest.raises(ValueError, match="maf"):
        simulate_effects(blocks, architecture="maf")


def test_sumstats_oracle_moments(blocks):
    blocks, _ = blocks
    beta = simulate_effects(blocks, h2=0.4, n_causal=12, seed=1)
    R = np.zeros((150, 150))
    for blk, ix in blocks:
        R[np.ix_(ix, ix)] = blk
    truth = R @ beta
    n = 400
    reps = np.stack(
        [simulate_sumstats(beta, blocks, n, seed=s) for s in range(240)]
    )
    np.testing.assert_allclose(reps.mean(0), truth, atol=0.05)
    np.testing.assert_allclose(reps.var(0), np.diag(R) / n, atol=0.01)


def test_sumstats_per_variant_n(blocks):
    blocks, _ = blocks
    beta = simulate_effects(blocks, h2=0.4, n_causal=12, seed=1)
    m = beta.size
    nvec = np.full(m, 5000.0)
    nvec[10:40] = 500.0  # noisy segment
    reps = np.stack(
        [simulate_sumstats(beta, blocks, nvec, seed=s) for s in range(120)]
    )
    assert reps.var(0)[20] > 5 * reps.var(0)[60]


def test_sumstats_pair_overlap(blocks):
    blocks, _ = blocks
    beta = simulate_effects(blocks, h2=0.3, n_causal=20, seed=2)
    n = 1000
    # perfect overlap: same beta -> identical draws (shared innovations)
    a, b = simulate_sumstats_pair(beta, beta, blocks, n, overlap=1.0, seed=3)
    np.testing.assert_allclose(a, b, rtol=1e-12)
    # intermediate overlap: the residual correlation tracks it
    reps = np.stack(
        [simulate_sumstats_pair(beta, beta * 0, blocks, n, overlap=0.6, seed=s)
         for s in range(400)]
    )
    noise_a = reps[:, 0] - np.tile(0, (400, 150))
    noise_b = reps[:, 1]
    ca = np.corrcoef(noise_a[:, 5], noise_b[:, 5])[0, 1]
    assert 0.4 < ca < 0.8
    with pytest.raises(ValueError):
        simulate_sumstats_pair(beta, beta, blocks, n, overlap=1.5)


def test_gwas_scan_matches_manual(blocks):
    blocks, maf = blocks
    G, _ = phensim.simulate_ar1_blocks(800, [60, 90], maf=maf, rho=0.6, seed=4)
    rng = np.random.default_rng(40)
    # null: y independent of G -> z-scores match the manual marginal scan,
    # p-values roughly uniform
    y0 = rng.standard_normal(800)
    out = gwas_scan(G, y0)
    Z = G.astype(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    y = (y0 - y0.mean()) / y0.std()
    r_manual = Z.T @ y / len(y)
    np.testing.assert_allclose(out["beta"], r_manual, atol=1e-10)
    se_manual = np.sqrt((1 - r_manual**2) / (len(y) - 2))
    np.testing.assert_allclose(out["se"], se_manual, atol=1e-10)
    np.testing.assert_allclose(out["z"], r_manual / se_manual, atol=1e-8)
    assert 0.3 < np.median(out["p"]) < 0.7
    # signal: a QTL trait makes the causal variants the extreme z-scores
    tr = phensim.simulate_trait(G, h2=0.6, n_causal=8, architecture="qtl", seed=5)
    out = gwas_scan(G, tr["y"])
    assert np.max(np.abs(out["z"][tr["causal"]])) > np.quantile(np.abs(out["z"]), 0.999)


def test_gwas_scan_missing_genotypes():
    rng = np.random.default_rng(6)
    G = rng.binomial(2, 0.4, (200, 40)).astype(np.int8)
    G[:50, :10] = -1  # missing calls in the first ten variants
    y = rng.standard_normal(200)
    out = gwas_scan(G, y)
    assert np.isfinite(out["z"]).all()
    assert (out["p"] >= 0).all() and (out["p"] <= 1).all()


def test_shake_ld_noise_shrinks_with_panel(blocks):
    blocks, _ = blocks
    R = blocks[0][0]
    exact = shake_ld(blocks, None)
    np.testing.assert_allclose(exact[0][0], (R + R.T) / 2, atol=1e-12)
    small = shake_ld(blocks, 40, seed=8)[0][0]
    big = shake_ld(blocks, 20000, seed=8)[0][0]
    assert np.abs(small - R).max() > np.abs(big - R).max()
    np.testing.assert_allclose(big, R, atol=0.06)
    assert np.allclose(np.diag(small), 1.0)
    np.testing.assert_allclose(small, small.T, atol=1e-12)
    with pytest.raises(ValueError):
        shake_ld(blocks, 1)
