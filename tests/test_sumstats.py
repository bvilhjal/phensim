"""Summary-statistic simulator tests."""

import numpy as np
import pytest

import phensim
from phensim.sumstats import simulate_effects, simulate_sumstats, \
    simulate_sumstats_pair, gwas_scan, shake_ld

@pytest.mark.parametrize("kwargs", [dict(p=True), dict(p=np.bool_(True)), dict(p=.5, n_shared=1)])
def test_effects_pair_rejects_ambiguous_causal_layout(kwargs):
    with pytest.raises(ValueError):
        phensim.simulate_effects_pair([(np.eye(4), np.arange(4))], **kwargs)

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
    m = sum(ix.size for _, ix in blocks)
    for kwargs in (dict(p=0.0), dict(p=1.5), dict(p=np.nan), dict(p=True),
                   dict(p=np.full(m - 1, 0.1)), dict(p=np.full(m, 1.2)), dict(p=np.zeros(m)),
                   dict(p=0.1, n_causal=3), dict(p=0.1, architecture="polygenic")):
        with pytest.raises(ValueError, match=r"\bp\b"):
            simulate_effects(blocks, **kwargs)


def test_effects_causal_probability(blocks):
    blocks, _ = blocks
    m = sum(ix.size for _, ix in blocks)
    for arch in ("sparse", "equal"):
        beta = simulate_effects(blocks, h2=0.3, p=0.1, architecture=arch, seed=4)
        np.testing.assert_allclose(_var(beta, blocks), 0.3, rtol=1e-10)
        assert 0 < np.count_nonzero(beta) < m
    # per-variant probabilities confine the causal set, the forced variant too
    prob = np.zeros(m)
    prob[:20] = 0.5
    beta = simulate_effects(blocks, h2=0.3, p=prob, seed=5)
    assert np.flatnonzero(beta).max() < 20
    prob = np.zeros(m)
    prob[7] = 1e-12
    np.testing.assert_array_equal(np.flatnonzero(simulate_effects(blocks, p=prob, seed=5)), [7])
    # p=1 still takes the uniform draw (ldpred3's order), selecting every variant
    assert np.count_nonzero(simulate_effects(blocks, p=1.0, seed=5)) == m
    rng = np.random.default_rng(6)
    simulate_effects(blocks, p=1.0, seed=rng)
    expected = np.random.default_rng(6)
    expected.random(m)
    expected.standard_normal(m)
    assert rng.random() == expected.random()
    # the draw is independent per variant, so the causal count tracks p * m
    counts = [np.count_nonzero(simulate_effects(blocks, p=0.2, seed=s)) for s in range(200)]
    assert abs(np.mean(counts) - 0.2 * m) < 4 * np.sqrt(0.2 * 0.8 * m / 200)


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
    # Perfect noise correlation: same beta -> identical draws.
    a, b = simulate_sumstats_pair(beta, beta, blocks, n, noise_correlation=1.0, seed=3)
    np.testing.assert_allclose(a, b, rtol=1e-12)
    # intermediate overlap: the residual correlation tracks it
    reps = np.stack(
        [simulate_sumstats_pair(beta, beta * 0, blocks, n, noise_correlation=0.6, seed=s)
         for s in range(400)]
    )
    noise_a = reps[:, 0] - np.tile(0, (400, 150))
    noise_b = reps[:, 1]
    ca = np.corrcoef(noise_a[:, 5], noise_b[:, 5])[0, 1]
    assert 0.4 < ca < 0.8
    with pytest.raises(ValueError):
        simulate_sumstats_pair(beta, beta, blocks, n, noise_correlation=1.5)


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


def _consumers(blocks):
    beta = np.zeros(sum(np.asarray(ix).size for _R, ix in blocks))
    return [lambda: phensim.simulate_effects(blocks, architecture="polygenic"),
            lambda: phensim.simulate_sumstats(beta, blocks, 100),
            lambda: phensim.simulate_sumstats_pair(beta, beta, blocks, 100),
            lambda: phensim.shake_ld(blocks, 10),
            lambda: phensim.simulate_gaussian_individuals(blocks, 3)]


@pytest.mark.parametrize("blocks", [
    [], [(np.eye(2), [0, 2])], [(np.eye(2), [0, 0])],
    [(np.eye(1), [0]), (np.eye(1), [0])], [(np.eye(1), [-1])],
    [(np.eye(1), [10**12])], [(np.eye(1), [0.0])], [(np.eye(1), [True])],
    [(np.eye(2), [0])],
])
def test_ld_rejects_invalid_layouts(blocks):
    with pytest.raises(ValueError):
        phensim.prepare_blocks(blocks)


@pytest.mark.parametrize("R", [
    [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]],
    [[1, np.nan, 0], [np.nan, 1, 0], [0, 0, 1]],
    [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]],
    [[1, 2, 0], [2, 1, 0], [0, 0, 1]],
])
def test_ld_rejects_invalid_correlations(R):
    with pytest.raises(ValueError):
        phensim.prepare_blocks([(np.array(R), np.arange(3))])


def test_raw_consumers_use_shared_ld_validation():
    bad = np.array([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]])
    for consume in _consumers([(bad, np.arange(3))]):
        with pytest.raises(ValueError, match="semidefinite"):
            consume()


def test_singular_ld_preserves_the_null_space():
    blocks = [(np.ones((2, 2)), np.arange(2))]
    for seed in range(10):
        bhat = phensim.simulate_sumstats(np.zeros(2), blocks, 100, seed=seed)
        assert abs(bhat[0] - bhat[1]) < 1e-14
    np.testing.assert_allclose(phensim.shake_ld(blocks, 20)[0][0], 1)


def test_ld_unsorted_partition_keeps_variant_alignment():
    blocks = [(np.array([[1, 0.2], [0.2, 1]]), np.array([2, 0])), (np.eye(1), [1])]
    result = phensim.simulate_sumstats(np.array([1., 2., 3.]), blocks, 1e300)
    np.testing.assert_allclose(result, [1.6, 2, 3.2])


@pytest.mark.parametrize("n", [0, -1, np.nan, np.inf, [100], [100, 0]])
def test_sumstats_rejects_invalid_sample_sizes(n):
    with pytest.raises(ValueError, match="n must"):
        phensim.simulate_sumstats(np.zeros(2), [(np.eye(2), np.arange(2))], n)


def test_noise_correlation_name_and_legacy_warning():
    blocks = [(np.eye(3), np.arange(3))]
    args = (np.zeros(3), np.zeros(3), blocks, 100)
    expected = phensim.simulate_sumstats_pair(*args, noise_correlation=-0.6)
    with pytest.warns(FutureWarning, match="not participant overlap"):
        legacy = phensim.simulate_sumstats_pair(*args, overlap=-0.6)
    np.testing.assert_array_equal(legacy, expected)
    with pytest.raises(ValueError, match="not both"):
        phensim.simulate_sumstats_pair(*args, noise_correlation=0, overlap=1)


def test_maf_architecture_uses_the_ldpred3_alpha_convention():
    # Same seed, same normals as the unscaled 'polygenic' draw, so the ratio
    # isolates the MAF scaling. alpha=-1 is flat on the standardized scale.
    f = np.linspace(0.01, 0.5, 40)
    blocks = [(np.eye(40), np.arange(40))]
    flat = phensim.simulate_effects(blocks, architecture="polygenic", seed=3)
    for alpha in (-1.0, -0.3, 0.0):
        beta = phensim.simulate_effects(blocks, architecture="maf", maf=f, alpha=alpha, seed=3)
        ratio = beta / flat
        H = 2 * f * (1 - f)
        np.testing.assert_allclose(ratio / ratio[0], (H / H[0]) ** ((1 + alpha) / 2), rtol=1e-12)


def test_float32_ld_is_judged_at_float32_precision():
    rng = np.random.default_rng(0)
    k = 60
    X = rng.standard_normal((30, k))                     # singular panel, n_ref < k
    X = (X - X.mean(0)) / X.std(0)
    R = (X.T @ X / 30).astype(np.float32)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1)
    asym = R.copy()
    asym[3, 7] = np.nextafter(asym[3, 7], np.float32(1))
    diag = R.copy()
    diag[5, 5] = np.float32(1) - 2 * np.finfo(np.float32).eps / 2
    for block in (R, asym, diag):
        blocks = [(block, np.arange(k))]
        phensim.prepare_blocks(blocks)
        for consume in _consumers(blocks):
            consume()
