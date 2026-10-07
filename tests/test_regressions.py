"""Independent oracles for the October 2026 adversarial-review failures."""

import math

import numpy as np
import pytest

import phensim


def _standardize(G):
    G = np.asarray(G, dtype=float)
    return (G - G.mean(0)) / G.std(0)


def test_plink_decodes_all_four_states_and_padding(tmp_path):
    G = np.array([[0, 2], [1, 0], [2, 1], [-1, 2], [np.nan, 1], [1, -1], [2, 0]])
    prefix = str(tmp_path / "four-states")
    phensim.write_plink(G, prefix)
    bed = (tmp_path / "four-states.bed").read_bytes()
    # Independent BED decoder, counting BIM allele 2: 00,01,10,11.
    lookup = np.array([0, -1, 1, 2])
    decoded = np.empty(G.shape)
    stride = (G.shape[0] + 3) // 4
    for j in range(G.shape[1]):
        for i in range(G.shape[0]):
            code = (bed[3 + j * stride + i // 4] >> (2 * (i % 4))) & 3
            decoded[i, j] = lookup[code]
        assert bed[3 + (j + 1) * stride - 1] >> 6 == 0
    np.testing.assert_array_equal(decoded, np.where(np.isnan(G), -1, G))


@pytest.mark.parametrize("bad", [0.5, 3, np.inf])
def test_plink_rejects_unrepresentable_calls_before_writing(tmp_path, bad):
    with pytest.raises(ValueError, match="genotypes"):
        phensim.write_plink(np.array([[bad]]), str(tmp_path / "bad"))
    assert not list(tmp_path.iterdir())


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


def _consumers(blocks):
    return [lambda: phensim.simulate_effects(blocks, architecture="polygenic"),
            lambda: phensim.simulate_sumstats(np.zeros(3), blocks, 100),
            lambda: phensim.simulate_sumstats_pair(np.zeros(3), np.zeros(3), blocks, 100),
            lambda: phensim.shake_ld(blocks, None)]


@pytest.mark.parametrize("blocks", [
    [], [(np.eye(2), [0, 2])], [(np.eye(2), [0, 0])],
    [(np.eye(1), [0]), (np.eye(1), [0])], [(np.eye(1), [-1])],
    [(np.eye(1), [10**12])], [(np.eye(1), [0.0])], [(np.eye(1), [True])],
    [(np.eye(2), [0])],
])
def test_ld_rejects_invalid_layouts(blocks):
    for consume in _consumers(blocks):
        with pytest.raises(ValueError):
            consume()


@pytest.mark.parametrize("R", [
    [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]],
    [[1, np.nan, 0], [np.nan, 1, 0], [0, 0, 1]],
    [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]],
    [[1, 2, 0], [2, 1, 0], [0, 0, 1]],
])
def test_ld_rejects_invalid_correlations_for_every_consumer(R):
    for consume in _consumers([(np.array(R), np.arange(3))]):
        with pytest.raises(ValueError):
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


@pytest.mark.parametrize("m", [1, 50, 150])
def test_haplotype_blocks_keep_partial_final_block(m):
    G = phensim.simulate_haplotype_blocks(20, m, block_size=100)
    assert G.shape == (20, m)
    assert set(np.unique(G)) <= {0, 1, 2}


@pytest.mark.parametrize("m,blocks", [(0, 10), (-1, 10), (10, 0), (10, -1), (2.5, 1)])
def test_block_geometry_rejects_impossible_counts(m, blocks):
    with pytest.raises(ValueError, match="positive integer"):
        phensim.realistic_block_sizes(m, blocks)


def test_birth_times_reject_self_parent_and_incompatible_generations():
    with pytest.raises(ValueError, match="own parent"):
        phensim.pedigree_birth_times(["a"], ["a"], [None])
    # a parents b; a and b are also co-parents of c. Union-find must not
    # erase the impossible a -> b edge when it merges the co-parents.
    with pytest.raises(ValueError, match="discrete generations"):
        phensim.pedigree_birth_times(["a", "b", "c"], [None, "a", "a"], [None, None, "b"])


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
        phensim.simulate_effects(blocks, architecture="polygenic")
        phensim.simulate_sumstats(np.zeros(k), blocks, 100)
        phensim.simulate_sumstats_pair(np.zeros(k), np.zeros(k), blocks, 100)
        phensim.shake_ld(blocks, 10)
    bad = np.array([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]], np.float32)
    for consume in _consumers([(bad, np.arange(3))]):
        with pytest.raises(ValueError, match="semidefinite"):
            consume()


def test_pedigree_accepts_zero_as_a_listed_id():
    A = phensim.kinship_from_pedigree([0, 1, 2], [None, None, 0], [None, None, 1])
    np.testing.assert_allclose(A[2, :2], 0.5)
    A = phensim.kinship_from_pedigree(["0", "1", "2"], [None, None, "0"], [None, None, "1"])
    np.testing.assert_allclose(A[2], [0.5, 0.5, 1])
    for missing in (None, "", np.nan):
        with pytest.raises(ValueError, match="missing"):
            phensim.kinship_from_pedigree(["a", missing], [None, None], [None, None])


@pytest.mark.parametrize("seed", [0, -1, 2**31, 2**32 + 5, True, 1.0])
def test_coalescent_seed_range_is_backend_neutral(seed):
    # The built-in kernel masks seeds to 31 bits: 5 and 5 + 2**31 used to
    # give identical draws, while msprime rejected 0 and >= 2**32.
    with pytest.raises(ValueError, match="seed"):
        phensim.simulate_by_mutation_rate(4, 1e4, seed=seed, backend="numba")
    assert phensim.simulate_by_mutation_rate(4, 1e4, seed=2**31 - 1, backend="numba").ndim == 2


def test_ar1_blocks_accept_counted_allele_frequencies_above_one_half():
    f = np.r_[np.full(20, 0.1), np.full(20, 0.8)]
    G, _ = phensim.simulate_ar1_blocks(20_000, [40], maf=f, rho=0.5, seed=1)
    np.testing.assert_allclose(G.mean(0) / 2, f, atol=0.01)
    # f and 1 - f are mirror images: the same draws count the other allele.
    lo, _ = phensim.simulate_ar1_blocks(500, [10], maf=0.3, rho=0.5, seed=2)
    hi, _ = phensim.simulate_ar1_blocks(500, [10], maf=0.7, rho=0.5, seed=2)
    assert abs((lo.mean() + hi.mean()) / 2 - 1.0) < 0.05
