"""phensim simulator property tests."""

import numpy as np
import pytest
from importlib.util import find_spec

import phensim
from phensim.kinship import grm
from phensim._common import norm_isf

BACKENDS = ["numba", pytest.param("msprime", marks=pytest.mark.skipif(
    find_spec("msprime") is None, reason="msprime backend not installed"))]


# ------------------------------------------------------------------
# Genotype simulators
# ------------------------------------------------------------------


def test_independent_shapes_and_sfs():
    for dist in ("fixed", "beta", "uniform", "rare", "common"):
        G = phensim.simulate_independent(120, 500, maf=0.3, freq_dist=dist, seed=1)
        assert G.shape == (120, 500)
        assert G.dtype == np.int8
        assert set(np.unique(G)) <= {0, 1, 2}
        af = G.mean(0) / 2
        maf = np.minimum(af, 1 - af)  # sites may carry either orientation
        assert (maf >= 0).all() and (maf <= 0.5 + 1e-9).all()
        if dist == "fixed":
            assert (maf > 0).all()


@pytest.mark.parametrize("model", ["normal", "balding-nichols"])
def test_population_structure_models_differentiate(model):
    for n_pops in (2, 3, 5):
        G, labels = phensim.simulate_population_structure(
            300, 2000, n_pops=n_pops, fst=0.2, model=model, seed=2
        )
        af0 = G[labels == 0].mean(0) / 2
        af1 = G[labels == 1].mean(0) / 2
        # Fst-driven divergence far exceeds sampling noise at this size
        assert np.abs(af0 - af1).mean() > 0.03
        assert set(np.unique(labels)) == set(range(n_pops))


def test_population_structure_rejects_bad_args():
    with pytest.raises(ValueError, match="model"):
        phensim.simulate_population_structure(10, 20, model="beta")
    with pytest.raises(ValueError, match="fst"):
        phensim.simulate_population_structure(10, 20, fst=1.5)


def test_ar1_blocks_contract_and_ld():
    sizes = phensim.realistic_block_sizes(2000, 20, cv=0.9, seed=9)
    G, blocks = phensim.simulate_ar1_blocks(200, sizes, maf=0.3, rho=0.9, seed=1)
    assert G.shape == (200, 2000)
    assert G.dtype == np.int8
    assert set(np.unique(G)) <= {0, 1, 2}
    assert [b.size for b in blocks] == sizes.tolist()
    Z = G.astype(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    adj = np.mean(np.abs(np.diag(Z.T @ Z / Z.shape[0], 1)))
    far = np.mean(np.abs((Z[:, :-150].T @ Z[:, 150:] / Z.shape[0]).diagonal()))
    assert adj > far  # LD within blocks, decay across block edges
    af = G.mean(0) / 2
    assert abs(af.mean() - 0.3) < 0.02


def test_ar1_blocks_per_site_maf_and_determinism():
    rng = np.random.default_rng(3)
    maf = rng.uniform(0.05, 0.5, 300)
    a = phensim.simulate_ar1_blocks(150, [100, 200], maf=maf, rho=0.8, seed=4)
    b = phensim.simulate_ar1_blocks(150, [100, 200], maf=maf, rho=0.8, seed=4)
    np.testing.assert_array_equal(a[0], b[0])
    with pytest.raises(ValueError, match="maf"):
        phensim.simulate_ar1_blocks(10, [50, 60], maf=rng.uniform(0.1, 0.4, 105))


def test_realistic_block_sizes_partition():
    sizes = phensim.realistic_block_sizes(5000, 37, cv=1.2, seed=5)
    assert sizes.sum() == 5000
    assert (sizes >= 1).all()
    assert sizes.size == 37
    assert sizes.max() > 2 * sizes.mean()  # right-skewed by construction


@pytest.mark.parametrize("cv", [500., 1e160, np.finfo(float).max, np.float32(1e30), np.int64(4_000_000_000)])
@pytest.mark.parametrize("n_blocks", [1, 37])
def test_realistic_block_sizes_extreme_cv(cv, n_blocks):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        sizes = phensim.realistic_block_sizes(5000, n_blocks, cv=cv, seed=5)
    assert sizes.shape == (n_blocks,)
    assert sizes.min() >= 1 and sizes.sum() == 5000


def test_haplotype_blocks_have_ld():
    G = phensim.simulate_haplotype_blocks(200, 2000, block_size=100, seed=3)
    Z = G.astype(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    # average within-block vs across-block correlation
    adj = np.mean(np.abs(np.diag(Z.T @ Z / Z.shape[0], 1)[::10]))
    far = np.mean(np.abs((Z[:, :-110].T @ Z[:, 110:] / Z.shape[0]).diagonal()[::10]))
    assert adj > far


@pytest.mark.parametrize("backend", BACKENDS)
def test_coalescent_contract(backend):
    if backend == "numba" and not phensim.HAVE_NUMBA:
        pytest.skip("numba not installed")
    G, blocks = phensim.simulate_coalescent(80, 400, 200, seed=5, backend=backend)
    assert G.shape == (80, 400)
    assert G.dtype == np.int8
    assert len(blocks) == 2
    af = G.mean(0) / 2
    assert ((af > 0.01) & (af < 0.99)).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutation_rate_density_lever(backend):
    if backend == "numba" and not phensim.HAVE_NUMBA:
        pytest.skip("numba not installed")
    lo = phensim.simulate_by_mutation_rate(60, 2e6, mut_rate=1e-8, seed=6, backend=backend)
    hi = phensim.simulate_by_mutation_rate(60, 2e6, mut_rate=3e-8, seed=6, backend=backend)
    assert hi.shape[1] > lo.shape[1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_mutation_rate_draw_is_seeded_filtered_and_contiguous(backend):
    if backend == "numba" and not phensim.HAVE_NUMBA:
        pytest.skip("numba not installed")

    def draw(seed):
        return phensim.simulate_by_mutation_rate(
            40, 200_000, mut_rate=1e-7, min_maf=0.05, seed=seed, backend=backend)

    a, b, c = draw(3), draw(3), draw(4)
    np.testing.assert_array_equal(a, b)  # same seed, same segment
    assert not np.array_equal(a, c)  # different seed, different draw
    assert a.shape[0] == 40 and a.dtype == np.int8 and a.flags.c_contiguous
    af = a.mean(axis=0) / 2.0
    assert af.min() > 0.05 and af.max() < 0.95  # the MAF filter is applied


# ------------------------------------------------------------------
# Kinship
# ------------------------------------------------------------------


def test_grm_naive_agreement():
    rng = np.random.default_rng(8)
    G = rng.binomial(2, 0.4, (40, 300))
    K = grm(G)
    Z = G.astype(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    Kn = Z @ Z.T / 300
    n = 40
    off = (Kn.sum() - np.trace(Kn)) / (n * (n - 1))
    Kn = (Kn - off) / ((np.trace(Kn - off)) / n)
    np.testing.assert_allclose(K, Kn, atol=1e-10)


# ------------------------------------------------------------------
# Phenotype simulators
# ------------------------------------------------------------------


def test_trait_h2_and_architectures():
    G = phensim.simulate_haplotype_blocks(600, 4000, block_size=200, seed=11)
    for architecture, h2_bg, h2_qtl in (
            ("mixed", 0.3, 0.3),
            ("infinitesimal", 0.6, 0.0),
            ("qtl", 0.0, 0.6)):
        tr = phensim.simulate_trait(
            G, h2=0.6, n_causal=15, architecture=architecture,
            effect_dist="equal", seed=12)
        np.testing.assert_allclose(tr["q"].var(), h2_qtl, atol=1e-12)
        if h2_bg:
            assert tr["u"].std() > 0
        else:
            np.testing.assert_array_equal(tr["u"], 0)
        assert abs(tr["e"].var() - 0.4) < 0.1
        np.testing.assert_allclose(
            tr["u"] + tr["q"] + tr["e"], tr["liability"], atol=1e-14)
        assert abs(tr["y"].mean()) < 1e-10
        assert abs(tr["y"].std() - 1) < 1e-10


def test_binary_trait_prevalence():
    G = phensim.simulate_haplotype_blocks(5000, 1000, seed=13)
    tr = phensim.simulate_binary_trait(G, prevalence=0.1, h2=0.5, n_causal=10, seed=14)
    assert set(np.unique(tr["y"])) <= {0.0, 1.0}
    assert 0.07 < tr["y"].mean() < 0.13


def test_confounded_trait_structure_loads():
    G = phensim.simulate_haplotype_blocks(400, 2000, seed=15)
    K = grm(G)
    tr = phensim.simulate_confounded_trait(
        G, confounding_strength=0.8, h2=0.1, n_causal=5, seed=16, K=K
    )
    lam, U = np.linalg.eigh(K)
    lead = U[:, -1]
    r = np.corrcoef(tr["y"], lead)[0, 1]
    assert abs(r) > 0.5  # the phenotype is structure-driven by design


def test_gxe_trait_has_interaction():
    rng = np.random.default_rng(17)
    G = phensim.simulate_independent(400, 2000, seed=18)
    E = rng.standard_normal(400)
    E = (E - E.mean()) / E.std()
    tr = phensim.simulate_gxe_trait(
        G, E=E, h2=0.5, interaction_h2=0.3, n_causal=3, seed=19
    )
    Gd = G.astype(float)
    # the interaction term correlates with g * E at the causal locus
    c = tr["causal"][0]
    g = Gd[:, c] - Gd[:, c].mean()
    r_causal = abs(np.corrcoef(g * E, tr["liability"])[0, 1])
    noise = [abs(np.corrcoef((Gd[:, j] - Gd[:, j].mean()) * E, tr["liability"])[0, 1])
             for j in range(0, 2000, 137)]
    assert r_causal > np.quantile(noise, 0.99)


def test_correlated_traits_rg():
    G = phensim.simulate_haplotype_blocks(500, 3000, seed=20)
    tr = phensim.simulate_correlated_traits(G, h2_a=0.6, h2_b=0.6, rg=0.8, n_causal=10, seed=21)
    # realized correlation of the genetic values tracks the target
    r = np.corrcoef(tr["u_a"], tr["u_b"])[0, 1]
    assert r > 0.6


# ------------------------------------------------------------------
# Ascertainment and case/control bookkeeping
# ------------------------------------------------------------------


def test_ascertain_exact_counts():
    G = phensim.simulate_haplotype_blocks(2000, 500, seed=30)
    tr = phensim.simulate_binary_trait(G, prevalence=0.2, h2=0.4, n_causal=10, seed=31)
    asc = phensim.ascertain_case_control(tr, n_cases=80, n_controls=240, seed=32)
    assert asc["index"].size == 320
    assert asc["case_control"].sum() == 80
    assert asc["liability"].shape == (320,)
    # sampled liabilities line up with the population draw
    np.testing.assert_array_equal(asc["liability"], tr["liability"][asc["index"]])


def test_ascertain_rejects_impossible_counts():
    cc = np.zeros(100, dtype=np.int8)
    cc[:3] = 1
    with pytest.raises(ValueError, match="cases"):
        phensim.ascertain_case_control({"case_control": cc, "liability": np.zeros(100)}, 4, 10)
    with pytest.raises(ValueError, match="controls"):
        phensim.ascertain_case_control({"case_control": cc, "liability": np.zeros(100)}, 2, 98)


def test_n_eff_case_control():
    np.testing.assert_allclose(phensim.n_eff_case_control(500, 500), 1000.0)
    np.testing.assert_allclose(
        phensim.n_eff_case_control(500, 100000),
        4.0 / (1.0 / 500 + 1.0 / 100000),
    )
    np.testing.assert_allclose(
        phensim.n_eff_case_control([100, 200], [100, 200]), [200, 400]
    )
    with pytest.raises(ValueError):
        phensim.n_eff_case_control(0, 10)


def test_h2_liability_formula_and_warning():
    K = 0.01
    from statistics import NormalDist

    nd = NormalDist()
    t = -nd.inv_cdf(K)
    z = nd.pdf(t)
    factor_balanced = (K * (1 - K)) ** 2 / (z * z * 0.25)
    np.testing.assert_allclose(
        phensim.h2_liability(0.01, K, prop_cases=0.5), 0.01 * factor_balanced
    )
    with pytest.warns(UserWarning, match="prop_cases"):
        out = phensim.h2_liability(0.01, K)  # P=K default warns
    np.testing.assert_allclose(
        out, phensim.h2_liability(0.01, K, prop_cases=K)
    )
    with pytest.raises(ValueError):
        phensim.h2_liability(0.01, 1.5)


def test_write_plink_layout(tmp_path):
    G = phensim.simulate_independent(50, 120, seed=22)
    prefix = str(tmp_path / "sim")
    phensim.write_plink(G, prefix, chromosome=np.repeat([1, 2], 60))
    bed = open(f"{prefix}.bed", "rb").read()
    assert bed[:3] == b"\x6c\x1b\x01"
    assert len(bed) == 3 + ((50 + 3) // 4) * 120
    assert len(open(f"{prefix}.bim").readlines()) == 120
    assert len(open(f"{prefix}.fam").readlines()) == 50


@pytest.mark.parametrize("m", [1, 50, 150])
def test_haplotype_blocks_keep_partial_final_block(m):
    G = phensim.simulate_haplotype_blocks(20, m, block_size=100)
    assert G.shape == (20, m)
    assert set(np.unique(G)) <= {0, 1, 2}


@pytest.mark.parametrize("m,blocks", [(0, 10), (-1, 10), (10, 0), (10, -1), (2.5, 1)])
def test_block_geometry_rejects_impossible_counts(m, blocks):
    with pytest.raises(ValueError, match="positive integer"):
        phensim.realistic_block_sizes(m, blocks)


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


@pytest.mark.parametrize("call", [
    lambda: phensim.simulate_independent(0, 10),
    lambda: phensim.simulate_independent(10, 2.5),
    lambda: phensim.simulate_independent(10, -1),
    lambda: phensim.simulate_independent(True, 10),
    lambda: phensim.simulate_independent(10, 10, maf=0.7),
    lambda: phensim.simulate_independent(10, 10, maf=np.nan),
    lambda: phensim.simulate_independent(10, 10, maf=-0.1),
    lambda: phensim.simulate_population_structure(0, 10, n_pops=2),
    lambda: phensim.simulate_population_structure(10, 10, n_pops=0),
    lambda: phensim.simulate_population_structure(10, 10, n_pops=2.5),
    lambda: phensim.simulate_population_structure(10, 10, maf=1.5),
    lambda: phensim.simulate_ar1_blocks(0, [10]),
    lambda: phensim.simulate_ar1_blocks(5, []),
    lambda: phensim.simulate_ar1_blocks(5, [10.5]),
    lambda: phensim.simulate_ar1_blocks(5, [[10, 20]]),
    lambda: phensim.simulate_ar1_blocks(5, [0]),
    lambda: phensim.simulate_ar1_blocks(5, [10], rho=1.5),
    lambda: phensim.simulate_ar1_blocks(5, [10], rho=np.nan),
    lambda: phensim.simulate_ar1_blocks(5, [10], maf=1.2),
    lambda: phensim.simulate_ar1_blocks(5, [10], method="bogus"),
    lambda: phensim.simulate_coalescent(10, 100, block_size=200),
    lambda: phensim.simulate_coalescent(10, 100, mut_rate=0),
    lambda: phensim.simulate_coalescent(10, 100, min_maf=0.5),
    lambda: phensim.simulate_coalescent(10, 100, Ne=-1),
    lambda: phensim.simulate_coalescent(10, 100, recomb_rate=-1e-8),
    lambda: phensim.simulate_by_mutation_rate(0, 1e4),
    lambda: phensim.simulate_by_mutation_rate(10, 0.5),
    lambda: phensim.simulate_by_mutation_rate(10, np.inf),
])
def test_genotype_simulator_argument_validation(call):
    with pytest.raises(ValueError):
        call()


def test_ar1_validation_precedes_rng_and_allocation():
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="block_sizes"):
        phensim.simulate_ar1_blocks(5, [10.5], seed=rng)
    with pytest.raises(ValueError, match="rho"):
        phensim.simulate_ar1_blocks(5, [10], rho=2, seed=rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("block_sizes", [
    np.array([np.iinfo(np.intp).max], dtype=np.uint64) + 1,  # single overflow
    np.array([2**62, 2**62], dtype=np.int64),                # sum overflow
    np.array([2**64 - 1], dtype=np.uint64),                  # fits uint64, not intp
])
def test_ar1_block_lengths_reject_overflow(block_sizes):
    with pytest.raises(ValueError, match="block_sizes"):
        phensim.simulate_ar1_blocks(5, block_sizes, seed=1)


def test_independent_ignores_maf_for_nonfixed_freq_dist():
    G = phensim.simulate_independent(5, 20, maf=99, freq_dist="uniform", seed=1)
    assert G.shape == (5, 20)
    with pytest.raises(ValueError, match="maf"):
        phensim.simulate_independent(5, 20, maf=99, freq_dist="fixed")


def _legacy_ar1(n, block_sizes, maf, rho, seed):
    """Pre-change implementation; the default method's bit-exact oracle."""
    rng = np.random.default_rng(seed)
    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    m = int(block_sizes.sum())
    maf = (np.full(m, float(maf)) if np.ndim(maf) == 0
           else np.asarray(maf, dtype=float))
    G = np.empty((n, m), dtype=np.int8)
    col = 0
    for k in block_sizes:
        k = int(k)
        idx = np.arange(k)
        corr = rho ** np.abs(idx[:, None] - idx[None, :])
        chol = np.linalg.cholesky(corr + 1e-8 * np.eye(k))
        thr = norm_isf(maf[col:col + k])
        hap_sum = np.zeros((n, k))
        for _ in range(2):
            z = rng.standard_normal((n, k)) @ chol.T
            hap_sum += (z > thr)
        G[:, col:col + k] = hap_sum.astype(np.int8)
        col += k
    return G


@pytest.mark.parametrize("maf", [0.3, "per-site"])
def test_ar1_cholesky_remains_bit_identical(maf):
    sizes = np.array([40, 60])
    maf_arg = (0.3 if maf == 0.3
               else np.random.default_rng(0).uniform(0.05, 0.5, 100))
    default = phensim.simulate_ar1_blocks(30, sizes, maf=maf_arg, rho=0.8, seed=7)
    np.testing.assert_array_equal(
        default[0], _legacy_ar1(30, sizes, maf_arg, 0.8, 7))
    explicit = phensim.simulate_ar1_blocks(
        30, sizes, maf=maf_arg, rho=0.8, seed=7, method="cholesky")
    np.testing.assert_array_equal(default[0], explicit[0])


def _scan_ar1_oracle(n, block_sizes, maf, rho, seed):
    """Direct forward-recursion oracle for ``method='scan'``."""
    rng = np.random.default_rng(seed)
    block_sizes = np.asarray(block_sizes, dtype=np.int64)
    m = int(block_sizes.sum())
    maf = (np.full(m, float(maf)) if np.ndim(maf) == 0
           else np.asarray(maf, dtype=float))
    sd = np.sqrt(1.0 - rho * rho)
    G = np.empty((n, m), dtype=np.int8)
    col = 0
    for k in block_sizes:
        k = int(k)
        thr = norm_isf(maf[col:col + k])
        hap_sum = np.zeros((n, k))
        for _ in range(2):
            eps = rng.standard_normal((n, k))
            z = np.empty((n, k))
            z[:, 0] = eps[:, 0]
            for j in range(1, k):
                z[:, j] = rho * z[:, j - 1] + sd * eps[:, j]
            hap_sum += (z > thr)
        G[:, col:col + k] = hap_sum.astype(np.int8)
        col += k
    return G


def test_ar1_scan_matches_forward_recursion_oracle():
    maf = np.random.default_rng(0).uniform(0.05, 0.5, 100)
    scan = phensim.simulate_ar1_blocks(
        30, [40, 60], maf=maf, rho=0.8, seed=7, method="scan")
    np.testing.assert_array_equal(
        scan[0], _scan_ar1_oracle(30, [40, 60], maf, 0.8, 7))


@pytest.mark.parametrize("rho", [1.0, -1.0])
def test_ar1_scan_rho_endpoints_deterministic(rho):
    G, _ = phensim.simulate_ar1_blocks(
        40, [30], maf=0.3, rho=rho, seed=1, method="scan")
    if rho > 0:
        assert (G == G[:, :1]).all()  # z_j == z_0 throughout the block
    else:
        assert (G[:, ::2] == G[:, :1]).all()    # even columns z = +z_0
        assert (G[:, 1::2] == G[:, 1:2]).all()  # odd columns z = -z_0


def test_ar1_scan_skips_cholesky(monkeypatch):
    calls, real = [], np.linalg.cholesky

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "cholesky", spy)
    phensim.simulate_ar1_blocks(20, [50], rho=0.5, seed=1, method="scan")
    assert not calls
    phensim.simulate_ar1_blocks(20, [50], rho=0.5, seed=1)
    assert calls


def test_ar1_scan_and_cholesky_share_the_ld_shape():
    def mean_lag1(method):
        G, blocks = phensim.simulate_ar1_blocks(
            2000, [400, 400, 400], maf=0.3, rho=0.9, seed=11, method=method)
        Z = G.astype(float)
        Z -= Z.mean(0)
        Z /= np.where(Z.std(0) > 0, Z.std(0), 1)
        R = Z.T @ Z / Z.shape[0]
        return np.mean([np.diag(R, 1)[b[:-1]].mean() for b in blocks])

    scan, chol = mean_lag1("scan"), mean_lag1("cholesky")
    assert scan > 0.5 and abs(scan - chol) < 0.05


def test_by_mutation_rate_allows_zero_mutation_rate():
    G = phensim.simulate_by_mutation_rate(5, 1e4, mut_rate=0, seed=1)
    assert G.shape == (5, 0)
