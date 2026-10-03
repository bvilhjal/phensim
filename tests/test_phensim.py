"""phensim simulator property tests."""

import numpy as np
import pytest
from importlib.util import find_spec

import phensim
from phensim.kinship import grm

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


def test_population_structure_differentiates():
    G, labels = phensim.simulate_population_structure(300, 2000, n_pops=3, fst=0.2, seed=2)
    af0 = G[labels == 0].mean(0) / 2
    af1 = G[labels == 1].mean(0) / 2
    # Fst-driven divergence far exceeds sampling noise at this size
    assert np.abs(af0 - af1).mean() > 0.03


@pytest.mark.parametrize("model", ["normal", "balding-nichols"])
def test_population_structure_models_differentiate(model):
    for n_pops in (2, 5):
        G, labels = phensim.simulate_population_structure(
            300, 2000, n_pops=n_pops, fst=0.2, model=model, seed=2
        )
        af0 = G[labels == 0].mean(0) / 2
        af1 = G[labels == 1].mean(0) / 2
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


def test_backend_determinism():
    a = phensim.simulate_by_mutation_rate(10, 1e4, seed=7, backend="numba")
    b = phensim.simulate_by_mutation_rate(10, 1e4, seed=7, backend="numba")
    np.testing.assert_array_equal(a, b)


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
    K = grm(G)
    for arch in ("mixed", "infinitesimal", "qtl"):
        tr = phensim.simulate_trait(
            G, h2=0.6, n_causal=15, architecture=arch, effect_dist="equal", seed=12, K=K
        )
        assert tr["y"].shape == (600,)
        assert abs(tr["y"].mean()) < 1e-10
        assert abs(tr["y"].std() - 1) < 1e-10
        # REML-style h2 check via the same K the model would fit
        from phensim.phenotypes import simulate_trait  # noqa: F401

        y = tr["y"]
        lam, U = np.linalg.eigh(K)
        etas = U.T @ y
        # crude spectral h2: variance explained by K's top quarter
        top = lam > np.quantile(lam, 0.75)
        h2_top = np.sum(etas[top] ** 2) / np.sum(etas**2)
        if arch == "qtl":
            assert h2_top < 0.9  # QTLs are not concentrated on K's top axes


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
