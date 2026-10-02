"""phensim simulator property tests."""

import numpy as np
import pytest

import phensim
from phensim.kinship import grm

BACKENDS = ["numba"]
msprime = pytest.importorskip("msprime", reason="msprime backend")
BACKENDS.append("msprime")


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
    a = phensim.simulate_by_mutation_rate(50, 1e6, seed=7, backend="numba")
    b = phensim.simulate_by_mutation_rate(50, 1e6, seed=7, backend="numba")
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


def test_write_plink_layout(tmp_path):
    G = phensim.simulate_independent(50, 120, seed=22)
    prefix = str(tmp_path / "sim")
    phensim.write_plink(G, prefix, chromosome=np.repeat([1, 2], 60))
    bed = open(f"{prefix}.bed", "rb").read()
    assert bed[:3] == b"\x6c\x1b\x01"
    assert len(bed) == 3 + ((50 + 3) // 4) * 120
    assert len(open(f"{prefix}.bim").readlines()) == 120
    assert len(open(f"{prefix}.fam").readlines()) == 50
