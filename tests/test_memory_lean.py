"""Peak memory of the dosage and trait paths, and the bit-identity it rests on.

The msprime backend built tskit's int32 genotype matrix over every site,
and the trait path held about four float64 copies of G: one data draw at
n = 4,000, m = 50,000 peaked at an 8.5 GB footprint. The oracles are the
earlier full-matrix computations.
"""

import tracemalloc

import numpy as np
import pytest

import phensim
from phensim.genotypes import _coalescent_dosages
from phensim.kinship import _called_standardized
from phensim.phenotypes import _trait_genotypes


def _called_standardized_full(G):
    """The earlier full-matrix pass."""
    Gd = np.asarray(G, dtype=np.float64)
    miss = (Gd < 0) | np.isnan(Gd)
    ok = ~miss
    gf = np.where(miss, 0.0, Gd)
    cnt = ok.sum(axis=0)
    mean = np.where(cnt > 0, gf.sum(axis=0) / np.maximum(cnt, 1), 0.0)
    cen = np.where(ok, Gd - mean, 0.0)
    var = (cen * cen).sum(axis=0) / np.maximum(cnt, 1)
    std = np.sqrt(var)
    return cen / np.where(std > 0, std, 1.0), cnt


def _peak_bytes(fn):
    tracemalloc.start()
    try:
        out = fn()
        return out, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("missing", [None, np.nan, -1])
def test_called_standardized_matches_full_matrix_pass(layout, missing):
    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, size=(120, 2 * 2048 + 37)).astype(np.float64)
    G[:, 5] = 1.0  # monomorphic
    if missing is not None:
        G[rng.random(G.shape) < 0.05] = missing
        G[:, 7] = missing  # no calls
    G = np.asarray(G, order=layout)
    Z, cnt = _called_standardized(G)
    Z_ref, cnt_ref = _called_standardized_full(G)
    np.testing.assert_array_equal(Z, Z_ref)
    np.testing.assert_array_equal(cnt, cnt_ref)
    Z8, _ = _called_standardized(np.where(np.isnan(G), -1, G).astype(np.int8))
    np.testing.assert_array_equal(Z8, _called_standardized_full(np.where(np.isnan(G), -1, G))[0])


def test_called_standardized_overwrites_only_when_asked():
    G = np.random.default_rng(1).integers(0, 3, size=(30, 50)).astype(np.float64)
    G0 = G.copy()
    Z, _ = _called_standardized(G)
    np.testing.assert_array_equal(G, G0)
    assert _called_standardized(G, overwrite=True)[0] is G
    np.testing.assert_array_equal(G, Z)


def test_msprime_dosages_match_the_genotype_matrix_in_int8():
    msprime = pytest.importorskip("msprime")  # imported before measuring: its import allocates
    n, seq_len, seed = 500, 1e6, 11
    (dos, af), peak = _peak_bytes(lambda: _coalescent_dosages(
        n, seq_len, recomb_rate=1e-8, mut_rate=1e-8, Ne=10_000, seed=seed, backend="msprime"))
    ts = msprime.sim_ancestry(samples=n, ploidy=2, population_size=10_000,
                              recombination_rate=1e-8, sequence_length=int(seq_len),
                              discrete_genome=False, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, discrete_genome=False,
                               model=msprime.BinaryMutationModel())
    H = ts.genotype_matrix()
    ref = (H[:, 0::2] + H[:, 1::2]).T
    assert dos.dtype == np.int8
    np.testing.assert_array_equal(dos, ref)
    np.testing.assert_array_equal(af, ref.mean(axis=0) / 2.0)
    assert peak < 2 * dos.nbytes  # the int32 matrix alone was 8x


def test_trait_draw_holds_one_float64_copy_of_int8_genotypes():
    G = np.random.default_rng(2).integers(0, 3, size=(400, 20_000), dtype=np.int8)
    assert _trait_genotypes(G) is G
    _, peak = _peak_bytes(lambda: phensim.simulate_trait(G, h2=0.5, n_causal=10, seed=3))
    assert peak < 2 * G.size * 8  # about 4.4 float64 copies before
