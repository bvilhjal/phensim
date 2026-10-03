"""Known-model, boundary and memory-mode checks for reference copying."""
import numpy as np
import pytest

import phensim
from phensim.hapnest import _fill_numpy


def fixture():
    rng = np.random.default_rng(94)
    H = rng.binomial(1, .4, (20, 2, 61)).astype(np.int8)
    cm = np.r_[np.arange(31)*.02, np.arange(30)*.02]
    ages = np.linspace(0, 300, 61)
    kw = dict(chromosome=np.r_[np.zeros(31), np.ones(30)],
              reference_populations=np.repeat([0, 1], 10),
              sample_populations=np.arange(17) % 2,
              ne=[150, 300], rho=[.3, .8], seed=83)
    return H, cm, ages, kw


def test_backends_and_batches_have_identical_draws(tmp_path):
    H, cm, ages, kw = fixture()
    expected = phensim.simulate_hapnest(H, 17, cm, ages, backend="numpy", **kw)
    for backend in (["numpy", "numba"] if phensim.HAVE_NUMBA else ["numpy"]):
        for batch in [1, 7, 50]:
            got = phensim.simulate_hapnest(H, 17, cm, ages, backend=backend, batch_size=batch, **kw)
            np.testing.assert_array_equal(got, expected)
        out = np.memmap(tmp_path / f"{backend}.dat", mode="w+", dtype="int8", shape=(17, 61))
        assert phensim.simulate_hapnest(H, 17, cm, ages, out=out, backend=backend, **kw) is out
        np.testing.assert_array_equal(out, expected)
    chunks = list(phensim.iter_hapnest(H, 17, cm, ages, batch_size=3, **kw))
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    assert not np.shares_memory(chunks[0], chunks[1])


def test_known_population_phase_and_mutation_limits():
    H = np.zeros((4, 2, 10), dtype=np.int8)
    H[2:] = 1
    kw = dict(reference_populations=[0, 0, 1, 1], sample_populations=[0, 1, 0, 1])
    got = phensim.simulate_hapnest(H, 4, np.zeros(10), np.full(10, np.inf), **kw)
    np.testing.assert_array_equal(got[:, 0], [0, 2, 0, 2])
    assert not phensim.simulate_hapnest(H, 4, np.zeros(10), np.zeros(10), **kw).any()
    H[:, 0] = 0
    H[:, 1] = 1
    assert np.all(phensim.simulate_hapnest(H, 4, np.zeros(10), np.full(10, np.inf)) == 1)


def test_uint64_reference_matches_int8_across_backends_and_output_modes(tmp_path):
    H, cm, ages, kw = fixture()
    expected = phensim.simulate_hapnest(H, 17, cm, ages, backend="numpy", **kw)
    path = tmp_path / "reference.npy"
    np.save(path, H.astype(np.uint64))
    reference = np.load(path, mmap_mode="r")
    for backend in (["numpy", "numba"] if phensim.HAVE_NUMBA else ["numpy"]):
        known = phensim.simulate_hapnest(np.ones((2, 2, 3), dtype=np.uint64),
            2, [0, 1, 2], [np.inf]*3, backend=backend)
        np.testing.assert_array_equal(known, np.full((2, 3), 2, dtype=np.int8))
        batches = list(phensim.iter_hapnest(reference, 17, cm, ages,
            backend=backend, batch_size=3, **kw))
        np.testing.assert_array_equal(np.concatenate(batches), expected)
        out = np.memmap(tmp_path / f"{backend}.dat", mode="w+", dtype="int8", shape=(17, 61))
        assert phensim.simulate_hapnest(reference, 17, cm, ages, out=out,
            backend=backend, batch_size=7, **kw) is out
        np.testing.assert_array_equal(out, expected)


def test_non_native_reference_falls_back_without_a_numba_typing_error(tmp_path):
    H, cm, ages, kw = fixture()
    expected = phensim.simulate_hapnest(H, 17, cm, ages, backend="numpy", **kw)
    path = tmp_path / "reference.npy"
    np.save(path, H.astype(np.dtype(np.int16).newbyteorder("S")))
    reference = np.load(path, mmap_mode="r")
    assert not reference.dtype.isnative
    for backend in ["auto", "numpy"]:
        np.testing.assert_array_equal(phensim.simulate_hapnest(
            reference, 17, cm, ages, backend=backend, batch_size=3, **kw), expected)
    out = np.full((17, 61), -1, dtype=np.int8)
    with pytest.raises(ValueError, match="native byte order"):
        phensim.simulate_hapnest(reference, 17, cm, ages, backend="numba", out=out, **kw)
    assert np.all(out == -1)


def test_gamma_age_filter_against_closed_form():
    # One site, all donor alleles 1: each phase survives with Gamma(2, Ne/N) CDF.
    H = np.ones((10, 2, 1), dtype=np.int8)
    G = phensim.simulate_hapnest(H, 10000, [0], [30], ne=200, rho=0, seed=901)
    probability = 1 - np.exp(-1.5)*(1+1.5)
    assert abs(G.mean()/2 - probability) < .015


def test_no_recombination_matches_reference_frequency_and_covariance():
    H = np.random.default_rng(173).binomial(1, .4, (16, 2, 6)).astype(np.int8)
    G = phensim.simulate_hapnest(H, 12000, np.arange(6), np.full(6, np.inf), rho=0, seed=991)
    # Independent uniform donor draws for each phase: diploid covariance is
    # the sum of the two empirical haplotype covariance matrices.
    mean = H[:, 0].mean(0)+H[:, 1].mean(0)
    covariance = np.cov(H[:, 0], rowvar=False, ddof=0)+np.cov(H[:, 1], rowvar=False, ddof=0)
    np.testing.assert_allclose(G.mean(0), mean, atol=.025)
    np.testing.assert_allclose(np.cov(G, rowvar=False, ddof=0), covariance, atol=.02)


def test_readonly_mapped_reference_and_numpy_only_fallback(tmp_path, monkeypatch):
    import phensim.hapnest as sampler
    H, cm, ages, kw = fixture()
    path = tmp_path/"reference.npy"
    np.save(path, H)
    mapped = np.load(path, mmap_mode="r")
    expected = phensim.simulate_hapnest(H, 17, cm, ages, backend="numpy", **kw)
    np.testing.assert_array_equal(phensim.simulate_hapnest(mapped, 17, cm, ages, **kw), expected)
    monkeypatch.setattr(sampler, "HAVE_NUMBA", False)
    np.testing.assert_array_equal(phensim.simulate_hapnest(mapped, 17, cm, ages, **kw), expected)
    with pytest.raises(ImportError, match="fast"):
        phensim.simulate_hapnest(mapped, 17, cm, ages, backend="numba", **kw)


def test_inclusive_breakpoint_and_chromosome_reset():
    class Fixed:
        def __init__(self):
            self.donor = -1
        def gamma(self, *args):
            return 1
        def exponential(self, *args):
            return .5
        def integers(self, *args):
            self.donor += 1
            return self.donor % 2
    H = np.array([np.zeros((2, 6)), np.ones((2, 6))], dtype=np.int8)
    out = np.empty((1, 6), dtype=np.int8)
    _fill_numpy(out, H, np.array([0, 1, 2, 0, 1, 2]), np.full(6, np.inf),
                np.array([0, 3, 6]), np.arange(2), np.array([0, 2]),
                np.array([0]), np.array([100]), np.array([1]), Fixed())
    np.testing.assert_array_equal(out[0], [0, 0, 2, 0, 0, 2])


@pytest.mark.parametrize("field,value", [("ne", 0), ("rho", -1), ("batch_size", True),
    ("backend", "fastish"), ("sample_populations", np.full(17, 3)),
    ("reference_populations", np.full(20, 2))])
def test_bad_parameters(field, value):
    H, cm, ages, kw = fixture()
    kw[field] = value
    with pytest.raises((ValueError, TypeError)):
        phensim.simulate_hapnest(H, 17, cm, ages, **kw)


@pytest.mark.parametrize("kind", ["map", "age", "haplotypes", "chromosome"])
def test_bad_reference_inputs_leave_output_untouched(kind):
    H, cm, ages, kw = fixture()
    if kind == "map":
        cm[3] = -1
    elif kind == "age":
        ages[5] = np.nan
    elif kind == "haplotypes":
        H[-1, 1, -1] = 2
    else:
        kw["chromosome"][-1] = 0
    out = np.full((17, 61), -1, dtype=np.int8)
    with pytest.raises(ValueError):
        phensim.simulate_hapnest(H, 17, cm, ages, out=out, **kw)
    assert np.all(out == -1)


def test_phased_reference_sums_to_existing_draw():
    kw = dict(n=80, m=60, block_sizes=[20]*3, fst=.08, seed=3)
    G, pop = phensim.simulate_population_structure(**kw)
    H, phase_pop = phensim.simulate_population_structure(**kw, phased=True)
    np.testing.assert_array_equal(H.sum(axis=1), G)
    np.testing.assert_array_equal(pop, phase_pop)


def test_blocked_trait_preserves_dense_factor_innovations():
    G = phensim.simulate_independent(137, 93, seed=81)
    kw = dict(h2=.6, n_causal=8, seed=86)
    dense = phensim.simulate_trait(G, **kw)
    for block in [1, 7, 150]:
        tiled = phensim.simulate_trait(G, genotype_block_size=block, **kw)
        for key in dense:
            np.testing.assert_allclose(tiled[key], dense[key], rtol=2e-13, atol=2e-13)
    with pytest.raises(ValueError, match="genotype_block_size"):
        phensim.simulate_trait(G, genotype_block_size=0)


def test_tiled_bed_payload_matches_full_encoding(tmp_path):
    from phensim.io import _encode_bed, _plink_genotypes
    G = np.random.default_rng(15).integers(-1, 3, (13, 77)).astype(float)
    G[2, 3] = np.nan
    checked, missing = _plink_genotypes(G)
    expected = b"\x6c\x1b\x01" + _encode_bed(missing, checked, *G.shape)
    for block in [1, 11, 200]:
        prefix = tmp_path / f"tile{block}"
        phensim.write_plink(G, prefix, block_size=block)
        assert prefix.with_suffix(".bed").read_bytes() == expected
    G[-1, -1] = .5
    with pytest.raises(ValueError):
        phensim.write_plink(G, tmp_path / "invalid", block_size=1)
    assert not (tmp_path / "invalid.bed").exists()
