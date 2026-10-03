"""Regression tests for the review fixes: coalescent RNG isolation and
breakpoint clamping, genotype/pedigree/ascertainment validation, PLINK
metadata and encoding, kinship guards and LOCO streaming, and the
jitted normal p-value loop."""

import math
import types
import warnings

import numpy as np
import pytest

import phensim
from phensim import _coalescent as coal
from phensim._common import norm_isf
from phensim.kinship import grm, ibs_kinship, iter_loco_kinships, loco_kinships
from phensim.pedigree import (kinship_from_pedigree, mendelian_draw,
                              pedigree_birth_times, simulate_pedigree)
from phensim.sumstats import _normal_pvalues


# --------------------------------------------------------------------- #
# Built-in coalescent: global-RNG isolation for the pure-Python kernels
# --------------------------------------------------------------------- #
def _force_pure_python_kernels(monkeypatch):
    """Run the simulator through the genuine pure-Python kernels."""
    monkeypatch.setattr(coal, "HAVE_NUMBA", False)
    monkeypatch.setattr(
        coal, "_hudson", getattr(coal._hudson, "py_func", coal._hudson))
    monkeypatch.setattr(
        coal, "_draw_mutations",
        getattr(coal._draw_mutations, "py_func", coal._draw_mutations))


def test_pure_python_kernels_leave_global_rng_untouched(monkeypatch):
    _force_pure_python_kernels(monkeypatch)
    outer_state = np.random.get_state()
    np.random.seed(7)
    try:
        expected_next = np.random.random()
        np.random.seed(7)
        args = dict(recomb_rate=1e-8, mut_rate=5e-8, seed=3)
        G1, pos1, af1 = coal.simulate_dosages(6, 20000, **args)
        assert np.random.random() == expected_next  # caller's stream intact
        G2, pos2, af2 = coal.simulate_dosages(6, 20000, **args)
        np.testing.assert_array_equal(G1, G2)  # seeded output unchanged
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(af1, af2)

        # Reference: invoke the kernels directly (the pre-isolation
        # behaviour); they reseed the global RNG internally, so the
        # isolated run must match them bit for bit.
        saved = np.random.get_state()
        try:
            with monkeypatch.context() as mp:
                mp.setattr(coal, "_call_random_kernel",
                           lambda kernel, *a: kernel(*a))
                ref_G, ref_pos, ref_af = coal.simulate_dosages(6, 20000, **args)
        finally:
            np.random.set_state(saved)
        np.testing.assert_array_equal(G1, ref_G)
        np.testing.assert_array_equal(pos1, ref_pos)
        np.testing.assert_array_equal(af1, ref_af)
    finally:
        np.random.set_state(outer_state)


def test_random_kernel_helper_restores_state_on_exception(monkeypatch):
    monkeypatch.setattr(coal, "HAVE_NUMBA", False)
    np.random.seed(9)
    before = np.random.get_state()

    def boom(*args):
        np.random.random()
        raise RuntimeError("kernel failed")

    with pytest.raises(RuntimeError, match="kernel failed"):
        coal._call_random_kernel(boom, 1, 2)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_random_kernel_helper_direct_call_when_numba(monkeypatch):
    monkeypatch.setattr(coal, "HAVE_NUMBA", True)
    sentinel, calls = object(), []

    def kernel(*args):
        calls.append(args)
        return sentinel

    assert coal._call_random_kernel(kernel, 1, 2) is sentinel
    assert calls == [(1, 2)]


@pytest.mark.parametrize("kwargs", [
    dict(n=0), dict(n=2.5), dict(n=True), dict(seq_len=0),
    dict(seq_len=np.inf), dict(recomb_rate=-1e-8), dict(mut_rate=np.nan),
    dict(Ne=0), dict(Ne=np.inf),
])
def test_simulate_dosages_validates_geometry(kwargs):
    base = dict(n=4, seq_len=1e4, recomb_rate=1e-8, mut_rate=1e-8,
                Ne=1000, seed=1)
    base.update(kwargs)
    with pytest.raises(ValueError):
        coal.simulate_dosages(**base)


# --------------------------------------------------------------------- #
# msprime backend: continuous genome
# --------------------------------------------------------------------- #
def test_msprime_backend_uses_continuous_genome(monkeypatch):
    msprime = pytest.importorskip("msprime")
    real_ancestry, real_mutations = msprime.sim_ancestry, msprime.sim_mutations
    seen, captured = {}, {}

    def rec_ancestry(*args, **kwargs):
        seen["ancestry"] = kwargs
        captured["ts"] = real_ancestry(*args, **kwargs)
        return captured["ts"]

    def rec_mutations(ts, *args, **kwargs):
        seen["mutations"] = kwargs
        captured["mts"] = real_mutations(ts, *args, **kwargs)
        return captured["mts"]

    monkeypatch.setattr(msprime, "sim_ancestry", rec_ancestry)
    monkeypatch.setattr(msprime, "sim_mutations", rec_mutations)
    G = phensim.simulate_by_mutation_rate(
        10, 1e6, mut_rate=5e-8, seed=8, backend="msprime")
    assert seen["ancestry"]["discrete_genome"] is False
    assert seen["mutations"]["discrete_genome"] is False
    breakpoints = captured["ts"].breakpoints(as_array=True)
    assert np.any(breakpoints != np.floor(breakpoints))
    positions = np.asarray(captured["mts"].tables.sites.position)
    assert positions.size and np.any(positions != np.floor(positions))
    assert np.all(np.diff(positions) >= 0)  # ascending site order
    assert G.shape[0] == 10 and 0 < G.shape[1] <= positions.size


# --------------------------------------------------------------------- #
# Pedigree: unlisted-parent warning and argument validation
# --------------------------------------------------------------------- #
def _typo_pedigree():
    """``'ax'`` is an unlisted father reference for child ``'c'``."""
    ids = ["a", "b", "c"]
    return ids, [None, None, "ax"], [None, None, "b"]


def test_unlisted_parent_warns_once_with_count():
    ids, father, mother = _typo_pedigree()
    with pytest.warns(UserWarning) as record:
        A = kinship_from_pedigree(ids, father, mother)
    assert len(record) == 1
    message = str(record[0].message)
    assert "1" in message and "unlisted parent" in message
    assert "unknown founders" in message
    np.testing.assert_allclose(A[1, 2], 0.5)  # c is still b's child


def test_unlisted_parent_warns_in_all_parent_consumers():
    ids, father, mother = _typo_pedigree()
    consumers = [
        lambda: kinship_from_pedigree(ids, father, mother),
        lambda: mendelian_draw(ids, father, mother, seed=1),
        lambda: pedigree_birth_times(ids, father, mother),
    ]
    for consume in consumers:
        with pytest.warns(UserWarning, match="unlisted parent"):
            consume()


def test_listed_parents_do_not_warn():
    ids = ["a", "b", "c"]
    father, mother = [None, None, "a"], [None, None, "b"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A = kinship_from_pedigree(ids, father, mother)
        mendelian_draw(ids, father, mother, seed=1)
        pedigree_birth_times(ids, father, mother)
    np.testing.assert_allclose(A[0, 2], 0.5)  # a is c's father


@pytest.mark.parametrize("marker", [None, 0, "0", "", np.nan])
def test_missing_parent_sentinels_do_not_warn(marker):
    ids = ["a", "b"]
    father, mother = [marker, None], [None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kinship_from_pedigree(ids, father, mother)
        mendelian_draw(ids, father, mother, seed=1)
        pedigree_birth_times(ids, father, mother)


def test_unlisted_parents_keep_external_parent_semantics():
    ids = ["f", "c", "c2"]
    with pytest.warns(UserWarning, match="unlisted parent"):
        A = kinship_from_pedigree(ids, [None, "fx", "f"], [None, None, "m"])
    # 'fx' and 'm' are unlisted: c is a founder, c2 has one known parent.
    expected = np.array([[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])
    np.testing.assert_allclose(A, expected)


def test_mendelian_innovations_must_be_finite_vector():
    ids = ["a", "b", "c"]
    with pytest.raises(ValueError, match="innovations"):
        mendelian_draw(ids, [None] * 3, [None] * 3,
                       innovations=np.ones((3, 1)))
    with pytest.raises(ValueError, match="innovations"):
        mendelian_draw(ids, [None] * 3, [None] * 3,
                       innovations=[1.0, np.inf, 0.0])
    with pytest.raises(ValueError, match="innovations"):
        mendelian_draw(ids, [None] * 3, [None] * 3,
                       innovations=np.ones(4))


@pytest.mark.parametrize("kwargs", [
    dict(n_founder_pairs=0), dict(n_founder_pairs=-1),
    dict(n_founder_pairs=2.5), dict(n_founder_pairs=True),
    dict(gens=-1), dict(gens=1.5), dict(gens=True),
    dict(remarry=-0.1), dict(remarry=1.5), dict(remarry=np.nan),
])
def test_simulate_pedigree_validates_geometry(kwargs):
    with pytest.raises(ValueError):
        simulate_pedigree(**kwargs)


def test_simulate_pedigree_zero_gens_is_founders_only():
    ids, father, mother = simulate_pedigree(
        n_founder_pairs=2, gens=0, seed=1)
    assert ids == ["p0", "p1", "p2", "p3"]
    assert father == [None] * 4 and mother == [None] * 4


# --------------------------------------------------------------------- #
# PLINK writer: whole-matrix validation, metadata, vectorized encoding
# --------------------------------------------------------------------- #
def _decode_bed_bytes(bed, n, m):
    """Independent SNP-major BED decoder; also checks zero padding bits."""
    assert bed[:3] == b"\x6c\x1b\x01"
    stride = (n + 3) // 4
    assert len(bed) == 3 + stride * m
    lookup = np.array([0, -1, 1, 2])
    decoded = np.empty((n, m))
    for j in range(m):
        for i in range(n):
            code = (bed[3 + j * stride + i // 4] >> (2 * (i % 4))) & 3
            decoded[i, j] = lookup[code]
        if n % 4:
            assert bed[3 + (j + 1) * stride - 1] >> (2 * (n % 4)) == 0
    return decoded


@pytest.mark.parametrize("n", range(1, 10))
@pytest.mark.parametrize("dtype", ["int8", "float64"])
def test_write_plink_byte_oracle(tmp_path, n, dtype):
    m = 6
    states = np.tile(np.array([0, 1, 2, -1]), n * m // 4 + 1)[:n * m]
    G = states.reshape(n, m).astype(dtype)
    if dtype == "float64":
        G[n - 1, m - 1] = np.nan  # NaN missing alongside the -1 codes
    prefix = str(tmp_path / "oracle")
    phensim.write_plink(G, prefix)
    decoded = _decode_bed_bytes((tmp_path / "oracle.bed").read_bytes(), n, m)
    np.testing.assert_array_equal(
        decoded, np.where(np.isnan(G) | (G < 0), -1, G))


@pytest.mark.parametrize("meta", [
    dict(sample_ids=["a", "b", "c"]),           # wrong count
    dict(sample_ids=["a", "a", "b", "c"]),      # duplicates
    dict(sample_ids=["a ", "b", "c", "d"]),     # whitespace
    dict(sample_ids=["a\tb", "b", "c", "d"]),   # control/whitespace
    dict(sample_ids=["", "b", "c", "d"]),       # empty
    dict(sample_ids=["a", None, "c", "d"]),     # None
    dict(chromosome=np.array([22.5, 1.0, 1.0])),
    dict(chromosome=np.array([-1, 1, 1])),
    dict(chromosome=np.array([np.inf, 1, 1])),
    dict(chromosome=["2 2", "1", "1"]),         # whitespace token
    dict(chromosome=np.array([[1, 1, 1]])),     # not 1-D
    dict(chromosome=np.array([1, 1])),          # wrong length
    dict(position=np.array([-1, 5, 9])),
    dict(position=np.array([1.5, 5, 9])),
    dict(position=np.array([np.inf, 5, 9])),
    dict(position=np.array([1, 5])),            # wrong length
    dict(position=[True, True, False]),         # bool
])
def test_write_plink_rejects_bad_metadata(tmp_path, meta):
    G = np.tile(np.array([0, 1, 2]), (4, 1))  # (4, 3) valid dosages
    # Sentinels at the *requested* prefix: a bad call must not create or
    # truncate any of the three outputs.
    for suffix in (".bed", ".bim", ".fam"):
        (tmp_path / f"bad{suffix}").write_bytes(b"sentinel")
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError):
        phensim.write_plink(G, str(tmp_path / "bad"), **meta)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    for suffix in (".bed", ".bim", ".fam"):
        assert (tmp_path / f"bad{suffix}").read_bytes() == b"sentinel"


def test_write_plink_bad_metadata_leaves_empty_dir(tmp_path):
    G = np.tile(np.array([0, 1, 2]), (4, 1))
    with pytest.raises(ValueError):
        phensim.write_plink(G, str(tmp_path / "bad"),
                            position=np.array([1.5, 5, 9]))
    assert list(tmp_path.iterdir()) == []


def test_write_plink_preserves_large_integer_positions(tmp_path):
    G = np.tile(np.array([0, 1, 2]), (4, 1))
    positions = np.array([2**53 + 1, 2**62 + 7, 9], dtype=np.int64)
    phensim.write_plink(G, str(tmp_path / "big"), position=positions)
    rows = [line.split() for line in (tmp_path / "big.bim").read_text().splitlines()]
    assert [int(row[3]) for row in rows] == [2**53 + 1, 2**62 + 7, 9]


@pytest.mark.parametrize("position", [
    np.array([2**63, 1, 1], dtype=np.uint64),   # does not fit int64
    np.array([2**64 - 1, 1, 1], dtype=np.uint64),
])
def test_write_plink_rejects_overflowing_positions(tmp_path, position):
    G = np.tile(np.array([0, 1, 2]), (4, 1))
    with pytest.raises(ValueError, match="base-pair"):
        phensim.write_plink(G, str(tmp_path / "bad"), position=position)
    assert list(tmp_path.iterdir()) == []


def test_write_plink_accepts_labels_and_integral_floats(tmp_path):
    G = np.tile(np.array([0, 1, 2, 0]), (4, 1))  # (4, 4)
    prefix = str(tmp_path / "labels")
    phensim.write_plink(
        G, prefix,
        chromosome=["X", "MT", 22.0, 0],
        position=np.array([0.0, 5, 9, 12]),
        sample_ids=["s1", "s2", "s3", "s4"])
    bim = [line.split() for line in (tmp_path / "labels.bim").read_text().splitlines()]
    assert [row[0] for row in bim] == ["X", "MT", "22", "0"]
    assert [row[3] for row in bim] == ["0", "5", "9", "12"]
    fam = [line.split() for line in (tmp_path / "labels.fam").read_text().splitlines()]
    assert [row[1] for row in fam] == ["s1", "s2", "s3", "s4"]


# --------------------------------------------------------------------- #
# Kinship guards and LOCO streaming
# --------------------------------------------------------------------- #
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


# --------------------------------------------------------------------- #
# Genotype simulator validation and the opt-in AR(1) scan path
# --------------------------------------------------------------------- #
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
    lambda: phensim.simulate_ar1_blocks(5, [10], maf=0.6),
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


# --------------------------------------------------------------------- #
# Ascertainment: strict trait dicts, binary vectors, integer counts
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("trait", [
    {"case_control": np.array([0, 1, 0, 1, 0])},                    # no liability
    {"liability": np.zeros(5)},                                     # no case_control
    {"case_control": np.array([0, 1, 2, 0, 1]),
     "liability": np.zeros(5)},                                     # value 2
    {"case_control": np.array([0, 1, 0.5, 0, 1]),
     "liability": np.zeros(5)},                                     # fractional
    {"case_control": np.array([0, 1, np.nan, 0, 1]),
     "liability": np.zeros(5)},                                     # NaN
    {"case_control": np.array([0, 1, 0, 0, 1]),
     "liability": np.zeros(6)},                                     # length mismatch
    {"case_control": np.array([0, 1, 0, 0, 1]),
     "liability": np.array([0., 1., np.nan, 0., 1.])},              # NaN liability
    {"case_control": np.ones((5, 1), dtype=int),
     "liability": np.zeros(5)},                                     # 2-D vector
])
def test_ascertain_rejects_bad_trait_dicts(trait):
    with pytest.raises(ValueError):
        phensim.ascertain_case_control(trait, 1, 1)


@pytest.mark.parametrize("n_cases,n_controls", [
    (-1, 1), (1, -1), (1.5, 1), (1, 0.5), (True, 1), (np.nan, 1),
])
def test_ascertain_rejects_bad_counts(n_cases, n_controls):
    trait = {"case_control": np.array([0, 1, 0, 1, 0]),
             "liability": np.arange(5.0)}
    with pytest.raises(ValueError, match="nonnegative integer"):
        phensim.ascertain_case_control(trait, n_cases, n_controls)


def test_ascertain_allows_zero_counts_and_vector_input():
    trait = {"case_control": np.array([0, 1, 0, 1, 0]),
             "liability": np.arange(5.0)}
    out = phensim.ascertain_case_control(trait, 0, 2, seed=1)
    assert out["index"].size == 2 and out["case_control"].sum() == 0
    vec = phensim.ascertain_case_control(np.array([0, 1, 0, 1, 0]), 1, 2, seed=1)
    assert vec["index"].size == 3 and "liability" not in vec


# --------------------------------------------------------------------- #
# gwas_scan: the p-value loop, jitted or not, is exact erfc
# --------------------------------------------------------------------- #
def test_normal_pvalues_match_erfc_exactly():
    helpers = {_normal_pvalues,
               getattr(_normal_pvalues, "py_func", _normal_pvalues)}
    edge = np.array([0.0, -0.0, 1.5, -3.2, np.inf, -np.inf, np.nan,
                     1e-310, -5e-324, 37.0, -100.0])
    rng = np.random.default_rng(0)
    z = np.concatenate([edge, rng.normal(0, 30, 5000),
                        rng.uniform(-1e-3, 1e-3, 100)])
    reference = np.array([math.erfc(abs(v) / math.sqrt(2.0)) for v in z])
    for helper in helpers:
        np.testing.assert_array_equal(helper(z), reference)
