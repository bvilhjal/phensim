"""Regression tests for the matrix-free genetic background, prepared
LD blocks and chunked reference-panel accumulation."""

import weakref

import numpy as np
import pytest

import phensim
import phensim.phenotypes as phen
import phensim.sumstats as ss
from phensim.kinship import grm
from phensim.phenotypes import _background_factor, _draw_background


# --------------------------------------------------------------------- #
# The matrix-free GRM factor: F F' is exactly the scaled kinship
# --------------------------------------------------------------------- #
def _factor_matrices():
    rng = np.random.default_rng(731)
    complete = rng.binomial(2, 0.31, (9, 7)).astype(float)
    structured = complete.copy()
    structured[:4, :3] = 0
    structured[4:, :3] = 2
    missing = structured.copy()
    missing[::3, ::2] = np.nan
    missing[1::3, 1::2] = -1
    monomorphic = np.column_stack([complete, np.zeros(9), np.ones(9)])
    return {"complete": complete, "structured": structured,
            "missing": missing, "monomorphic_columns": monomorphic}


@pytest.mark.parametrize(
    "name", ["complete", "structured", "missing", "monomorphic_columns"])
def test_background_factor_reconstructs_scaled_grm(name):
    G = _factor_matrices()[name]
    Z, marker_scale, common_scale = _background_factor(G)
    F = np.column_stack([marker_scale * Z,
                         common_scale * np.ones(G.shape[0])])
    np.testing.assert_allclose(F @ F.T, grm(G), rtol=1e-12, atol=1e-12)


class _BasisRNG:
    """Serve one (m + 1)-vector of innovations; record the call sizes."""

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)
        self.calls = []

    def standard_normal(self, size=None):
        self.calls.append(size)
        if size is None:
            return float(self.vector[-1])
        assert size == self.vector.size - 1
        return self.vector[:-1].copy()


def test_draw_background_common_innovation_is_constant():
    G = _factor_matrices()["complete"]
    factor = _background_factor(G)

    class CommonOnly:
        def standard_normal(self, size=None):
            return np.zeros(G.shape[1]) if size is not None else 2.0

    u = _draw_background(factor, 4.0, CommonOnly())
    assert u[0] != 0 and np.all(u == u[0])  # the J/n component survives
    np.testing.assert_allclose(u[0], 2.0 * 2.0 / np.sqrt(G.shape[0]))


def test_draw_background_basis_covariance_and_call_sizes():
    G = _factor_matrices()["missing"]
    Z, marker_scale, common_scale = factor = _background_factor(G)
    F = np.column_stack([marker_scale * Z,
                         common_scale * np.ones(G.shape[0])])
    columns = []
    for vector in np.eye(G.shape[1] + 1):
        rng = _BasisRNG(vector)
        columns.append(_draw_background(factor, 0.37, rng))
        assert rng.calls == [G.shape[1], None]  # marker draw, then scalar
    drawn = np.column_stack(columns)
    np.testing.assert_allclose(
        drawn @ drawn.T, 0.37 * (F @ F.T), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------- #
# simulate_trait / simulate_correlated_traits: no n x n work by default
# --------------------------------------------------------------------- #
def test_default_trait_paths_do_not_eigendecompose(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("n x n eigendecomposition should not run")

    monkeypatch.setattr(np.linalg, "eigh", boom)
    monkeypatch.setattr(phen, "_grm", boom)
    G = phensim.simulate_independent(60, 300, seed=1)
    tr = phensim.simulate_trait(G, h2=0.6, n_causal=5, seed=2)
    assert tr["y"].shape == (60,)
    tr2 = phensim.simulate_correlated_traits(G, n_causal=3, seed=2)
    assert tr2["y_a"].shape == (60,)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supplied_k_draw_matches_manual_eigh_oracle(dtype):
    G = phensim.simulate_independent(60, 300, seed=3).astype(float)
    K = grm(G).astype(dtype)
    K = (K + K.T) / 2  # exactly symmetric, kept in the caller's dtype
    tr = phensim.simulate_trait(
        G, h2=0.6, n_causal=5, architecture="mixed", K=K, seed=9)
    rng = np.random.default_rng(9)
    lam, U = np.linalg.eigh(K)
    lam = np.maximum(lam, 0.0)
    u = U @ (np.sqrt(lam * 0.3) * rng.standard_normal(60))
    np.testing.assert_array_equal(tr["u"], u)


def test_supplied_k_must_be_finite_symmetric_square():
    G = phensim.simulate_independent(20, 50, seed=4)
    with pytest.raises(ValueError, match="kinship"):
        phensim.simulate_trait(G, K=np.ones((20, 19)))
    asymmetric = grm(G)
    asymmetric[0, 1] += 0.5
    with pytest.raises(ValueError, match="kinship"):
        phensim.simulate_trait(G, K=asymmetric)
    with pytest.raises(ValueError, match="kinship"):
        phensim.simulate_trait(G, K=np.full((20, 20), np.nan))
    # a background-free architecture ignores K entirely
    tr = phensim.simulate_trait(G, h2=0.5, n_causal=3, architecture="qtl",
                                K=np.ones((20, 19)), seed=1)
    assert tr["y"].shape == (20,)


def test_confounded_single_eigendecomposition(monkeypatch):
    G = phensim.simulate_independent(40, 200, seed=5)
    K = grm(G)
    real_eigh = np.linalg.eigh
    calls = 0

    def spy(*a, **k):
        nonlocal calls
        calls += 1
        return real_eigh(*a, **k)

    monkeypatch.setattr(np.linalg, "eigh", spy)
    tr = phensim.simulate_confounded_trait(G, seed=16)
    assert calls == 1
    tr2 = phensim.simulate_confounded_trait(G, seed=16, K=K)
    assert calls == 2

    # The cached eigenpair reproduces the original two-eigh seeded draw.
    rng = np.random.default_rng(16)
    sub = int(rng.integers(1, 2**31 - 1))
    inner = np.random.default_rng(sub)
    lam, U = real_eigh(K)
    u = U @ (np.sqrt(np.maximum(lam, 0) * 0.25) * inner.standard_normal(40))
    np.testing.assert_array_equal(tr["u"], np.sqrt(0.4) * u)
    np.testing.assert_array_equal(tr2["u"], tr["u"])
    np.testing.assert_allclose(
        tr["liability"], tr["structure"] + tr["u"] + tr["q"] + tr["e"],
        atol=1e-14)


def test_confounded_single_eigendecomposition_float32_k(monkeypatch):
    G = phensim.simulate_independent(40, 200, seed=5)
    K = grm(G).astype(np.float32)
    K = (K + K.T) / 2
    real_eigh = np.linalg.eigh
    calls = 0

    def spy(*a, **k):
        nonlocal calls
        calls += 1
        return real_eigh(*a, **k)

    monkeypatch.setattr(np.linalg, "eigh", spy)
    tr = phensim.simulate_confounded_trait(G, seed=16, K=K)
    assert calls == 1

    rng = np.random.default_rng(16)
    sub = int(rng.integers(1, 2**31 - 1))
    inner = np.random.default_rng(sub)
    lam, U = real_eigh(K)  # the supplied float32 eigendecomposition
    u = U @ (np.sqrt(np.maximum(lam, 0) * 0.25) * inner.standard_normal(40))
    np.testing.assert_array_equal(tr["u"], np.sqrt(0.4) * u)


def test_monomorphic_default_background_fails_but_h2_zero_works():
    G = np.ones((50, 40))
    with pytest.raises(ValueError, match="polymorphic"):
        phensim.simulate_trait(G, h2=0.5, seed=1)
    tr = phensim.simulate_trait(G, h2=0, n_causal=0, seed=1)
    np.testing.assert_array_equal(tr["u"], 0)
    np.testing.assert_array_equal(tr["liability"], tr["e"])


def test_positive_qtl_requires_a_causal_variant():
    G = phensim.simulate_independent(30, 60, seed=6)
    for kwargs in (dict(architecture="qtl", h2=0.5, n_causal=0),
                   dict(architecture="mixed", h2=0.5, n_causal=0),
                   dict(architecture="qtl", h2=0.5,
                        causal=np.array([], dtype=int))):
        with pytest.raises(ValueError, match="causal variant"):
            phensim.simulate_trait(G, **kwargs)
    # zero causal variants stay valid without a QTL component
    for kwargs in (dict(architecture="infinitesimal", h2=0.5, n_causal=0),
                   dict(architecture="qtl", h2=0, n_causal=0),
                   dict(architecture="mixed", h2=0, n_causal=0)):
        tr = phensim.simulate_trait(G, **kwargs)
        assert tr["causal"].size == 0
    with pytest.raises(ValueError, match="causal variant"):
        phensim.simulate_correlated_traits(G, n_causal=0)


# --------------------------------------------------------------------- #
# prepare_blocks: a validated, factored snapshot accepted by consumers
# --------------------------------------------------------------------- #
def _ld_blocks():
    """PD + singular PSD blocks over an unsorted index partition (m = 7)."""
    return [
        (np.array([[1.0, 0.4, 0.1], [0.4, 1.0, 0.2], [0.1, 0.2, 1.0]]),
         np.array([3, 0, 5])),
        (np.ones((2, 2)), np.array([1, 2])),
        (np.eye(2), np.array([4, 6])),
    ]


def _assert_same_blocks(raw_out, prepared_out):
    if isinstance(raw_out, list):  # shake_ld's (R, ix) list
        for (R1, i1), (R2, i2) in zip(raw_out, prepared_out):
            np.testing.assert_array_equal(R1, R2)
            np.testing.assert_array_equal(i1, i2)
    elif isinstance(raw_out, tuple):  # the pair simulator
        for a, b in zip(raw_out, prepared_out):
            np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_array_equal(raw_out, prepared_out)


def test_prepare_blocks_raw_and_prepared_outputs_identical():
    blocks = _ld_blocks()
    prepared = ss.prepare_blocks(blocks)
    assert prepared is ss.prepare_blocks(prepared)  # fast path
    m = prepared.m
    beta = np.linspace(-0.5, 0.5, m)
    maf = np.full(m, 0.3)
    n_vec = np.linspace(100, 500, m)
    cases = [
        (ss.simulate_effects, (blocks,),
         dict(h2=0.4, n_causal=3, seed=7)),
        (ss.simulate_effects, (blocks,),
         dict(h2=0.4, n_causal=3, architecture="equal", seed=7)),
        (ss.simulate_effects, (blocks,),
         dict(h2=0.4, architecture="polygenic", seed=7)),
        (ss.simulate_effects, (blocks,),
         dict(h2=0.4, architecture="maf", maf=maf, seed=7)),
        (ss.simulate_sumstats, (beta, blocks, 400), dict(seed=7)),
        (ss.simulate_sumstats, (beta, blocks, n_vec), dict(seed=7)),
        (ss.simulate_sumstats_pair, (beta, beta[::-1].copy(), blocks, 300),
         dict(noise_correlation=-0.3, seed=7)),
        (ss.simulate_sumstats_pair, (beta, beta, blocks, 300),
         dict(noise_correlation=0.4, seed=7)),
        (ss.simulate_sumstats_pair, (beta, beta, blocks, 300),
         dict(noise_correlation=1.0, seed=7)),
        (ss.shake_ld, (blocks, None), {}),
        (ss.shake_ld, (blocks, 40), dict(seed=7)),
    ]
    for fn, args, kw in cases:
        raw_out = fn(*args, **kw)
        prepared_out = fn(*[prepared if a is blocks else a for a in args], **kw)
        _assert_same_blocks(raw_out, prepared_out)


def test_prepare_blocks_snapshots_and_freezes():
    R = np.array([[1.0, 0.3], [0.3, 1.0]])
    ix = np.array([2, 0])
    prepared = ss.prepare_blocks([(R, ix), (np.eye(1), np.array([1]))])
    expected = ss.shake_ld(prepared, 30, seed=3)
    R[0, 0] = 9.0
    ix[0] = 1  # later edits to the caller's arrays cannot reach the copy
    assert prepared.entries[0][0][0, 0] == 1.0
    np.testing.assert_array_equal(prepared.entries[0][1], [2, 0])
    for Rp, ixp, factor in prepared.entries:
        for arr in (Rp, ixp, factor):
            assert not arr.flags.writeable
    with pytest.raises(AttributeError):
        prepared.m = 0
    _assert_same_blocks(expected, ss.shake_ld(prepared, 30, seed=3))
    writable = ss.shake_ld(prepared, None)
    assert writable[0][1].flags.writeable


def test_raw_consumers_do_not_prepare_and_factor_per_block(monkeypatch):
    blocks = _ld_blocks()

    def boom(*a, **k):
        raise AssertionError("raw calls must not go through prepare_blocks")

    monkeypatch.setattr(ss, "prepare_blocks", boom)
    real_chol = ss._chol
    calls = []

    def spy(R):
        calls.append(1)
        return real_chol(R)

    monkeypatch.setattr(ss, "_chol", spy)
    beta = np.zeros(7)
    for consume in (
            lambda: ss.simulate_effects(blocks, architecture="polygenic",
                                        seed=1),
            lambda: ss.simulate_sumstats(beta, blocks, 100, seed=1),
            lambda: ss.simulate_sumstats_pair(beta, beta, blocks, 100,
                                              seed=1),
            lambda: ss.shake_ld(blocks, 10, seed=1)):
        calls.clear()
        consume()
        assert len(calls) == len(blocks)  # one factorization per block


def test_sumstats_does_not_retain_raw_factors(monkeypatch):
    blocks = [(np.eye(2), np.array([0, 1])), (np.eye(2), np.array([2, 3])),
              (np.eye(2), np.array([4, 5]))]
    real_chol = ss._chol
    refs, live_at_call = [], []

    def spy(R):
        live_at_call.append(sum(r() is not None for r in refs))
        factor = real_chol(R)
        refs.append(weakref.ref(factor))
        return factor

    monkeypatch.setattr(ss, "_chol", spy)
    ss.simulate_sumstats(np.zeros(6), blocks, 100, seed=1)
    assert all(count <= 1 for count in live_at_call)  # factors not pooled
    assert all(r() is None for r in refs)  # none retained after return


def test_prepared_blocks_skip_revalidation(monkeypatch):
    prepared = ss.prepare_blocks(_ld_blocks())

    def boom(*a, **k):
        raise AssertionError("prepared input must not be re-validated")

    monkeypatch.setattr(ss, "_as_blocks", boom)
    monkeypatch.setattr(ss, "_chol", boom)
    beta = np.zeros(prepared.m)
    ss.simulate_effects(prepared, architecture="polygenic", seed=1)
    ss.simulate_sumstats(beta, prepared, 100, seed=1)
    ss.simulate_sumstats_pair(beta, beta, prepared, 100, seed=1)
    ss.shake_ld(prepared, 10, seed=1)


@pytest.mark.parametrize("blocks", [
    [], [(np.eye(2), [0, 2])], [(np.eye(2), [0, 0])],
    [(np.eye(1), [0]), (np.eye(1), [0])], [(np.eye(1), [-1])],
    [(np.eye(1), [10**12])], [(np.eye(1), [0.0])], [(np.eye(1), [True])],
    [(np.eye(2), [0])],
    # materially indefinite, and a non-unit diagonal
    [(np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]]),
      np.arange(3))],
    [(np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]]),
      np.arange(3))],
])
def test_prepare_blocks_rejects_invalid(blocks):
    with pytest.raises(ValueError):
        ss.prepare_blocks(blocks)


# --------------------------------------------------------------------- #
# shake_ld(chunk_size): chunked Wishart accumulation
# --------------------------------------------------------------------- #
def test_shake_ld_chunked_matches_full_panel():
    blocks = _ld_blocks()
    full = phensim.shake_ld(blocks, 73, seed=819)
    for chunk_size in (1, 7, 30):
        chunked = phensim.shake_ld(blocks, 73, seed=819, chunk_size=chunk_size)
        for (Rc, ic), (Rf, if_) in zip(chunked, full):
            np.testing.assert_array_equal(ic, if_)
            np.testing.assert_allclose(Rc, Rf, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(Rc, Rc.T, atol=1e-14)
            np.testing.assert_array_equal(np.diag(Rc), 1.0)
            assert np.linalg.eigvalsh(Rc)[0] >= -1e-12


def test_shake_ld_chunked_preserves_the_rng_stream():
    blocks = _ld_blocks()
    rng_chunked = np.random.default_rng(11)
    phensim.shake_ld(blocks, 73, seed=rng_chunked, chunk_size=7)
    rng_full = np.random.default_rng(11)
    phensim.shake_ld(blocks, 73, seed=rng_full)
    assert rng_chunked.bit_generator.state == rng_full.bit_generator.state


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, np.nan])
def test_shake_ld_chunk_size_validation(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        phensim.shake_ld(_ld_blocks(), 20, chunk_size=chunk_size)


def test_shake_ld_full_panel_paths_are_bit_identical():
    blocks = _ld_blocks()
    base = phensim.shake_ld(blocks, 73, seed=5)
    for kw in (dict(), dict(chunk_size=73), dict(chunk_size=100)):
        _assert_same_blocks(base, phensim.shake_ld(blocks, 73, seed=5, **kw))


def test_shake_ld_chunk_bounds_draw_sizes(monkeypatch):
    real_rng = np.random.default_rng(0)
    calls = []

    class Wrapped:
        def standard_normal(self, size=None):
            calls.append(size)
            return real_rng.standard_normal(size)

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: Wrapped())
    phensim.shake_ld(_ld_blocks(), 73, seed=1, chunk_size=7)
    assert calls
    for size in calls:
        assert size[0] <= 7
    assert sum(size[0] for size in calls) == 73 * 3  # three blocks


def test_shake_ld_none_nref_ignores_chunk_after_validation():
    blocks = _ld_blocks()
    _assert_same_blocks(
        phensim.shake_ld(blocks, None),
        phensim.shake_ld(blocks, None, chunk_size=3))
    with pytest.raises(ValueError, match="chunk_size"):
        phensim.shake_ld(blocks, None, chunk_size=0)
