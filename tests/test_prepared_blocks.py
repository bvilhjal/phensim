"""Prepared LD blocks and chunked reference-panel accumulation."""

import weakref

import numpy as np
import pytest

import phensim
import phensim.sumstats as ss


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
    # One case per consumer: architecture, rho and sample-size options have
    # their own tests and do not change the raw/prepared block contract.
    cases = [
        (ss.simulate_effects, (blocks,),
         dict(h2=0.4, n_causal=3, seed=7)),
        (ss.simulate_sumstats, (beta, blocks, 400), dict(seed=7)),
        (ss.simulate_sumstats_pair, (beta, beta[::-1].copy(), blocks, 300),
         dict(noise_correlation=-0.3, seed=7)),
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


# Invalid layouts and correlations are rejected by prepare_blocks alongside
# every raw consumer: test_sumstats.py::_consumers.


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
            np.testing.assert_allclose(np.diag(Rc), 1.0, rtol=0, atol=1e-14)
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
