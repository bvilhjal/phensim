"""Coalescent RNG isolation and backend validation."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import phensim
from phensim import _coalescent as coal


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


@pytest.mark.parametrize("disable_jit", ["0", "1"])
def test_coalescent_rng_isolation_in_actual_numba_modes(tmp_path, disable_jit):
    pytest.importorskip("numba")
    env = dict(os.environ, NUMBA_DISABLE_JIT=disable_jit)
    env.setdefault("NUMBA_CACHE_DIR", str(tmp_path / "numba"))
    result = subprocess.run(
        [sys.executable, "-c", """
import numpy as np
import phensim

assert phensim.HAVE_NUMBA
np.random.seed(42)
expected = np.random.random(5)
np.random.seed(42)
kwargs = dict(recomb_rate=0, seed=17, backend="numba")
first = phensim.simulate_by_mutation_rate(3, 1000, **kwargs)
np.testing.assert_array_equal(np.random.random(5), expected)
second = phensim.simulate_by_mutation_rate(3, 1000, **kwargs)
np.testing.assert_array_equal(first, second)
assert first.shape[0] == 3 and first.shape[1] > 0
"""], cwd=Path(phensim.__file__).resolve().parent.parent,
        env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


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

    kernel.py_func = kernel  # dispatcher marker; no global RNG state access
    monkeypatch.setattr(np.random, "get_state", lambda: pytest.fail("compiled call saved global RNG state"))
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
