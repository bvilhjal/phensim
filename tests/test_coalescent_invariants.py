"""Linked-segment invariants and a bounded independent coalescent comparison."""

import inspect
import sys

import numpy as np
import pytest

from phensim import _coalescent as coal


def _hudson_buffers():
    """Generously sized work buffers for a direct ``_hudson`` call."""
    S, E, N, C = 8192, 32768, 8192, 2048
    return [np.empty(S), np.empty(S), np.empty(S, dtype=np.int64), np.empty(S, dtype=np.int64),
            np.empty(S, dtype=np.int64), np.empty(S, dtype=np.int64),
            np.empty(C, dtype=np.int64), np.empty(C, dtype=np.int64), np.zeros(C), np.zeros(C+1),
            np.empty(E), np.empty(E), np.empty(E, dtype=np.int64), np.empty(E, dtype=np.int64), np.empty(N)]


def _lineage_invariant_observer(fn):
    """``sys.settrace`` observer asserting the linked-segment and
    recombination-weight invariants at every event boundary of ``fn``.
    Returns ``(trace, state)`` with counters in ``state``."""
    lines, first = inspect.getsourcelines(fn)
    event_line = first + next(i for i, line in enumerate(lines) if line.strip() == "while num_lineages > 1:")
    state = {"observations": 0, "previous_count": 4, "recombinations": 0}

    def trace(frame, event, arg):
        if frame.f_code is fn.__code__ and event == "line" and frame.f_lineno == event_line:
            s = frame.f_locals
            k = s["num_lineages"]
            state["recombinations"] += k > state["previous_count"]
            state["previous_count"] = k
            spans = []
            for slot in range(k):
                head = node = int(s["slot_head"][slot])
                assert s["seg_prev"][head] == -1
                while s["seg_next"][node] != -1:
                    following = int(s["seg_next"][node])
                    assert s["seg_prev"][following] == node
                    assert s["seg_right"][node] <= s["seg_left"][following]
                    node = following
                assert s["slot_tail"][slot] == node
                span = s["seg_right"][node] - s["seg_left"][head]
                np.testing.assert_allclose(s["slot_link"][slot], span, atol=1e-8)
                spans.append(span)
            np.testing.assert_allclose(s["total_links"], sum(spans), atol=1e-7)
            # Fenwick's last slot is the total because capacity is a power of 2.
            np.testing.assert_allclose(s["fw"][-1], sum(spans), atol=1e-7)
            state["observations"] += 1
        return trace

    return trace, state


@pytest.mark.parametrize("seed", [1, 17])
def test_hudson_lineage_tails_and_recombination_weights(seed):
    # Observe the reference Python body between events. No production debug
    # branch or full-ARG instrumentation is needed in the compiled hot loop.
    fn = getattr(coal._hudson, "py_func", coal._hudson)
    trace, state = _lineage_invariant_observer(fn)
    previous_trace, random_state = sys.gettrace(), np.random.get_state()
    sys.settrace(trace)
    try:
        status, _, _ = fn(4, 100000., 1e-8, 10000., seed, *_hudson_buffers())
    finally:
        sys.settrace(previous_trace)
        np.random.set_state(random_state)
    assert status == 0 and state["observations"] > 5 and state["recombinations"] > 0


@pytest.mark.parametrize("forced", [0.0, np.nextafter(1.0, 0.0), "past-tail"])
def test_hudson_breakpoint_endpoint_draws(forced):
    """Endpoint breakpoint draws clamp to the representable interior.

    ``np.random.random`` is intercepted only at the breakpoint draw line
    of the reference Python body. A 0.0 draw lands exactly on
    ``left(head)``; a largest-sub-1 draw can round up onto
    ``right(tail)``, and ``'past-tail'`` pushes the draw to a quotient
    that lands strictly above it. Without the interior clamp these used
    to walk off the segment list and index the arrays at -1. The lineage
    invariants must still hold at every event boundary.
    """
    fn = getattr(coal._hudson, "py_func", coal._hudson)
    lines, first = inspect.getsourcelines(fn)
    bp_line = first + next(
        i for i, line in enumerate(lines)
        if "np.random.random()" in line and "slot_link" in line)
    original_random = np.random.random
    breakpoint_draws = 0

    def guarded_random(*args, **kwargs):
        nonlocal breakpoint_draws
        caller = inspect.currentframe().f_back
        if caller.f_code is fn.__code__ and caller.f_lineno == bp_line:
            breakpoint_draws += 1
            if forced == "past-tail":
                s = caller.f_locals
                slot = s["slot"]
                hi = s["seg_right"][s["slot_tail"][slot]]
                quotient = (hi - s["lo_pos"]) / s["slot_link"][slot]
                return np.nextafter(np.nextafter(quotient, np.inf), np.inf)
            return forced
        return original_random(*args, **kwargs)

    trace, state = _lineage_invariant_observer(fn)
    previous_trace, random_state = sys.gettrace(), np.random.get_state()
    np.random.random = guarded_random
    sys.settrace(trace)
    try:
        status, _, _ = fn(4, 100000., 1e-8, 10000., 3, *_hudson_buffers())
    finally:
        sys.settrace(previous_trace)
        np.random.random = original_random
        np.random.set_state(random_state)
    assert breakpoint_draws > 0
    assert status == 0 and state["observations"] > 5 and state["recombinations"] > 0


def _summaries(G, pos):
    af = G.mean(0)/2
    common = (af >= 0.1) & (af <= 0.9)
    X, pos = G[:, common].astype(float), pos[common]
    sd = X.std(0)
    X, pos = X[:, sd > 0], pos[sd > 0]
    X = (X-X.mean(0))/X.std(0)
    i, j = np.triu_indices(X.shape[1], 1)
    r2 = (X.T @ X / X.shape[0])[i, j]**2
    distance = pos[j]-pos[i]
    return [G.shape[1], (2*af*(1-af)).sum(),
            r2[distance < 5000].mean(), r2[distance >= 20000].mean()]


@pytest.mark.slow
@pytest.mark.parametrize("recomb_rate", [0, 1e-8])
def test_msprime_site_diversity_and_ld_summaries(recomb_rate):
    """Selected distributional checks, not a claim of backend equivalence."""
    msprime = pytest.importorskip("msprime")
    builtin, reference = [], []
    for seed in range(1, 76):
        G, pos, _ = coal.simulate_dosages(20, 50000, recomb_rate=recomb_rate, seed=seed)
        builtin.append(_summaries(G, pos))
        ts = msprime.sim_ancestry(20, ploidy=2, population_size=10000, sequence_length=50000,
                                 recombination_rate=recomb_rate, discrete_genome=False, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=1e-8, discrete_genome=False,
                                  random_seed=seed+100000, model=msprime.BinaryMutationModel())
        H = ts.genotype_matrix()
        reference.append(_summaries((H[:, ::2] + H[:, 1::2]).T, ts.tables.sites.position))
    a, b = np.asarray(builtin), np.asarray(reference)
    difference = np.abs(a.mean(0)-b.mean(0))
    standard_error = np.sqrt((a.var(0, ddof=1)+b.var(0, ddof=1))/len(a))
    assert np.all(difference < 5*standard_error), (difference, standard_error)
    if recomb_rate > 0:
        assert a[:, 2].mean() > a[:, 3].mean()
