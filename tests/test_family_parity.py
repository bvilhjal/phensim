"""Bit parity with the family's benchmark simulators phensim now hosts.

Each oracle is a verbatim copy of the sibling code it names (constants
turned into arguments), so these tests pin the draw protocols the
siblings can migrate onto without re-freezing committed results.
"""

import numpy as np
import pytest

import phensim


def _ld(k, seed, n=300):
    rng = np.random.default_rng(seed)
    C = 0.8 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    X = rng.standard_normal((n, k)) @ np.linalg.cholesky(C).T
    X = (X - X.mean(0)) / X.std(0)
    return X.T @ X / n


K, NB = 30, 4
POP_R = [_ld(K, b) for b in range(NB)]
IDX = [np.arange(b * K, (b + 1) * K) for b in range(NB)]
POP_C = [np.linalg.cholesky(R + 1e-4 * np.eye(K)) for R in POP_R]
BLOCKS = list(zip(POP_R, IDX))
M = K * NB


def gv(a, b):  # bipred benchmarks/rg_architectures.py
    return sum(a[ix] @ (POP_R[i] @ b[ix]) for i, ix in enumerate(IDX))


def ldpred3_metrics_sumstats(beta, blocks, chols, n, rng):
    """ldpred3 benchmarks/_metrics.py ``sumstats``."""
    bhat = np.empty(beta.size)
    per_variant = np.ndim(n) > 0
    for (R, ix), chol in zip(blocks, chols):
        noise = (chol @ rng.standard_normal(len(ix)))
        bhat[ix] = np.asarray(R, np.float64) @ beta[ix] + noise / np.sqrt(
            n[ix] if per_variant else n)
    return bhat


def bipred_sim_effects(p, rg, rng, h2=0.5):
    """bipred benchmarks/rg_architectures.py ``sim_effects`` (p=1: infinitesimal)."""
    L = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
    b1 = np.zeros(M)
    b2 = np.zeros(M)
    c = np.ones(M, bool) if p == 1 else rng.random(M) < p
    if not c.any():
        c[rng.integers(M)] = True
    raw = L @ rng.standard_normal((2, int(c.sum())))
    b1[c] = raw[0]
    b2[c] = raw[1]
    b1 *= np.sqrt(h2 / gv(b1, b1))
    b2 *= np.sqrt(h2 / gv(b2, b2))
    return b1, b2


def bipred_sim_mixture(rng, n1_causal, n2_causal, n_shared, rho_beta, h2=0.5):
    """bipred benchmarks/mixer_overlap.py ``_sim_mixture`` (effects only)."""
    n_uniq1 = n1_causal - n_shared
    n_uniq2 = n2_causal - n_shared
    picks = rng.choice(M, n_shared + n_uniq1 + n_uniq2, replace=False)
    shared = picks[:n_shared]
    u1 = picks[n_shared:n_shared + n_uniq1]
    u2 = picks[n_shared + n_uniq1:]
    b1 = np.zeros(M)
    b2 = np.zeros(M)
    b1[u1] = rng.standard_normal(n_uniq1)
    b2[u2] = rng.standard_normal(n_uniq2)
    if n_shared:
        L = np.linalg.cholesky([[1.0, rho_beta], [rho_beta, 1.0]])
        raw = L @ rng.standard_normal((2, n_shared))
        b1[shared] = raw[0]
        b2[shared] = raw[1]
    b1 *= np.sqrt(h2 / gv(b1, b1))
    b2 *= np.sqrt(h2 / gv(b2, b2))
    return b1, b2


def bipred_sumstats_pair(b1, b2, n1, n2, rng, rho_e=0.0):
    """bipred benchmarks/rg_architectures.py ``sumstats_pair``."""
    bh1 = np.empty(M)
    bh2 = np.empty(M)
    for i, ix in enumerate(IDX):
        u1 = rng.standard_normal(K)
        u2 = (rho_e * u1 + np.sqrt(1 - rho_e ** 2) * rng.standard_normal(K)
              if rho_e else rng.standard_normal(K))
        bh1[ix] = POP_R[i] @ b1[ix] + (POP_C[i] @ u1) / np.sqrt(n1)
        bh2[ix] = POP_R[i] @ b2[ix] + (POP_C[i] @ u2) / np.sqrt(n2)
    return bh1, bh2


def family_ref_panel(nref, shrink, seed):
    """ldpred3/gwfm ``_realistic_ld.panel_genome``, bipred ``ref_panel``."""
    rng = np.random.default_rng(seed)
    ref = []
    for b in range(NB):
        Z = rng.standard_normal((nref, K)) @ POP_C[b].T
        Z = (Z - Z.mean(0)) / Z.std(0)
        Rr = (1 - shrink) * ((Z.T @ Z) / nref) + shrink * np.eye(K)
        ref.append(Rr.astype(np.float32))
    return ref


@pytest.mark.parametrize("n", [50_000, np.linspace(1e4, 9e4, M)])
def test_sumstats_match_ldpred3_metrics(n):
    beta = np.random.default_rng(1).standard_normal(M)
    expected = ldpred3_metrics_sumstats(beta, BLOCKS, POP_C, n, np.random.default_rng(5))
    for kw in (dict(jitter=1e-4), dict(factors=POP_C)):
        got = phensim.simulate_sumstats(beta, BLOCKS, n, seed=np.random.default_rng(5), **kw)
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("p", [1.0, 0.2, 0.01, 1e-9])
@pytest.mark.parametrize("rg", [0.0, 0.6, -0.95])
def test_effects_pair_match_bipred_shared_architectures(p, rg):
    expected = bipred_sim_effects(p, rg, np.random.default_rng(7))
    got = phensim.simulate_effects_pair(BLOCKS, 0.5, 0.5, rg, p=p, seed=np.random.default_rng(7))
    np.testing.assert_array_equal(got, expected)
    assert phensim.genetic_correlation(*got, BLOCKS) == gv(*got) / np.sqrt(gv(got[0], got[0]) * gv(got[1], got[1]))


@pytest.mark.parametrize("counts", [(30, 30, 12, 0.7), (50, 20, 20, -0.4), (10, 25, 0, 0.5)])
def test_effects_pair_match_bipred_four_state_mixture(counts):
    n_a, n_b, shared, rho = counts
    expected = bipred_sim_mixture(np.random.default_rng(11), n_a, n_b, shared, rho)
    got = phensim.simulate_effects_pair(BLOCKS, 0.5, 0.5, rho, n_causal=(n_a, n_b),
                                        n_shared=shared, seed=np.random.default_rng(11))
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("rho_e", [0.0, 0.3])
def test_sumstats_pair_match_bipred_with_unequal_n(rho_e):
    b1, b2 = bipred_sim_effects(0.2, 0.6, np.random.default_rng(3))
    expected = bipred_sumstats_pair(b1, b2, 40_000, 25_000, np.random.default_rng(9), rho_e)
    got = phensim.simulate_sumstats_pair(b1, b2, BLOCKS, 40_000, rho_e, seed=np.random.default_rng(9),
                                         n_b=25_000, jitter=1e-4)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("shrink", [0.0, 0.05])
def test_shake_ld_matches_family_reference_panels(shrink):
    expected = family_ref_panel(120, shrink, seed=4)
    got = phensim.shake_ld(BLOCKS, 120, seed=4, shrink=shrink, jitter=1e-4)
    for R, (Rg, _ix) in zip(expected, got):
        np.testing.assert_array_equal(Rg.astype(np.float32), R)


def test_supplied_factors_accept_thresholded_ld():
    # Thresholding small entries leaves LD indefinite: the default factor
    # refuses it, a caller's clipped root (ldpred3 p_vs_nref) is used as given.
    R = np.where(np.abs(POP_R[0]) < 0.3, 0.0, POP_R[0])
    blocks = [(R, np.arange(K))]
    assert np.linalg.eigvalsh(R)[0] < -1e-3
    with pytest.raises(ValueError, match="semidefinite"):
        phensim.simulate_sumstats(np.zeros(K), blocks, 1000)
    w, V = np.linalg.eigh(R)
    root = (V * np.sqrt(np.maximum(w, 1e-6))) @ V.T
    bhat = phensim.simulate_sumstats(np.zeros(K), blocks, 1000, seed=2, factors=[root])
    np.testing.assert_array_equal(bhat, root @ np.random.default_rng(2).standard_normal(K) / np.sqrt(1000))


def test_four_state_layout_and_targets():
    a, b = phensim.simulate_effects_pair(BLOCKS, 0.3, 0.6, 0.8, n_causal=(40, 25), n_shared=10, seed=1)
    assert (np.count_nonzero(a), np.count_nonzero(b), np.count_nonzero(a * b)) == (40, 25, 10)
    entries = [(R, ix, None) for R, ix in BLOCKS]
    from phensim.sumstats import _quadratic
    np.testing.assert_allclose([_quadratic(a, a, entries), _quadratic(b, b, entries)], [0.3, 0.6])
    x, y = phensim.simulate_effects_pair(BLOCKS, rho=-1.0, p=0.5, seed=2)
    assert phensim.genetic_correlation(x, y, BLOCKS) == pytest.approx(-1.0)


@pytest.mark.parametrize("kwargs", [
    dict(), dict(p=0.1, n_causal=5), dict(p=0.0), dict(p=1.5), dict(n_causal=5, n_shared=6),
    dict(n_causal=(M, M), n_shared=0), dict(n_causal=-1), dict(n_causal=3.0),
    dict(p=0.1, rho=1.5), dict(p=0.1, h2_a=2.0),
])
def test_effects_pair_validation(kwargs):
    with pytest.raises(ValueError):
        phensim.simulate_effects_pair(BLOCKS, **kwargs)


def test_noise_option_validation():
    beta = np.zeros(M)
    for kw in (dict(jitter=-1.0), dict(jitter=np.nan), dict(jitter=1e-4, factors=POP_C),
               dict(factors=POP_C[:2]), dict(factors=[np.ones((K + 1, K))] * NB)):
        with pytest.raises(ValueError):
            phensim.simulate_sumstats(beta, BLOCKS, 1000, **kw)
    for shrink in (-0.1, 1.1, np.nan, "a"):
        with pytest.raises(ValueError, match="shrink"):
            phensim.shake_ld(BLOCKS, 50, shrink=shrink)
    bad = [(np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]]), np.arange(3))]
    with pytest.raises(ValueError, match="jitter"):
        phensim.simulate_sumstats(np.zeros(3), bad, 1000, jitter=1e-4)
    assert np.isnan(phensim.genetic_correlation(np.zeros(M), np.ones(M), BLOCKS))
