"""Matrix-free genetic background and its covariance."""

import numpy as np
import pytest

import phensim
import phensim.phenotypes as phen
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


def _symmetric_float32(K):
    K = K.astype(np.float32)
    return (K + K.T) / 2


# Each case maps grm(G) to (the K passed in, the K the oracle decomposes):
# the default builds grm(G) itself; a supplied K, float32 included, is used
# as given.
@pytest.mark.parametrize("supplied", [
    pytest.param(lambda K: (None, K), id="default"),
    pytest.param(lambda K: (K, K), id="float64_k"),
    pytest.param(lambda K: (_symmetric_float32(K),) * 2, id="float32_k"),
])
def test_confounded_single_eigendecomposition(monkeypatch, supplied):
    G = phensim.simulate_independent(40, 200, seed=5)
    K_arg, K = supplied(grm(G))
    real_eigh = np.linalg.eigh
    calls = 0

    def spy(*a, **k):
        nonlocal calls
        calls += 1
        return real_eigh(*a, **k)

    monkeypatch.setattr(np.linalg, "eigh", spy)
    tr = phensim.simulate_confounded_trait(G, seed=16, K=K_arg)
    assert calls == 1

    # The cached eigenpair reproduces the original two-eigh seeded draw.
    rng = np.random.default_rng(16)
    sub = int(rng.integers(1, 2**31 - 1))
    inner = np.random.default_rng(sub)
    lam, U = real_eigh(K)
    u = U @ (np.sqrt(np.maximum(lam, 0) * 0.25) * inner.standard_normal(40))
    np.testing.assert_array_equal(tr["u"], np.sqrt(0.4) * u)
    np.testing.assert_allclose(
        tr["liability"], tr["structure"] + tr["u"] + tr["q"] + tr["e"],
        atol=1e-14)


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
