"""Regression probes for causal eligibility and the default structure axis."""

import numpy as np
import pytest

import phensim
from phensim import phenotypes as phen


@pytest.mark.parametrize("simulate", [phensim.simulate_trait,
    phensim.simulate_binary_trait, phensim.simulate_confounded_trait,
    phensim.simulate_gxe_trait, phensim.simulate_correlated_traits])
@pytest.mark.parametrize("dtype", [np.int8, np.float64])
def test_automatic_causal_variants_have_observed_variation(simulate, dtype):
    G = np.ones((30, 20), dtype=dtype)
    if dtype == np.float64:
        G[:, :10] = 0.1  # exact constancy, despite roundoff in centering
    G[:, 17] = np.arange(30) % 3
    for seed in range(8):
        tr = simulate(G, n_causal=1, seed=seed)
        np.testing.assert_array_equal(tr["causal"], [17])


def test_causal_count_is_capped_at_variable_columns():
    G = np.tile([0, 1, 2, 0], (30, 1))
    G[:, 2] = np.arange(30) % 3
    tr = phensim.simulate_trait(G, architecture="qtl", n_causal=20)
    np.testing.assert_array_equal(tr["causal"], [2])
    assert tr["q"].var() == pytest.approx(0.5)


@pytest.mark.parametrize("causal", [[0], [0, 2]])
def test_explicit_constant_causal_column_is_rejected(causal):
    G = np.tile([0.1, 1, 2], (30, 1))
    G[:, 2] = np.arange(30) % 3
    with pytest.raises(ValueError, match="causal.*constant"):
        phensim.simulate_trait(G, architecture="qtl", causal=np.array(causal))


def test_no_variable_causal_columns_has_actionable_error():
    with pytest.raises(ValueError, match="polymorphic causal variant"):
        phensim.simulate_trait(np.ones((20, 5)), architecture="qtl")
    tr = phensim.simulate_trait(np.ones((20, 5)), h2=0)
    assert tr["causal"].size == 0
    np.testing.assert_array_equal(tr["liability"], tr["e"])


@pytest.mark.parametrize("h2,interaction_h2", [(0.5, 0.2), (0.2, 0.2), (0.5, 0)])
def test_gxe_zero_causal_error_names_components(h2, interaction_h2):
    G = phensim.simulate_independent(30, 20)
    with pytest.raises(ValueError, match="additive or interaction variance"):
        phensim.simulate_gxe_trait(G, h2=h2, interaction_h2=interaction_h2, n_causal=0)


def test_default_environment_is_invariant_to_eigenvector_sign(monkeypatch):
    G = phensim.simulate_independent(30, 50)
    kwargs = dict(h2=0, n_causal=0, seed=23)
    expected = phensim.simulate_confounded_trait(G, **kwargs)
    real_eigh = np.linalg.eigh

    def flipped(*args, **kwargs):
        lam, U = real_eigh(*args, **kwargs)
        U[:, -1] *= -1
        return lam, U

    monkeypatch.setattr(np.linalg, "eigh", flipped)
    actual = phensim.simulate_confounded_trait(G, **kwargs)
    for key in ("structure", "liability", "y"):
        np.testing.assert_array_equal(actual[key], expected[key])
    axis = real_eigh(phensim.grm(G))[1][:, -1]
    axis *= np.sign(axis[np.argmax(np.abs(axis))])
    np.testing.assert_allclose(expected["structure"],
        np.sqrt(0.6) * (axis - axis.mean()) / axis.std())


def test_binary_threshold_uses_the_small_tail(monkeypatch):
    liability = np.array([7.1, 7.4, 8.0])
    monkeypatch.setattr(phen, "simulate_trait", lambda *a, **k: {"liability": liability})
    tr = phensim.simulate_binary_trait(None, prevalence=1e-13)
    np.testing.assert_array_equal(tr["case_control"], [0, 1, 1])


def test_supplied_kinship_scale_is_retained():
    G = phensim.simulate_independent(30, 20)
    kwargs = dict(architecture="infinitesimal", h2=0.5, n_causal=0, seed=11)
    base = phensim.simulate_trait(G, K=np.eye(30), **kwargs)
    scaled = phensim.simulate_trait(G, K=3 * np.eye(30), **kwargs)
    np.testing.assert_allclose(scaled["u"], np.sqrt(3) * base["u"], rtol=1e-14)
