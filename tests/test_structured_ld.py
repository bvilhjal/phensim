"""Independent properties of structured LD and explicit environmental means."""
import numpy as np
import pytest

import phensim


@pytest.mark.parametrize("fst", [0.0, 0.1])
def test_structured_ld_has_within_population_ld_and_expected_divergence(fst):
    G, labels = phensim.simulate_population_structure(
        1800, 400, n_pops=3, fst=fst, model="balding-nichols",
        block_sizes=[20] * 20, rho=0.85, seed=51)
    a, b = G[labels == 0].astype(float), G[labels == 1].astype(float)
    # Removing population means isolates local LD from ancestry correlations.
    Z = (a - a.mean(0)) / a.std(0)
    within = np.mean(np.mean(Z[:, 0::20] * Z[:, 1::20], axis=0))
    across = np.mean(np.mean(Z[:, 19:-1:20] * Z[:, 20::20], axis=0))
    assert within > 0.45
    assert abs(across) < 0.06
    divergence = np.mean(np.abs(a.mean(0) - b.mean(0))) / 2
    assert divergence > 0.1 if fst else divergence < 0.025
    assert G.dtype == np.int8 and set(np.unique(G)) == {0, 1, 2}
    again = phensim.simulate_population_structure(
        1800, 400, n_pops=3, fst=fst, model="balding-nichols",
        block_sizes=[20] * 20, rho=0.85, seed=51)
    np.testing.assert_array_equal(G, again[0])
    np.testing.assert_array_equal(labels, again[1])


def test_existing_structured_draw_unchanged():
    # Reconstruct the original public algorithm, including RNG ordering.
    for model in ["normal", "balding-nichols"]:
        rng = np.random.default_rng(123)
        base = np.clip(0.3 + rng.normal(0, 0.05, 25), 0.05, 0.95)
        if model == "normal":
            freq = np.clip(base[:, None] + rng.normal(0, 1, (25, 3))
                           * np.sqrt(0.1 * base * (1-base))[:, None], 0.01, 0.99)
        else:
            freq = np.clip(rng.beta((9*base)[:, None], (9*(1-base))[:, None],
                                   size=(25, 3)), 0.01, 0.99)
        labels = rng.integers(0, 3, 50)
        expected = rng.binomial(2, freq[:, labels].T).astype(np.int8)
        G, got_labels = phensim.simulate_population_structure(
            50, 25, model=model, seed=123)
        np.testing.assert_array_equal(G, expected)
        np.testing.assert_array_equal(got_labels, labels)


@pytest.mark.parametrize("kwargs", [
    {"fst": -0.1}, {"fst": np.nan}, {"fst": 1},
    {"block_sizes": [4, 5]}, {"block_sizes": [10.]},
    {"block_sizes": [0, 10]}, {"block_sizes": [10], "rho": np.nan},
])
def test_invalid_structured_ld_inputs(kwargs):
    with pytest.raises(ValueError):
        phensim.simulate_population_structure(20, 10, **kwargs)


def test_explicit_environment_reconstruction_and_matrix_free(monkeypatch):
    import phensim.phenotypes as module
    G = phensim.simulate_independent(90, 80, seed=7)
    E = np.repeat([-2., 0., 1.], 30)
    def no_grm(*args):
        raise AssertionError("explicit environment must not materialize kinship")
    monkeypatch.setattr(module, "_grm", no_grm)
    tr = phensim.simulate_confounded_trait(
        G, environment=E, confounding_strength=.2, h2=.4,
        architecture="qtl", causal=np.array([2, 5, 9]), seed=99)
    np.testing.assert_allclose(tr["structure"], np.sqrt(.2)*(E-E.mean())/E.std())
    np.testing.assert_allclose(tr["liability"], tr["u"]+tr["q"]+tr["e"]+tr["structure"])
    Z = G[:, tr["causal"]].astype(float)
    Z = (Z-Z.mean(0))/Z.std(0)
    np.testing.assert_allclose(Z @ tr["effects"], tr["q"])
    null = phensim.simulate_confounded_trait(
        G, environment=E, confounding_strength=0, h2=0, n_causal=0)
    assert not null["u"].any() and not null["q"].any()
    assert not null["structure"].any() and null["causal"].size == 0


@pytest.mark.parametrize("environment", [np.ones(20), np.zeros(19), np.full(20, np.nan)])
def test_bad_explicit_environment(environment):
    with pytest.raises(ValueError):
        phensim.simulate_confounded_trait(
            phensim.simulate_independent(20, 30), environment=environment)
