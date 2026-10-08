"""Multi-population and admixture simulators against their defining models."""
import numpy as np
import pytest

import phensim


def _adjacent_r(G, block):
    """Mean adjacent-variant correlation inside blocks of length ``block``."""
    X = G.astype(float)
    Z = (X - X.mean(0)) / X.std(0)
    r = np.mean(Z[:, :-1] * Z[:, 1:], axis=0)
    return r[np.arange(r.size) % block != block - 1].mean()


def test_drift_frequencies_variance_controls_and_seed():
    p = np.full(40_000, 0.4)
    f = phensim.drift_frequencies(40_000, 3, fst=[0.0, 0.05, 0.2], ancestral=p,
                                  min_freq=0, seed=4)
    assert f.shape == (3, 40_000) and f.dtype == np.float64
    np.testing.assert_array_equal(f[0], p)
    for row, F in zip(f[1:], [0.05, 0.2]):
        assert abs(np.var(row) / (0.4 * 0.6) - F) < 0.1 * F
        assert abs(row.mean() - 0.4) < 0.005
    normal = phensim.drift_frequencies(40_000, 1, 0.05, ancestral=p, model="normal", seed=4)
    assert abs(np.var(normal) / 0.24 - 0.05) < 0.005
    clipped = phensim.drift_frequencies(5_000, 2, 0.5, seed=1)
    assert clipped.min() >= 0.01 and clipped.max() <= 0.99
    np.testing.assert_array_equal(clipped, phensim.drift_frequencies(5_000, 2, 0.5, seed=1))


@pytest.mark.parametrize("model", ["normal", "balding-nichols"])
def test_drift_reproduces_structured_frequency_stream(model):
    # The original simulate_population_structure formulas and RNG order.
    rng = np.random.default_rng(123)
    base = np.clip(0.3 + rng.normal(0, 0.05, 25), 0.05, 0.95)
    if model == "normal":
        expected = base[:, None] + rng.normal(0, 1, (25, 3)) * np.sqrt(0.1 * base * (1 - base))[:, None]
    else:
        expected = rng.beta((base * 0.9 / 0.1)[:, None], ((1 - base) * 0.9 / 0.1)[:, None],
                            size=(25, 3))
    rng = np.random.default_rng(123)
    base = np.clip(0.3 + rng.normal(0, 0.05, 25), 0.05, 0.95)
    got = phensim.drift_frequencies(25, 3, 0.1, ancestral=base, model=model, seed=rng)
    np.testing.assert_array_equal(got, np.clip(expected, 0.01, 0.99).T)


def test_populations_are_sequential_ar1_draws_with_own_ld():
    freqs = phensim.drift_frequencies(120, 2, 0.1, seed=2)
    sizes = [np.array([30, 30, 60]), np.array([10] * 12)]
    G, labels = phensim.simulate_populations([40, 25], freqs, sizes, rho=[0.9, 0.2], seed=8)
    rng = np.random.default_rng(8)
    first, _ = phensim.simulate_ar1_blocks(40, sizes[0], maf=freqs[0], rho=0.9,
                                           method="scan", seed=rng)
    second, _ = phensim.simulate_ar1_blocks(25, sizes[1], maf=freqs[1], rho=0.2,
                                            method="scan", seed=rng)
    np.testing.assert_array_equal(G, np.vstack([first, second]))
    np.testing.assert_array_equal(labels, np.repeat([0, 1], [40, 25]))
    H, _ = phensim.simulate_populations([40, 25], freqs, sizes, rho=[0.9, 0.2],
                                        phased=True, seed=8)
    np.testing.assert_array_equal(H.sum(axis=1), G)
    shared, _ = phensim.simulate_populations([3000, 3000], freqs, [20] * 6,
                                             rho=[0.9, 0.3], seed=1)
    assert _adjacent_r(shared[:3000], 20) > _adjacent_r(shared[3000:], 20) + 0.2


def test_admixture_ld_matches_pulse_theory():
    # One-SNP blocks: every correlation is admixture LD. Under the pulse,
    # corr(d) = delta^2 a(1-a) exp(-T d / 100) / (pbar (1 - pbar)).
    m = 600
    freqs = np.vstack([np.full(m, 0.9), np.full(m, 0.1)])
    H, local = phensim.simulate_admixed(8_000, freqs, [1] * m, [0.5, 0.5],
                                        generations=10, cm=0.02, phased=True, seed=3)
    X = H.reshape(-1, m).astype(float)
    for d in (1, 100, 300):
        r = np.mean([np.corrcoef(X[:, j], X[:, j + d])[0, 1] for j in range(0, m - d, 23)])
        assert abs(r - 0.64 * np.exp(-0.1 * d * 0.02)) < 0.02
    A = local.reshape(-1, m)
    switches = (np.diff(A, axis=1) != 0).sum(axis=1).mean()
    assert abs(switches / (0.1 * (m - 1) * 0.02 * 0.5) - 1) < 0.08
    assert abs((A == 0).mean() - 0.5) < 0.02


def test_individual_proportions_and_chromosome_restarts():
    freqs = phensim.drift_frequencies(200, 3, 0.1, seed=5)
    alpha = np.tile([[1, 0, 0], [0, 0, 1], [0.2, 0.3, 0.5]], (1000, 1))
    G, local = phensim.simulate_admixed(3000, freqs, [20] * 10, alpha, generations=6,
                                        cm=0.5, chromosome=np.repeat([1, 2], 100), seed=9)
    assert G.shape == (3000, 200) and local.shape == (3000, 2, 200)
    assert G.dtype == local.dtype == np.int8
    assert np.all(local[0::3] == 0) and np.all(local[1::3] == 2)
    mixed = local[2::3]
    np.testing.assert_allclose(np.bincount(mixed.ravel(), minlength=3) / mixed.size,
                               [0.2, 0.3, 0.5], atol=0.02)
    # A vanishing pulse: one ancestry per chromosome, independent across them.
    _, still = phensim.simulate_admixed(4000, freqs[:2], [20] * 10, [0.5, 0.5],
                                        generations=1e-9, cm=0.5,
                                        chromosome=np.repeat([1, 2], 100), seed=2)
    assert np.all(still[:, :, :100] == still[:, :, :1])
    assert np.all(still[:, :, 100:] == still[:, :, 100:101])
    assert abs(np.mean(still[:, :, 0] == still[:, :, 100]) - 0.5) < 0.03


def test_tracts_follow_their_population_model():
    freqs = phensim.drift_frequencies(400, 2, 0.1, seed=1)
    pure, _ = phensim.simulate_admixed(5000, freqs, [20] * 20, [1, 0],
                                       generations=1e-9, rho=[0.9, 0.3], seed=5)
    pop, _ = phensim.simulate_populations([5000], freqs[:1], [20] * 20, rho=0.9, seed=6)
    assert abs(_adjacent_r(pure, 20) - _adjacent_r(pop, 20)) < 0.02
    assert np.mean(np.abs(pure.mean(0) / 2 - freqs[0])) < 0.01
    G, local = phensim.simulate_admixed(4000, freqs, [20] * 20, [0.5, 0.5],
                                        generations=4, rho=[0.9, 0.3], phased=True, seed=7)
    for k in range(2):
        hits = (local == k)
        observed = (G * hits).sum(axis=(0, 1)) / hits.sum(axis=(0, 1))
        assert np.mean(np.abs(observed - freqs[k])) < 0.02
    dosages, again = phensim.simulate_admixed(4000, freqs, [20] * 20, [0.5, 0.5],
                                              generations=4, rho=[0.9, 0.3], seed=7)
    np.testing.assert_array_equal(G.sum(axis=1), dosages)
    np.testing.assert_array_equal(local, again)


@pytest.mark.parametrize("call", [
    lambda: phensim.drift_frequencies(10, 2, fst=1.0),
    lambda: phensim.drift_frequencies(10, 2, fst=[0.1, 0.2, 0.3]),
    lambda: phensim.drift_frequencies(10, 2, ancestral=np.r_[np.zeros(1), np.full(9, .5)]),
    lambda: phensim.drift_frequencies(10, 2, model="logit"),
    lambda: phensim.drift_frequencies(10, 2, min_freq=0.5),
    lambda: phensim.simulate_populations(5, np.full((1, 10), .3), [10]),
    lambda: phensim.simulate_populations([5, 5], np.full((1, 10), .3), [10]),
    lambda: phensim.simulate_populations([5], np.full((1, 10), .3), [4, 5]),
    lambda: phensim.simulate_populations([5], np.full((1, 10), 1.2), [10]),
    lambda: phensim.simulate_populations([5, 5], np.full((2, 10), .3), [10], rho=[.5, 2]),
    lambda: phensim.simulate_admixed(5, np.full((2, 10), .3), [10], [.5, .5, 0]),
    lambda: phensim.simulate_admixed(5, np.full((2, 10), .3), [10], [-1, 2]),
    lambda: phensim.simulate_admixed(5, np.full((2, 10), .3), [10], [.5, .5], generations=0),
    lambda: phensim.simulate_admixed(5, np.full((2, 10), .3), [10], [.5, .5], cm=-1),
    lambda: phensim.simulate_admixed(5, np.full((2, 10), .3), [10], [.5, .5],
                                     cm=np.arange(10.)[::-1]),
    lambda: phensim.simulate_admixed(5, np.full((2, 10), .3), [10], [.5, .5],
                                     chromosome=[1, 1, 2, 2, 1, 1, 1, 1, 1, 1]),
])
def test_invalid_ancestry_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_split_coalescent_divergence_and_local_ancestry():
    pytest.importorskip("msprime")
    out = phensim.simulate_split_coalescent(
        [300, 300, 300], 3000, 200, fst=0.1, admixed=300, proportions=[0.2, 0.3, 0.5],
        generations=8, seed=11)
    G, pop, local = out["G"], out["population"], out["local_ancestry"]
    assert G.shape == (1200, 3000) and G.dtype == np.int8 and local.shape == (300, 2, 3000)
    np.testing.assert_array_equal(pop, np.repeat([0, 1, 2, 3], 300))
    assert np.all(np.diff(out["positions"]) > 0) and len(out["blocks"]) == 15
    p = np.stack([G[pop == k].mean(0) / 2 for k in range(3)])
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        x, y = p[a], p[b]
        num = (x - y) ** 2 - x * (1 - x) / 599 - y * (1 - y) / 599
        # A few Mb hold few independent genealogies: 20 seeds span 0.08-0.15.
        assert 0.05 < num.sum() / (x * (1 - y) + y * (1 - x)).sum() < 0.2
    np.testing.assert_allclose(np.bincount(local.ravel(), minlength=3) / local.size,
                               [0.2, 0.3, 0.5], atol=0.06)
    # Admixed individuals homozygous for ancestry 2 carry its frequencies.
    both = (local[:, 0] == 2) & (local[:, 1] == 2)
    est = (G[pop == 3] * both).sum(0) / np.maximum(both.sum(0), 1) / 2
    keep = both.sum(0) > 30
    assert keep.sum() > 1000 and np.corrcoef(est[keep], p[2][keep])[0, 1] > 0.9
    again = phensim.simulate_split_coalescent(
        [300, 300, 300], 3000, 200, fst=0.1, admixed=300, proportions=[0.2, 0.3, 0.5],
        generations=8, seed=11)
    np.testing.assert_array_equal(G, again["G"])
    np.testing.assert_array_equal(local, again["local_ancestry"])
    plain = phensim.simulate_split_coalescent([50, 50], 400, 100, fst=0.05, seed=3)
    assert plain["local_ancestry"] is None and plain["G"].shape == (100, 400)


@pytest.mark.parametrize("kwargs", [
    {"n": [100]}, {"fst": 0}, {"fst": 1},
    {"admixed": 10}, {"admixed": 10, "proportions": [1, 0, 0]},
    {"admixed": 10, "proportions": [.5, .5], "generations": 5000},
    {"admixed": -1}, {"block_size": 500},
])
def test_invalid_split_coalescent_inputs(kwargs):
    args = {"n": [20, 20], "m": 400, "block_size": 100, "fst": 0.1}
    args.update(kwargs)
    with pytest.raises(ValueError):
        phensim.simulate_split_coalescent(args.pop("n"), args.pop("m"), **args)
