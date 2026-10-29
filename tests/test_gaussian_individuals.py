"""Population covariance and the seeded PPB/panel simulation contracts."""
import numpy as np
import pytest

import phensim


def test_gaussian_individuals_preserve_ppb_draw_order_and_prepared_snapshot():
    blocks = [(np.array([[1., .6], [.6, 1.]]), np.array([2, 0])),
              (np.eye(1), np.array([1]))]
    prepared = phensim.prepare_blocks(blocks)
    rng, oracle = np.random.default_rng(8), np.random.default_rng(8)
    expected = np.empty((200, 3))
    for R, ix in blocks:
        expected[:, ix] = oracle.standard_normal((200, len(ix))) @ np.linalg.cholesky(R).T
    got = phensim.simulate_gaussian_individuals(prepared, 200, seed=rng)
    np.testing.assert_array_equal(got, expected)
    assert rng.random() == oracle.random()
    np.testing.assert_array_equal(phensim.simulate_gaussian_individuals(blocks, 200, seed=8), got)
    blocks[0][0][:] = np.nan
    np.testing.assert_array_equal(phensim.simulate_gaussian_individuals(prepared, 200, seed=8), got)


def test_gaussian_individuals_population_moments_and_singular_support():
    R = np.array([[1., .4], [.4, 1.]])
    blocks = [(R, np.array([0, 2])), (np.ones((2, 2)), np.array([1, 3]))]
    x = phensim.simulate_gaussian_individuals(blocks, 50000, seed=4)
    target = np.eye(4)
    target[np.ix_([0, 2], [0, 2])] = R
    target[np.ix_([1, 3], [1, 3])] = 1
    np.testing.assert_allclose(x.mean(0), 0, atol=.015)
    np.testing.assert_allclose(x.T @ x / len(x), target, atol=.02)
    np.testing.assert_array_equal(x[:, 1], x[:, 3])
    assert np.any(x.mean(0) != 0)  # no realized-sample centering


@pytest.mark.parametrize("n", [0, -1, True, 2.5, np.nan])
def test_gaussian_individuals_reject_invalid_sample_size(n):
    with pytest.raises(ValueError, match="positive integer"):
        phensim.simulate_gaussian_individuals([(np.eye(2), np.arange(2))], n)


@pytest.mark.parametrize("chunk", [None, 7, 100])
def test_panel_supplied_rectangular_factor_and_rng(chunk):
    # R is indefinite; the caller deliberately draws from a rank-two projection.
    R = np.array([[1., .9, .9], [.9, 1., -.9], [.9, -.9, 1.]])
    F = np.array([[1., 0], [.4, .6], [.3, -.5]])
    blocks = [(R, np.arange(3))]
    rng, oracle = np.random.default_rng(4), np.random.default_rng(4)
    x = oracle.standard_normal((40, 2)) @ F.T
    x = (x-x.mean(0)) / x.std(0)
    expected = .9 * (x.T @ x / 40) + .1 * np.eye(3)
    got = phensim.shake_ld(blocks, 40, seed=rng, factors=[F], shrink=.1, chunk_size=chunk)
    np.testing.assert_allclose(got[0][0], expected, rtol=0, atol=6e-16)
    assert rng.random() == oracle.random()
    assert np.linalg.eigvalsh(got[0][0]).min() >= .1 - 1e-14
    rng, oracle = np.random.default_rng(4), np.random.default_rng(4)
    np.testing.assert_array_equal(phensim.shake_ld(blocks, None, seed=rng, factors=[F])[0][0], R)
    assert rng.random() == oracle.random()


@pytest.mark.parametrize("kwargs", [dict(factors=[]), dict(factors=[np.ones((3, 2))]),
    dict(factors=[np.full((2, 1), np.nan)]), dict(factors=[np.ones((2, 0))]),
    dict(factors=[np.eye(2)], jitter=.1)])
def test_panel_factor_validation(kwargs):
    with pytest.raises(ValueError, match="factor|jitter"):
        phensim.shake_ld([(np.eye(2), np.arange(2))], 20, **kwargs)
