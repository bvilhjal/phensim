"""Normal quantile tails against the independent standard-library inverse."""

from statistics import NormalDist

import numpy as np
import pytest

from phensim._common import norm_isf, norm_ppf
from phensim.genotypes import simulate_ar1_blocks


@pytest.mark.parametrize("clip", [True, False])
def test_quantiles_cover_float64_interior(clip):
    p = np.unique(np.r_[
        np.nextafter(0.0, 1.0), np.logspace(-323, -2, 100),
        np.linspace(0.01, 0.99, 100),
        1 - np.logspace(-15, -2, 30), np.nextafter(1.0, 0.0),
    ])
    # NormalDist uses Wichura's AS241, independently of Acklam's approximation.
    expected = np.array([NormalDist().inv_cdf(float(x)) for x in p])
    np.testing.assert_allclose(norm_ppf(p, clip=clip), expected, rtol=2e-9, atol=1e-9)
    np.testing.assert_allclose(norm_isf(p, clip=clip), -expected, rtol=2e-9, atol=1e-9)
    assert np.all(np.diff(norm_ppf(p, clip=clip)) > 0)
    assert np.all(np.diff(norm_isf(p, clip=clip)) < 0)
    assert float(norm_isf(1e-13, clip=clip)) == pytest.approx(7.348796102800677, abs=2e-8)


def test_quantile_boundaries_and_invalid_values():
    p = np.array([-np.inf, -0.1, 0, 0.5, 1, 1.1, np.inf, np.nan])
    expected = np.array([np.nan, np.nan, -np.inf, 0, np.inf, np.nan, np.nan, np.nan])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        np.testing.assert_array_equal(norm_ppf(p, clip=False), expected)
        np.testing.assert_array_equal(norm_isf(p, clip=False), -expected)
        assert np.isneginf(norm_ppf(0, clip=False))
        assert np.isposinf(norm_isf(0, clip=False))
    assert norm_ppf(np.empty((2, 0))).shape == (2, 0)


def test_clipped_endpoints_are_nearest_interior_values():
    p = np.array([[-0.1, 0], [1, 1.1]])
    expected = np.array([
        [norm_ppf(np.nextafter(0.0, 1.0), clip=False)] * 2,
        [norm_ppf(np.nextafter(1.0, 0.0), clip=False)] * 2,
    ])
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(norm_ppf(p), expected)
    np.testing.assert_array_equal(norm_isf(p), -expected)
    assert np.isnan(norm_ppf(np.nan))
    assert np.isnan(norm_isf(np.nan))


@pytest.mark.parametrize("method", ["cholesky", "scan"])
@pytest.mark.parametrize("phased", [True, False])
def test_ar1_exact_frequency_endpoints_even_for_extreme_latent_draws(method, phased):
    class ExtremeGenerator(np.random.Generator):
        def standard_normal(self, size):
            return np.broadcast_to([40.0, -40.0], size).copy()

    # Finite clipped thresholds would incorrectly code these two columns.
    rng = ExtremeGenerator(np.random.PCG64(0))
    g, _ = simulate_ar1_blocks(
        2, [2], maf=[0, 1], rho=0, seed=rng, method=method, phased=phased)
    np.testing.assert_array_equal(g[..., 0], 0)
    np.testing.assert_array_equal(g[..., 1], 1 if phased else 2)
