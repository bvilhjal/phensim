"""Summary-statistic simulation from genotype-level truth.

The RSS (regression-on-summary-statistics) oracle shared across the
family's benchmark suites: given per-variant effects ``beta`` and
block-diagonal population LD, marginal GWAS effects are
``bhat = R beta + N(0, R / n)``. Plus the effect-size architectures the
draws start from, a marginal GWAS scan over individual-level genotypes,
and a reference-panel LD-noise generator.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "simulate_effects",
    "simulate_sumstats",
    "simulate_sumstats_pair",
    "gwas_scan",
    "shake_ld",
]

#: Blocks are ``(R, ix)`` pairs: a correlation matrix and the variant
#: indices it covers. Functions consume them in sequence, drawing one
#: block of RNG noise per entry in block order.
Block = Tuple[np.ndarray, np.ndarray]


def _as_blocks(blocks: Sequence[Block]) -> list:
    """LDpred3-style coverage and tiled dense-correlation checks.

    Preserve input order (and hence RNG order); never assemble genome-wide
    dense LD. Dense floating correlations are required, not encoded D8/LR8.
    PSD is checked by the factorization used by each consumer.
    """
    normalized = []
    for item in blocks:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("each block must be an (R, ix) pair")
        R, ix = map(np.asarray, item)
        if ix.ndim != 1 or ix.size == 0 or not np.issubdtype(ix.dtype, np.integer):
            raise ValueError("LD block indices must be non-empty integer vectors")
        if np.any(ix < 0) or np.unique(ix).size != ix.size:
            raise ValueError("LD block indices must be unique and nonnegative")
        if R.shape != (ix.size, ix.size):
            raise ValueError("each block must be (R, ix) with R len(ix) x len(ix)")
        if not np.issubdtype(R.dtype, np.number) or np.iscomplexobj(R):
            raise ValueError("LD must contain real numeric correlations")
        exact_symmetry = True
        for start in range(0, ix.size, 256):
            band, transpose = R[start:start + 256], R[:, start:start + 256].T
            if not np.isfinite(band).all():
                raise ValueError("LD correlations must be finite")
            if not np.allclose(band, transpose, rtol=1e-7, atol=1e-10):
                raise ValueError("LD correlations must be symmetric")
            if np.any(band < -1.0000001) or np.any(band > 1.0000001):
                raise ValueError("LD correlations must lie in [-1, 1]; decode encoded LD first")
            exact_symmetry &= np.array_equal(band, transpose)
        if not np.allclose(np.diag(R), 1.0, rtol=1e-7, atol=1e-10):
            raise ValueError("LD correlations must have a unit diagonal")
        if not exact_symmetry:
            # Canonicalize roundoff only, so signal and noise use the same R.
            R = (np.asarray(R, dtype=np.float64) + R.T) * 0.5
        normalized.append((R, ix))
    m = sum(ix.size for _, ix in normalized)
    if m == 0:
        raise ValueError("LD blocks must cover at least one variant")
    # Allocate by the number of supplied indices, not their maximum: a bad
    # index must not cause a huge allocation before it can be rejected.
    seen = np.zeros(m, dtype=bool)
    for _, ix in normalized:
        if np.any(ix >= m):
            raise ValueError("LD blocks must cover every index in 0..m-1 exactly once")
        if np.any(seen[ix]):
            raise ValueError("LD block indices must not overlap or repeat")
        seen[ix] = True
    if not seen.all():
        raise ValueError("LD blocks must cover every index in 0..m-1 exactly once")
    return normalized


def _chol(R: np.ndarray) -> np.ndarray:
    """A covariance factor: Cholesky for PD, eigenfactor for singular PSD.

    Negative eigenvalues beyond floating-point roundoff are errors. Never
    add diagonal noise: a singular LD block has a genuine null space.
    """
    R = np.asarray(R, dtype=np.float64)
    try:
        return np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(R)
        tolerance = 64 * np.finfo(float).eps * R.shape[0] * max(1.0, values[-1])
        if values[0] < -tolerance:
            raise ValueError("LD correlations must be positive semidefinite") from None
        return vectors * np.sqrt(np.maximum(values, 0.0))


def _block_grid(blocks: Sequence[Block]) -> int:
    return sum(ix.size for _, ix in blocks)


def _effects_vector(beta, m):
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (m,) or not np.isfinite(beta).all():
        raise ValueError("beta must be a finite vector matching the LD blocks")
    return beta


def _sample_size(n, m):
    n = np.asarray(n, dtype=float)
    if n.shape not in ((), (m,)) or not np.isfinite(n).all() or np.any(n <= 0):
        raise ValueError("n must be positive and finite, scalar or one value per variant")
    return n


def simulate_effects(
    blocks: Sequence[Block],
    h2: float = 0.5,
    n_causal: Optional[int] = None,
    architecture: str = "sparse",
    maf: Optional[np.ndarray] = None,
    alpha: float = -0.3,
    seed: Union[int, np.random.Generator, None] = 0,
) -> np.ndarray:
    """Effect sizes with ``beta' R beta = h2`` over the block-diagonal LD.

    ``architecture`` draws the effect *shape*; the result is then rescaled
    so the population genetic variance under ``blocks`` hits ``h2``
    exactly:

    - ``'sparse'``: ``n_causal`` random normal effects (required);
    - ``'polygenic'``: every variant;
    - ``'maf'``: every variant, scaled by ``[2 f (1-f)]^(alpha/2)`` with
      ``alpha`` the usual negative MAF exponent (needs ``maf``);
    - ``'equal'``: ``n_causal`` random-sign, equal-magnitude effects.

    ``maf`` is a per-variant array indexed like the blocks.
    """
    if architecture not in ("sparse", "polygenic", "maf", "equal"):
        raise ValueError(f"unknown architecture {architecture!r}")
    if not 0 <= h2 <= 1:
        raise ValueError("h2 must be in [0, 1]")
    blocks = _as_blocks(blocks)
    m = _block_grid(blocks)
    if architecture in ("sparse", "equal") and n_causal is None:
        raise ValueError(f"architecture {architecture!r} needs n_causal")
    if architecture == "maf" and maf is None:
        raise ValueError("architecture 'maf' needs per-variant maf")
    # This consumer does not draw LD noise, but its variance claim still
    # requires valid PSD blocks. Use the same check as the noise samplers.
    for R, _ in blocks:
        _chol(R)
    rng = np.random.default_rng(seed)
    beta = np.zeros(m)
    if architecture in ("sparse", "equal"):
        if (isinstance(n_causal, (bool, np.bool_)) or not isinstance(n_causal, (int, np.integer))
                or n_causal < 0):
            raise ValueError("n_causal must be a nonnegative integer")
        causal = rng.choice(m, size=min(int(n_causal), m), replace=False)
        if architecture == "equal":
            beta[causal] = np.sign(rng.standard_normal(causal.size))
        else:
            beta[causal] = rng.standard_normal(causal.size)
    elif architecture == "polygenic":
        beta = rng.standard_normal(m)
    else:
        f = np.asarray(maf, dtype=float)
        if f.shape != (m,) or not np.isfinite(f).all() or np.any((f <= 0) | (f >= 1)) or not np.isfinite(alpha):
            raise ValueError("maf must be a finite length-m vector in (0, 1), and alpha finite")
        beta = rng.standard_normal(m) * (2.0 * f * (1.0 - f)) ** (alpha / 2.0)
    if not np.isfinite(beta).all():
        raise ValueError("effect architecture produced non-finite effects")
    if h2 == 0:
        return np.zeros(m)
    var = sum(beta[ix] @ (np.asarray(R, np.float64) @ beta[ix]) for R, ix in blocks)
    if var <= 0:
        raise ValueError("zero genetic variance; check n_causal / architecture")
    return beta * np.sqrt(h2 / var)


def simulate_sumstats(
    beta: np.ndarray,
    blocks: Sequence[Block],
    n,
    seed: Union[int, np.random.Generator, None] = 0,
) -> np.ndarray:
    """Marginal effects from the LDpred model: ``R beta + N(0, R / n)``.

    ``n`` is the GWAS sample size -- a scalar, or a per-variant vector
    indexed by the same ``ix`` as the blocks (heterogeneous N). One RNG
    draw per block, in block order. Blocks must tile 0..m-1 exactly once
    and contain finite, symmetric, unit-diagonal PSD correlations. For
    heterogeneous N the noise covariance is D R D, D_jj = 1/sqrt(n_j);
    this is an oracle model, not a model of arbitrary sample missingness.
    """
    blocks = _as_blocks(blocks)
    m = _block_grid(blocks)
    beta = _effects_vector(beta, m)
    n = _sample_size(n, m)
    rng = np.random.default_rng(seed)
    bhat = np.empty(m)
    per_variant = np.ndim(n) > 0
    for R, ix in blocks:
        R = np.asarray(R, np.float64)
        noise = _chol(R) @ rng.standard_normal(len(ix))
        bhat[ix] = (
            np.asarray(R, np.float64) @ beta[ix]
            + noise / np.sqrt(n[ix] if per_variant else n)
        )
    return bhat


def simulate_sumstats_pair(
    beta_a: np.ndarray,
    beta_b: np.ndarray,
    blocks: Sequence[Block],
    n,
    noise_correlation: Optional[float] = None,
    seed: Union[int, np.random.Generator, None] = 0,
    *,
    overlap: Optional[float] = None,
):
    """Two GWAS marginal-effect vectors with correlated sampling noise.

    ``noise_correlation`` is rho in ``Cov(noise_a, noise_b) = rho R/n``
    (default 0), or rho D R D for per-variant N. It is not the fraction
    of shared participants: for equal-size studies with independent
    standardized residuals, even complete overlap gives rho=0. Under
    the conditional RSS model, rho is overlap fraction times residual
    correlation. ``overlap`` is a deprecated spelling for the historical
    noise correlation; it warns rather than silently reinterpreting old
    calls. Returns ``(bhat_a, bhat_b)``.
    """
    if overlap is not None:
        if noise_correlation is not None:
            raise ValueError("pass noise_correlation, not both noise_correlation and overlap")
        warnings.warn("overlap means noise correlation, not participant overlap; use "
                      "noise_correlation explicitly", FutureWarning, stacklevel=2)
        noise_correlation = overlap
    rho = 0.0 if noise_correlation is None else float(noise_correlation)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("noise_correlation must be in [-1, 1]")
    blocks = _as_blocks(blocks)
    rng = np.random.default_rng(seed)
    m = _block_grid(blocks)
    beta_a = _effects_vector(beta_a, m)
    beta_b = _effects_vector(beta_b, m)
    n = _sample_size(n, m)
    bhat_a = np.empty(m)
    bhat_b = np.empty(m)
    per_variant = np.ndim(n) > 0
    scale = np.sqrt(1.0 - rho**2)
    for R, ix in blocks:
        Rf = np.asarray(R, np.float64)
        chol = _chol(Rf)
        z1 = rng.standard_normal(len(ix))
        z2 = rng.standard_normal(len(ix))
        rootn = np.sqrt(n[ix] if per_variant else n)
        bhat_a[ix] = Rf @ beta_a[ix] + (chol @ z1) / rootn
        bhat_b[ix] = Rf @ beta_b[ix] + (chol @ (rho * z1 + scale * z2)) / rootn
    return bhat_a, bhat_b


def gwas_scan(
    G: np.ndarray, y: np.ndarray
) -> dict:
    """Marginal GWAS scan of a phenotype over genotype columns.

    Each variant uses its called samples (missing = NaN or negative).
    Both genotype and phenotype are standardized within that subset, so
    beta is Pearson r and ``se = sqrt((1-r^2)/(n_called-2))``. ``z`` is
    the OLS t statistic, retained under its historical name; ``p`` uses
    the large-sample normal approximation ``erfc(|z|/sqrt(2))``, not an
    exact finite-sample t test. Perfect associations give signed infinity
    and p=0. Untestable variants (<3 calls or a constant genotype/called
    phenotype) have NaN outputs. The phenotype must be finite. Returns
    ``{"beta", "se", "z", "p"}`` on the standardized scale.
    """
    Gd = np.asarray(G, dtype=np.float64)
    y = np.asarray(y, dtype=float)
    if Gd.ndim != 2 or y.ndim != 1 or Gd.shape[0] != y.size:
        raise ValueError("G and y must have the same number of samples")
    if not np.isfinite(y).all() or y.size < 3 or y.std() == 0 or np.isinf(Gd).any():
        raise ValueError("need a finite nonconstant phenotype, at least 3 samples, and no infinite genotypes")
    n, m = Gd.shape
    miss = (Gd < 0) | np.isnan(Gd)
    ok = ~miss
    cen = np.where(miss, 0.0, Gd)
    cnt = ok.sum(axis=0)
    cen -= cen.sum(axis=0) / np.maximum(cnt, 1)
    cen[miss] = 0.0
    ss_g = np.einsum("ij,ij->j", cen, cen)
    # A global shift improves stability without changing any subset's OLS.
    yc = y - y.mean()
    sum_y = np.einsum("ij,i->j", ok, yc)
    ss_y = np.einsum("ij,i->j", ok, yc * yc) - sum_y**2 / np.maximum(cnt, 1)
    num = cen.T @ yc
    valid = (cnt >= 3) & (ss_g > 0) & (ss_y > 0)
    r, se, z = (np.full(m, np.nan) for _ in range(3))
    r[valid] = np.clip(num[valid] / np.sqrt(ss_g[valid] * ss_y[valid]), -1.0, 1.0)
    se[valid] = np.sqrt(np.maximum(1 - r[valid]**2, 0) / (cnt[valid] - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        z[valid] = r[valid] / se[valid]
    p = np.array([math.erfc(abs(v) / math.sqrt(2.0)) for v in z])
    return {"beta": r, "se": se, "z": z, "p": p}


def shake_ld(
    blocks: Sequence[Block],
    n_ref: Optional[int],
    seed: Union[int, np.random.Generator, None] = 0,
):
    """Reference-panel LD: the truth, or a finite noisy panel of it.

    ``n_ref=None`` returns the blocks symmetrised (the exact population
    LD). Otherwise each block draws a Wishart panel ``X = Z chol(R)'``
    with ``n_ref`` rows and returns its sample correlation, with the
    diagonal reset to 1 -- exactly the mismatch a finite reference panel
    hands an LD-based method. Returns a new ``(R, ix)`` list.
    """
    blocks = _as_blocks(blocks)
    if n_ref is not None and (isinstance(n_ref, (bool, np.bool_))
                              or not isinstance(n_ref, (int, np.integer)) or n_ref < 2):
        raise ValueError("n_ref must be an integer at least 2")
    rng = np.random.default_rng(seed)
    out = []
    for R, ix in blocks:
        R = np.asarray(R, dtype=np.float64)
        factor = _chol(R)
        if n_ref is not None:
            Z = rng.standard_normal((int(n_ref), len(ix)))
            X = Z @ factor.T
            Xc = X - X.mean(0)
            s = Xc.std(0)
            s[s == 0] = 1.0
            Xs = Xc / s
            R = Xs.T @ Xs / n_ref
            np.fill_diagonal(R, 1.0)
        R = (R + R.T) / 2.0
        out.append((R, ix))
    return out
