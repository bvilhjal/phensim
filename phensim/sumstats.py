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
    blocks = list(blocks)
    for R, ix in blocks:
        if np.asarray(R).shape != (len(ix), len(ix)):
            raise ValueError("each block must be (R, ix) with R len(ix) x len(ix)")
    return blocks


def _chol(R: np.ndarray) -> np.ndarray:
    """Cholesky factor of a correlation block, with tiny PSD repair."""
    R = np.asarray(R, dtype=np.float64)
    try:
        return np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        k = R.shape[0]
        return np.linalg.cholesky((R + R.T) / 2.0 + 1e-8 * np.eye(k))


def _block_grid(blocks: Sequence[Block]) -> int:
    return max(int(ix.max()) for _, ix in blocks) + 1


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
    blocks = _as_blocks(blocks)
    m = _block_grid(blocks)
    if architecture in ("sparse", "equal") and n_causal is None:
        raise ValueError(f"architecture {architecture!r} needs n_causal")
    if architecture == "maf" and maf is None:
        raise ValueError("architecture 'maf' needs per-variant maf")
    rng = np.random.default_rng(seed)
    beta = np.zeros(m)
    if architecture in ("sparse", "equal"):
        causal = rng.choice(m, size=min(int(n_causal), m), replace=False)
        if architecture == "equal":
            beta[causal] = np.sign(rng.standard_normal(causal.size))
        else:
            beta[causal] = rng.standard_normal(causal.size)
    elif architecture == "polygenic":
        beta = rng.standard_normal(m)
    else:
        f = np.asarray(maf, dtype=float)
        beta = rng.standard_normal(m) * (2.0 * f * (1.0 - f)) ** (alpha / 2.0)
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
    draw per block, in block order.
    """
    blocks = _as_blocks(blocks)
    beta = np.asarray(beta, dtype=float)
    rng = np.random.default_rng(seed)
    bhat = np.empty(_block_grid(blocks))
    per_variant = np.ndim(n) > 0
    for R, ix in blocks:
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
    overlap: float = 0.0,
    seed: Union[int, np.random.Generator, None] = 0,
):
    """Two GWAS marginal-effect vectors with correlated sampling noise.

    Sample overlap ``overlap`` (the fraction of participants shared
    between the two studies) induces ``Cov(bhat_a, bhat_b) = rho * R / n``
    block by block: per block, ``u_a = z1`` and
    ``u_b = overlap * z1 + sqrt(1 - overlap^2) * z2`` feed the same LD
    Cholesky factor. Returns ``(bhat_a, bhat_b)``.
    """
    if not -1.0 <= overlap <= 1.0:
        raise ValueError("overlap must be in [-1, 1]")
    blocks = _as_blocks(blocks)
    beta_a = np.asarray(beta_a, dtype=float)
    beta_b = np.asarray(beta_b, dtype=float)
    rng = np.random.default_rng(seed)
    m = _block_grid(blocks)
    bhat_a = np.empty(m)
    bhat_b = np.empty(m)
    per_variant = np.ndim(n) > 0
    scale = np.sqrt(np.maximum(1.0 - overlap**2, 0.0))
    for R, ix in blocks:
        chol = _chol(R)
        Rf = np.asarray(R, np.float64)
        z1 = rng.standard_normal(len(ix))
        z2 = rng.standard_normal(len(ix))
        rootn = np.sqrt(n[ix] if per_variant else n)
        bhat_a[ix] = Rf @ beta_a[ix] + (chol @ z1) / rootn
        bhat_b[ix] = Rf @ beta_b[ix] + (chol @ (overlap * z1 + scale * z2)) / rootn
    return bhat_a, bhat_b


def gwas_scan(
    G: np.ndarray, y: np.ndarray
) -> dict:
    """Marginal GWAS scan of a phenotype over genotype columns.

    Columns are standardized over called genotypes (missing = NaN or -1
    is mean-imputed to the standardized 0) and ``y`` is standardized, so
    the per-variant estimate is the correlation ``r`` with
    ``se = sqrt((1 - r^2) / (n_called - 2))``, ``z = r / se`` and a
    two-sided ``p`` from the exact chi2(1) survival function
    (``erfc(|z| / sqrt(2))`` -- no SciPy needed). Returns
    ``{"beta", "se", "z", "p"}`` on the standardized scale.
    """
    Gd = np.asarray(G, dtype=np.float64)
    y = np.asarray(y, dtype=float).ravel()
    if Gd.shape[0] != y.size:
        raise ValueError("G and y must have the same number of samples")
    n, m = Gd.shape
    miss = (Gd < 0) | np.isnan(Gd)
    ok = ~miss
    gf = np.where(miss, 0.0, Gd)
    cnt = ok.sum(axis=0)
    mean = np.where(cnt > 0, gf.sum(axis=0) / np.maximum(cnt, 1), 0.0)
    cen = np.where(ok, Gd - mean, 0.0)
    var = (cen * cen).sum(axis=0) / np.maximum(cnt, 1)
    Z = cen / np.where(np.sqrt(var) > 0, np.sqrt(var), 1.0)
    yc = (y - y.mean()) / y.std()
    num = Z.T @ yc
    denom = np.sqrt(np.maximum(cnt, 1) * n)
    r = np.clip(num / denom, -1.0, 1.0)
    se = np.sqrt(np.clip(1.0 - r * r, 0.0, None) / np.maximum(cnt - 2, 1))
    z = r / np.where(se > 0, se, 1.0)
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
    rng = np.random.default_rng(seed)
    out = []
    for R, ix in blocks:
        R = np.asarray(R, dtype=np.float64)
        if n_ref is not None:
            if n_ref < 2:
                raise ValueError("n_ref must be at least 2")
            Z = rng.standard_normal((int(n_ref), len(ix)))
            X = Z @ _chol(R).T
            Xc = X - X.mean(0)
            s = Xc.std(0)
            s[s == 0] = 1.0
            Xs = Xc / s
            R = Xs.T @ Xs / n_ref
            np.fill_diagonal(R, 1.0)
        R = (R + R.T) / 2.0
        out.append((R, ix))
    return out
