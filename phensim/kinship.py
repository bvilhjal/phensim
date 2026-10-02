"""Kinship estimators: additive GRM, IBS, LOCO and windowed splits."""

from __future__ import annotations

import numpy as np

__all__ = [
    "grm",
    "ibs_kinship",
    "loco_kinships",
    "windowed_kinships",
]


def _emmax_scale(K: np.ndarray) -> np.ndarray:
    """EMMAX scaling: mean off-diagonal 0, mean diagonal 1."""
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    off = (K.sum() - np.trace(K)) / (n * (n - 1))
    K = K - off
    return K / (np.trace(K) / n)


def _called_standardized(G: np.ndarray):
    """Per-variant Yang-2010 called-only standardized columns.

    Returns ``(Z, cnt)`` with ``Z`` float64 ``(n, m)`` (missing calls map
    to 0 in the standardized space) and ``cnt`` the per-variant called
    counts.
    """
    Gd = np.asarray(G, dtype=np.float64)
    miss = (Gd < 0) | np.isnan(Gd)
    ok = ~miss
    gf = np.where(miss, 0.0, Gd)
    cnt = ok.sum(axis=0)
    mean = np.where(cnt > 0, gf.sum(axis=0) / np.maximum(cnt, 1), 0.0)
    cen = np.where(ok, Gd - mean, 0.0)
    var = (cen * cen).sum(axis=0) / np.maximum(cnt, 1)
    std = np.sqrt(var)
    return cen / np.where(std > 0, std, 1.0), cnt


def grm(G: np.ndarray, scale: bool = True) -> np.ndarray:
    """GRM with Yang-2010 called-only standardization, mean diagonal 1.

    ``G`` is a sample-major (n, m) dosage matrix; per variant the
    column is standardized over called genotypes (missing = NaN or -1)
    and no-calls contribute zero to every accumulator.
    """
    Gd = np.asarray(G, dtype=np.float64)
    Zs, _cnt = _called_standardized(Gd)
    K = (Zs @ Zs.T) / Gd.shape[1]
    return _emmax_scale(K) if scale else K


def ibs_kinship(G: np.ndarray, scale: bool = True) -> np.ndarray:
    """Identity-by-state similarity: mean fraction of shared genotypes.

    Diploid-aware via one-hot GEMMs per genotype value (0/1/2 matched
    pairs count once; missing calls are skipped in the denominator per
    pair), so unrelated pairs sit near 5/9 and clones at 1.
    """
    Gd = np.asarray(G, dtype=np.float64)
    miss = (Gd < 0) | np.isnan(Gd)
    ok = (~miss).astype(np.float64)
    g64 = np.where(miss, 0.0, Gd)
    S = np.zeros((Gd.shape[0], Gd.shape[0]), dtype=np.float64)
    for a in (0, 1, 2):
        H = (g64 == a) * ok  # called genotypes only
        S += H @ H.T
    C = ok @ ok.T
    with np.errstate(invalid="ignore", divide="ignore"):
        K = np.where(C > 0, S / C, 0.0)
    return _emmax_scale(K) if scale else K


def loco_kinships(
    G: np.ndarray,
    chromosomes: np.ndarray,
    *,
    scale: bool = True,
) -> dict:
    """Leave-one-chromosome-out kinships by additive subtraction (exact).

    Because the additive GRM is a sum over globally standardized SNPs,
    the LOCO matrix for chromosome ``c`` is ``(m K - m_c K_c) / (m -
    m_c)`` with the same standardization throughout -- no
    re-standardization, hence exact. ``chromosomes`` is the per-variant
    chromosome label array. Returns ``{chrom: K_loco}``.
    """
    Gd = np.asarray(G, dtype=np.float64)
    chromosomes = np.asarray(chromosomes)
    n, m = Gd.shape
    if chromosomes.size != m:
        raise ValueError("chromosomes must have one entry per variant")
    Zs, _cnt = _called_standardized(Gd)
    chroms = np.unique(chromosomes)
    per_chrom = {c: np.zeros((n, n), dtype=np.float64) for c in chroms}
    for c in chroms:
        Zc = Zs[:, chromosomes == c]
        per_chrom[c] += Zc @ Zc.T
    K = sum(per_chrom.values())
    out = {}
    for c in chroms:
        mc = float((chromosomes == c).sum())
        Kloco = (K / m - per_chrom[c] / m) * (m / (m - mc))
        out[c] = _emmax_scale(Kloco) if scale else Kloco
    return out


def windowed_kinships(
    G: np.ndarray,
    window_size: int,
    jump_size: int,
    *,
    scale: bool = True,
):
    """Local (window) and global (rest) kinship pairs along the genome.

    Yields ``(window_index, K_local, K_global)`` for windows of
    ``window_size`` variants every ``jump_size`` variants; both
    accumulate the same globally standardized columns, so ``K_local +
    K_global`` reconstructs the full GRM up to rescaling.
    """
    Gd = np.asarray(G, dtype=np.float64)
    n, m = Gd.shape
    if window_size < 1 or jump_size < 1:
        raise ValueError("window_size and jump_size must be positive")
    Zs, _cnt = _called_standardized(Gd)
    windows = [
        (start, min(start + window_size, m)) for start in range(0, m, jump_size)
    ]
    K_parts = {w: np.zeros((n, n), dtype=np.float64) for w in windows}
    for (start, stop), acc in K_parts.items():
        Zw = Zs[:, start:stop]
        acc += Zw @ Zw.T
    Kfull = sum(K_parts.values())
    for wi, (start, stop) in enumerate(windows):
        span = stop - start
        Kloc = K_parts[(start, stop)] / span
        Krest = (Kfull / m - K_parts[(start, stop)] / m) * (m / (m - span))
        if scale:
            Kloc = _emmax_scale(Kloc)
            Krest = _emmax_scale(Krest)
        yield wi, Kloc, Krest
