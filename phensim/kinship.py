"""Additive genomic relationship matrix."""

from __future__ import annotations

import numpy as np

__all__ = ["grm"]


def grm(G: np.ndarray, scale: bool = True) -> np.ndarray:
    """GRM with Yang-2010 called-only standardization, mean diagonal 1.

    ``G`` is a sample-major (n, m) dosage matrix; per variant the
    column is standardized over called genotypes (missing = NaN or -1)
    and no-calls contribute zero to every accumulator.
    """
    Gd = np.asarray(G, dtype=np.float64)
    Z = np.where((Gd < 0) | np.isnan(Gd), 0.0, Gd)
    ok = ~((Gd < 0) | np.isnan(Gd))
    cnt = ok.sum(axis=0)
    mean = np.where(cnt > 0, Z.sum(axis=0) / np.maximum(cnt, 1), 0.0)
    cen = np.where(ok, Gd - mean, 0.0)
    var = (cen * cen).sum(axis=0) / np.maximum(cnt, 1)
    std = np.sqrt(var)
    Zs = cen / np.where(std > 0, std, 1.0)
    K = (Zs @ Zs.T) / Gd.shape[1]
    if scale:
        n = K.shape[0]
        off = (K.sum() - np.trace(K)) / (n * (n - 1))
        K = K - off
        K = K / (np.trace(K) / n)
    return K
