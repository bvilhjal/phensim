"""Minimal PLINK 1 binary writer for simulated data."""

from __future__ import annotations

import numpy as np

__all__ = ["write_plink"]

_BED_MAGIC = b"\x6c\x1b\x01"


def write_plink(
    G: np.ndarray,
    prefix: str,
    chromosome=None,
    position=None,
    sample_ids=None,
) -> None:
    """Write sample-major dosages as a SNP-major .bed with .bim/.fam.

    Dosages count allele 2 (G in the BIM). Negative values and NaN are
    missing. Called values must be 0, 1 or 2; fractional dosages cannot
    be represented by PLINK 1 binary hard calls.
    """
    G = np.asarray(G)
    if G.ndim != 2 or not all(G.shape):
        raise ValueError("G must be a non-empty sample-major matrix")
    n, m = G.shape
    # Validate before opening any output file, one column at a time.
    for j in range(m):
        col = G[:, j]
        missing = (col < 0) | np.isnan(col)
        if np.any(~np.isfinite(col) & ~np.isnan(col)) or np.any(~missing & ~np.isin(col, [0, 1, 2])):
            raise ValueError("called genotypes must be 0, 1 or 2 (missing: negative or NaN)")
    chromosome = (
        np.ones(m, dtype=int) if chromosome is None else np.asarray(chromosome)
    )
    position = (
        np.arange(1, m + 1) if position is None else np.asarray(position)
    )
    sample_ids = (
        [f"S{i}" for i in range(n)] if sample_ids is None else list(sample_ids)
    )
    if chromosome.shape != (m,) or position.shape != (m,) or len(sample_ids) != n:
        raise ValueError("chromosome/position must match variants and sample_ids must match samples")
    with open(f"{prefix}.fam", "w") as fh:
        for s in sample_ids:
            fh.write(f"0 {s} 0 0 0 -9\n")
    with open(f"{prefix}.bim", "w") as fh:
        for c, p in zip(chromosome, position):
            fh.write(f"{c} sim_{c}_{p} 0 {p} A G\n")
    nbytes_row = (n + 3) // 4
    with open(f"{prefix}.bed", "wb") as fh:
        fh.write(_BED_MAGIC)
        for j in range(m):
            col = G[:, j]
            missing = (col < 0) | np.isnan(col)
            # BED: 00=A/A, 01=missing, 10=A/G, 11=G/G.
            g = np.array([0, 2, 3], dtype=np.uint8)[np.where(missing, 0, col).astype(np.intp)]
            g[missing] = 1
            inter = np.zeros(8 * nbytes_row, dtype=np.uint8)
            inter[0::2][:n] = g & 1
            inter[1::2][:n] = (g >> 1) & 1
            fh.write(np.packbits(inter, bitorder="little").tobytes())
