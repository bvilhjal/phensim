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

    Missing calls (negative values) are encoded as PLINK no-calls.
    """
    G = np.asarray(G)
    n, m = G.shape
    chromosome = (
        np.ones(m, dtype=int) if chromosome is None else np.asarray(chromosome)
    )
    position = (
        np.arange(m) if position is None else np.asarray(position)
    )
    sample_ids = (
        [f"S{i}" for i in range(n)] if sample_ids is None else list(sample_ids)
    )
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
            g = G[:, j].astype(np.uint8).copy()
            g[G[:, j] < 0] = 3
            inter = np.zeros(8 * nbytes_row, dtype=np.uint8)
            inter[0::2][:n] = g & 1
            inter[1::2][:n] = (g >> 1) & 1
            fh.write(np.packbits(inter, bitorder="little").tobytes())
