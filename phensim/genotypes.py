"""Genotype simulators: independent, structured, haplotype-block and
coalescent backends."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = [
    "simulate_independent",
    "simulate_population_structure",
    "simulate_haplotype_blocks",
    "simulate_coalescent",
    "simulate_by_mutation_rate",
    "resolve_backend",
]


def resolve_backend(backend: str = "auto") -> str:
    """Pick a coalescent backend: 'numba', 'msprime' or 'auto'."""
    if backend not in ("auto", "numba", "msprime"):
        raise ValueError("backend must be 'auto', 'numba' or 'msprime'")
    if backend != "auto":
        return backend
    from phensim._numba import HAVE_NUMBA

    if HAVE_NUMBA:
        return "numba"
    try:
        import msprime  # noqa: F401

        return "msprime"
    except ImportError:
        return "numba"  # pure-Python fallback (correct, just slower)


def _draw_freqs(
    m: int, freq_dist: str, rng: np.random.Generator
) -> np.ndarray:
    """Per-site minor allele frequencies under a named SFS shape."""
    if freq_dist == "beta":
        f = rng.beta(0.35, 1.0, m) * 0.5
    elif freq_dist == "uniform":
        f = rng.uniform(0.01, 0.5, m)
    elif freq_dist == "rare":
        f = rng.exponential(0.02, m).clip(1e-4, 0.5)
    elif freq_dist == "common":
        f = rng.uniform(0.1, 0.5, m)
    else:
        raise ValueError(f"unknown freq_dist {freq_dist!r}")
    return f


def simulate_independent(
    n: int,
    m: int,
    maf: float = 0.3,
    freq_dist: str = "fixed",
    seed: Union[int, np.random.Generator, None] = 0,
) -> np.ndarray:
    """(n, m) int8 dosages with independent SNPs.

    ``freq_dist='fixed'`` uses a single ``maf`` for every site; 'beta',
    'uniform', 'rare' and 'common' draw per-site minor allele frequencies
    from the corresponding allele-frequency spectrum shape.
    """
    rng = np.random.default_rng(seed)
    if freq_dist == "fixed":
        f = np.full(m, float(maf))
    else:
        f = _draw_freqs(m, freq_dist, rng)
    flip = rng.random(m) < 0.5
    p = np.where(flip, 1 - f, f)
    return rng.binomial(2, p[None, :], size=(n, m)).astype(np.int8)


def simulate_population_structure(
    n: int,
    m: int,
    n_pops: int = 3,
    fst: float = 0.1,
    maf: float = 0.3,
    seed: Union[int, np.random.Generator, None] = 0,
):
    """Independent SNPs with diverged per-population allele frequencies.

    Returns ``(G, pop_labels)``; between-population frequency variance is
    ``fst * p (1 - p)`` per site. Cheap structure for LMM-confounding
    studies; use :func:`simulate_coalescent` when LD realism matters.
    """
    rng = np.random.default_rng(seed)
    base = np.clip(maf + rng.normal(0, 0.05, m), 0.05, 0.95)
    spread = np.sqrt(fst * base * (1 - base))
    freqs = np.clip(
        base[:, None] + rng.normal(0, 1, (m, n_pops)) * spread[:, None], 0.01, 0.99
    )
    labels = rng.integers(0, n_pops, n)
    p = freqs[:, labels].T
    return rng.binomial(2, p).astype(np.int8), labels


def simulate_haplotype_blocks(
    n: int,
    m: int,
    block_size: int = 100,
    n_founders: int = 20,
    mutation_rate: float = 0.001,
    seed: Union[int, np.random.Generator, None] = 0,
) -> np.ndarray:
    """LD-structured genotypes from founder haplotype copying.

    Each block of ``block_size`` SNPs is founded by ``n_founders``
    haplotypes; descendants inherit a founder haplotype with per-site
    mutation flips. Fast (one pass, no coalescent) with genuine
    haplotypic LD within blocks and sharp decay between blocks.
    """
    rng = np.random.default_rng(seed)
    n_blocks = m // block_size
    G = np.empty((n, n_blocks * block_size), dtype=np.int8)
    for b in range(n_blocks):
        founders = rng.binomial(1, 0.3, size=(n_founders, block_size))
        parents = rng.integers(0, n_founders, size=(n, 2))
        hap = founders[parents]  # (n, 2, block_size)
        flip = rng.random(hap.shape) < mutation_rate
        hap = np.where(flip, 1 - hap, hap)
        G[:, b * block_size : (b + 1) * block_size] = (
            hap[:, 0, :] + hap[:, 1, :]
        ).astype(np.int8)
    return G


def _coalescent_dosages(n, seq_len, *, recomb_rate, mut_rate, Ne, seed, backend):
    """One coalescent replicate -> ``(dos, af)`` via the chosen backend."""
    if backend == "numba":
        from phensim._coalescent import simulate_dosages

        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        dos, _pos, af = simulate_dosages(
            n, seq_len, recomb_rate=recomb_rate, mut_rate=mut_rate, Ne=Ne, seed=seed
        )
        return dos, af
    try:
        import msprime
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the msprime backend needs msprime (pip install phensim[msprime])"
        ) from e
    ms_seed = None if seed is None else int(seed)
    ts = msprime.sim_ancestry(
        samples=n,
        ploidy=2,
        population_size=Ne,
        recombination_rate=recomb_rate,
        sequence_length=int(seq_len),
        random_seed=ms_seed,
    )
    mts = msprime.sim_mutations(
        ts, rate=mut_rate, random_seed=ms_seed, model=msprime.BinaryMutationModel()
    )
    H = mts.genotype_matrix()  # (sites, 2n), 0/1
    dos = (H[:, 0::2] + H[:, 1::2]).T  # (n, sites), 0/1/2
    af = dos.mean(axis=0) / 2.0
    return dos.astype(np.int8), af


def simulate_coalescent(
    n: int,
    m: int,
    block_size: int = 200,
    *,
    Ne: int = 10_000,
    recomb_rate: float = 1e-8,
    mut_rate: float = 1e-8,
    min_maf: float = 0.01,
    seed: Union[int, None] = None,
    backend: str = "auto",
):
    """Coalescent genotypes with recombination-driven LD.

    Human-like defaults (Ne = 10,000; recombination and mutation rates
    1e-8/bp/generation). The sequence length grows until at least ``m``
    common SNPs (MAF > ``min_maf``) exist; the first ``m`` are kept and
    cut into contiguous blocks of ``block_size``.

    ``backend``: ``'numba'`` (built-in JIT coalescent,
    :mod:`phensim._coalescent`), ``'msprime'`` (the msprime C library) or
    ``'auto'`` (built-in when Numba is available, else msprime, else the
    pure-Python built-in). Returns ``(G, blocks)`` with ``G`` int8
    ``(n, m')`` sample-major dosages and ``blocks`` contiguous index
    arrays; ``m'`` is ``m`` rounded down to a multiple of ``block_size``.
    """
    backend = resolve_backend(backend)
    rng = np.random.default_rng(seed)
    seq_len = max(1e6, m / 1200 * 1e6)  # ~1200 common SNPs per Mb to start
    G = None
    for _ in range(7):
        rep_seed = int(rng.integers(1, 2**31 - 1))
        dos, af = _coalescent_dosages(
            n,
            seq_len,
            recomb_rate=recomb_rate,
            mut_rate=mut_rate,
            Ne=Ne,
            seed=rep_seed,
            backend=backend,
        )
        dos = dos[:, (af > min_maf) & (af < 1 - min_maf)]
        if dos.shape[1] >= m:
            G = dos
            break
        seq_len *= 1.8
    if G is None or G.shape[1] < m:
        raise RuntimeError(
            "coalescent simulation produced too few common SNPs; "
            "increase sequence length / Ne"
        )
    n_blocks = m // block_size
    m2 = n_blocks * block_size
    G = np.ascontiguousarray(G[:, :m2].astype(np.int8))
    blocks = [
        np.arange(i * block_size, (i + 1) * block_size) for i in range(n_blocks)
    ]
    return G, blocks


def simulate_by_mutation_rate(
    n: int,
    seq_len: float,
    *,
    recomb_rate: float = 1e-8,
    mut_rate: float = 1e-8,
    Ne: int = 10_000,
    min_maf: float = 0.01,
    seed: Union[int, None] = None,
    backend: str = "auto",
) -> np.ndarray:
    """Coalescent genotypes on a *fixed* segment; density set by mutation.

    Unlike :func:`simulate_coalescent` (which grows the segment to hit a
    SNP-count target, coupling LD extent to SNP count), the segment
    length and recombination rate here fix the LD structure and the
    mutation rate controls how many variants sit on it. With a fixed
    seed the genealogy is identical across mutation rates, so raising
    the rate is the same chromosome with more discovered variants.
    Returns ``G`` int8 ``(n, k)``; ``k`` emerges from the rate. Columns
    are in physical order, so contiguous slices are contiguous LD.
    """
    backend = resolve_backend(backend)
    dos, af = _coalescent_dosages(
        n,
        seq_len,
        recomb_rate=recomb_rate,
        mut_rate=mut_rate,
        Ne=Ne,
        seed=seed,
        backend=backend,
    )
    dos = dos[:, (af > min_maf) & (af < 1 - min_maf)]
    return np.ascontiguousarray(dos.astype(np.int8))
