"""Phenotype simulators on top of genotype data.

All quantitative-trait simulators draw the infinitesimal component
through the empirical GRM's eigendecomposition (u ~ N(0, sigma2 K)), so
the data-generating covariance matches what a mixed model will fit.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from phensim._common import norm_ppf

__all__ = [
    "simulate_trait",
    "simulate_binary_trait",
    "simulate_confounded_trait",
    "simulate_gxe_trait",
    "simulate_correlated_traits",
]


def _grm(G: np.ndarray) -> np.ndarray:
    """Additive GRM, standardized columns, mean diagonal 1."""
    from phensim.kinship import grm

    return grm(G)


def _standardized(x: np.ndarray) -> np.ndarray:
    x = (x - x.mean()) / x.std()
    return x


def simulate_trait(
    G: np.ndarray,
    h2: float = 0.5,
    n_causal: int = 20,
    architecture: str = "mixed",
    effect_dist: str = "normal",
    causal: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    seed: Union[int, np.random.Generator, None] = 1,
) -> dict:
    """Quantitative trait with a model-consistent genetic component.

    ``architecture`` splits the target heritability ``h2``:

    - ``'mixed'``: half infinitesimal (u ~ N(0, (h2/2) K) through K's
      eigendecomposition), half from ``n_causal`` QTLs;
    - ``'infinitesimal'``: all h2 through the kinship;
    - ``'qtl'``: all h2 at the causal variants, no background.

    ``effect_dist`` is ``'normal'`` (random effect sizes) or ``'equal'``
    (same absolute effect per causal variant -- deterministic per-locus
    power). Returns ``{"y", "u", "liability", "causal", "effects"}``
    with ``y`` standardized.
    """
    if architecture not in ("mixed", "infinitesimal", "qtl"):
        raise ValueError(f"unknown architecture {architecture!r}")
    if not 0.0 <= h2 <= 1.0:
        raise ValueError("h2 must be in [0, 1]")
    rng = np.random.default_rng(seed)
    Gd = np.asarray(G, dtype=np.float64)
    n, m = Gd.shape

    h2_bg = {"mixed": h2 / 2, "infinitesimal": h2, "qtl": 0.0}[architecture]
    h2_qtl = h2 - h2_bg

    u = np.zeros(n)
    if h2_bg > 0:
        if K is None:
            K = _grm(G)
        lam, U = np.linalg.eigh(K)
        lam = np.maximum(lam, 0.0)
        u = U @ (np.sqrt(lam * h2_bg) * rng.standard_normal(n))

    if causal is None:
        causal = rng.choice(m, size=min(n_causal, m), replace=False)
    causal = np.asarray(causal)
    if h2_qtl > 0 and causal.size:
        if effect_dist == "equal":
            effects = np.sign(rng.standard_normal(causal.size))
        else:
            effects = rng.normal(0, 1, causal.size)
        Z = Gd[:, causal]
        Z = Z - Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True)
        Z = Z / np.where(sd > 0, sd, 1.0)
        q = Z @ effects
        q = q * (np.sqrt(h2_qtl) / q.std()) if q.std() > 0 else q
    else:
        effects = np.zeros(causal.size)
        q = np.zeros(n)

    e = rng.standard_normal(n) * np.sqrt(max(1.0 - h2, 0.0))
    liability = u + q + e
    return {
        "y": _standardized(liability),
        "liability": liability,
        "u": u,
        "causal": causal,
        "effects": effects,
    }


def simulate_binary_trait(
    G: np.ndarray,
    prevalence: float = 0.05,
    **trait_kwargs,
) -> dict:
    """Liability-threshold case/control trait.

    Simulates a quantitative liability via :func:`simulate_trait`, then
    thresholds at the ``prevalence`` quantile. Returns the trait dict
    with ``y`` binary, ``liability`` and ``case_control`` added.
    """
    tr = simulate_trait(G, **trait_kwargs)
    thresh = norm_ppf(1.0 - prevalence)
    cases = tr["liability"] > thresh
    tr["case_control"] = cases.astype(np.int8)
    tr["y"] = cases.astype(np.float64)
    return tr


def simulate_confounded_trait(
    G: np.ndarray,
    confounding_strength: float = 0.6,
    h2: float = 0.5,
    n_causal: int = 10,
    K: Optional[np.ndarray] = None,
    seed: Union[int, np.random.Generator, None] = 2,
) -> dict:
    """Trait driven partly by population structure.

    A share ``confounding_strength`` of the phenotype variance rides the
    leading eigenvector of the kinship (a proxy for population
    structure / batch effects that a mixed model should absorb), with a
    regular QTL architecture underneath. The canonical scenario for
    testing genomic-control correction.
    """
    rng = np.random.default_rng(seed)
    if K is None:
        K = _grm(G)
    lam, U = np.linalg.eigh(K)
    lead = (U[:, -1] - U[:, -1].mean()) / U[:, -1].std()
    tr = simulate_trait(
        G,
        h2=h2,
        n_causal=n_causal,
        K=K,
        seed=int(rng.integers(1, 2**31 - 1)),
    )
    s = confounding_strength
    liability = np.sqrt(s) * lead + np.sqrt(1 - s) * tr["liability"]
    return {
        "y": _standardized(liability),
        "liability": liability,
        "u": tr["u"],
        "causal": tr["causal"],
        "effects": tr["effects"],
    }


def simulate_gxe_trait(
    G: np.ndarray,
    E: Optional[np.ndarray] = None,
    h2: float = 0.5,
    interaction_h2: float = 0.2,
    n_causal: int = 5,
    seed: Union[int, np.random.Generator, None] = 3,
) -> dict:
    """Trait with genotype-environment interaction effects.

    ``interaction_h2`` of the phenotypic variance comes from
    ``g * E`` interactions at ``n_causal`` loci; ``E`` is a standard
    normal environment vector (drawn if not given).
    """
    rng = np.random.default_rng(seed)
    Gd = np.asarray(G, dtype=np.float64)
    n, m = Gd.shape
    if E is None:
        E = rng.standard_normal(n)
    E = (E - E.mean()) / E.std()

    base = simulate_trait(G, h2=h2 - interaction_h2, n_causal=n_causal, seed=seed)
    causal = base["causal"]
    Z = Gd[:, causal]
    Z = Z - Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    Z = Z / np.where(sd > 0, sd, 1.0)
    inter = (Z * E[:, None]) @ np.sign(rng.standard_normal(causal.size))
    inter = inter * (np.sqrt(interaction_h2) / inter.std())
    liability = base["liability"] + inter
    liability = liability / liability.std()
    return {
        "y": (liability - liability.mean()) / liability.std(),
        "liability": liability,
        "u": base["u"],
        "causal": causal,
        "effects": base["effects"],
        "environment": E,
    }


def simulate_correlated_traits(
    G: np.ndarray,
    h2_a: float = 0.5,
    h2_b: float = 0.5,
    rg: float = 0.6,
    n_causal: int = 20,
    seed: Union[int, np.random.Generator, None] = 4,
) -> dict:
    """Two traits with a target genetic correlation ``rg``.

    The shared infinitesimal component carries the correlation: trait A
    uses effects ``xi``, trait B uses ``rg * xi + sqrt(1 - rg^2) * zeta``
    on the same kinship eigenbasis, plus independent residuals and
    independent QTL blocks.
    """
    if not -1.0 <= rg <= 1.0:
        raise ValueError("rg must be in [-1, 1]")
    rng = np.random.default_rng(seed)
    K = _grm(G)
    lam, U = np.linalg.eigh(K)
    lam = np.maximum(lam, 0.0)
    xi = rng.standard_normal(G.shape[0])
    zeta = rng.standard_normal(G.shape[0])
    shared = U @ (np.sqrt(lam) * xi)
    idio = U @ (np.sqrt(lam) * zeta)
    for v in (shared, idio):
        v /= v.std()

    n, m = G.shape
    causal = rng.choice(m, size=min(n_causal, m), replace=False)
    Za = _standardize_cols(G, causal)
    Zb = _standardize_cols(G, causal)
    qa = Za @ rng.normal(size=causal.size)
    qb = Zb @ rng.normal(size=causal.size)

    g_a = np.sqrt(h2_a * 0.5) * shared + np.sqrt(h2_a * 0.5) * (qa / qa.std())
    g_b = (
        np.sqrt(h2_b * 0.5) * (rg * shared + np.sqrt(1 - rg**2) * idio)
        + np.sqrt(h2_b * 0.5) * (qb / qb.std())
    )
    e_a = rng.standard_normal(n) * np.sqrt(1 - h2_a)
    e_b = rng.standard_normal(n) * np.sqrt(1 - h2_b)
    ya = g_a + e_a
    yb = g_b + e_b
    return {
        "y_a": _standardized(ya),
        "y_b": _standardized(yb),
        "u_a": shared,
        "u_b": rg * shared + np.sqrt(1 - rg**2) * idio,
        "causal": causal,
    }


def _standardize_cols(G: np.ndarray, idx: np.ndarray) -> np.ndarray:
    Z = np.asarray(G, dtype=np.float64)[:, idx]
    Z = Z - Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    return Z / np.where(sd > 0, sd, 1.0)
