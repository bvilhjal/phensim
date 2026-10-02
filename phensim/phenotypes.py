"""Phenotype simulators on top of genotype data.

All quantitative-trait simulators draw the infinitesimal component
through the empirical GRM's eigendecomposition (u ~ N(0, sigma2 K)), so
the data-generating covariance matches what a mixed model will fit.
"""

from __future__ import annotations

import warnings
from statistics import NormalDist
from typing import Optional, Union

import numpy as np

from phensim._common import norm_ppf

__all__ = [
    "simulate_trait",
    "simulate_binary_trait",
    "simulate_confounded_trait",
    "simulate_gxe_trait",
    "simulate_correlated_traits",
    "ascertain_case_control",
    "n_eff_case_control",
    "h2_liability",
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


# --------------------------------------------------------------------------- #
# Ascertainment and case/control bookkeeping
# --------------------------------------------------------------------------- #
def ascertain_case_control(
    trait: Union[dict, np.ndarray],
    n_cases: int,
    n_controls: int,
    seed: Union[int, np.random.Generator, None] = 5,
) -> dict:
    """Sample exact case/control counts from a liability-threshold trait.

    ``trait`` is the dict from :func:`simulate_binary_trait` (or any dict
    with ``case_control`` and ``liability``). Cases and controls are drawn
    without replacement to the *exact* requested counts -- the
    all-cases-plus-k-controls register design and the balanced cohort, the
    two ascertainment schemes every case/control method faces. Returns
    ``{"index", "case_control", "liability"}`` aligned to the sampled
    participants; raises ``ValueError`` when the population holds fewer
    cases or controls than requested.
    """
    if isinstance(trait, dict):
        cc = np.asarray(trait["case_control"]).astype(int)
        liab = np.asarray(trait["liability"], dtype=float)
    else:
        cc = np.asarray(trait).astype(int)
        liab = None
    if cc.ndim != 1:
        raise ValueError("trait must be a simulation dict or a 1-D case/control vector")
    cases = np.flatnonzero(cc == 1)
    controls = np.flatnonzero(cc == 0)
    if cases.size < n_cases:
        raise ValueError(
            f"population has {cases.size} cases, {n_cases} requested; "
            "raise the prevalence or the population size"
        )
    if controls.size < n_controls:
        raise ValueError(
            f"population has {controls.size} controls, {n_controls} requested"
        )
    rng = np.random.default_rng(seed)
    index = np.concatenate(
        [
            rng.choice(cases, size=int(n_cases), replace=False),
            rng.choice(controls, size=int(n_controls), replace=False),
        ]
    )
    out = {
        "index": index,
        "case_control": cc[index],
    }
    if liab is not None and liab.size == cc.size:
        out["liability"] = liab[index]
    return out


def n_eff_case_control(n_case, n_control):
    """Effective sample size of a case/control GWAS: ``4/(1/N_case + 1/N_control)``.

    Equals total N for a balanced study and tends to ``4*N_case`` with
    many more controls than cases. Accepts scalars or arrays.
    """
    n_case = np.asarray(n_case, dtype=float)
    n_control = np.asarray(n_control, dtype=float)
    if np.any(n_case <= 0) or np.any(n_control <= 0):
        raise ValueError("n_case and n_control must be positive")
    n = 4.0 / (1.0 / n_case + 1.0 / n_control)
    return float(n) if n.ndim == 0 else n


def h2_liability(h2_observed, prevalence, *, prop_cases=None):
    """Convert observed-scale SNP h² to the liability scale (Lee et al. 2011).

    For population prevalence ``K``, study case fraction ``P``, threshold
    ``t = Phi^-1(1-K)``, and ``z = phi(t)``::

        h²_liab = h²_obs * [K(1-K)]² / (z² * P(1-P))

    ``prop_cases=None`` defaults ``P`` to ``K`` **with a warning**: that
    default is correct only when the study sample mirrors the population
    case fraction. A balanced case/control GWAS of a 1% trait must pass
    ``prop_cases=0.5``; the silent default overstates h² there by
    ``P(1-P)/(K(1-K))`` (about 25x).
    """
    K = float(prevalence)
    if not 0.0 < K < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    if prop_cases is None:
        warnings.warn(
            "h2_liability(prop_cases=None) assumes the study sample mirrors the "
            "population case fraction (P=K). For an ascertained case/control "
            "study pass the GWAS case fraction explicitly (balanced: "
            "prop_cases=0.5); the silent default overstates liability h2 by "
            "P(1-P)/(K(1-K)) there.", UserWarning, stacklevel=2)
    P = K if prop_cases is None else float(prop_cases)
    if not 0.0 < P < 1.0:
        raise ValueError("prop_cases must be in (0, 1)")
    nd = NormalDist()
    # Avoid forming 1-K: Phi^-1(1-K) = -Phi^-1(K), including tiny K.
    t = -nd.inv_cdf(K)
    z = nd.pdf(t)
    factor = (K * (1.0 - K)) ** 2 / (z * z * P * (1.0 - P))
    h2 = np.asarray(h2_observed, dtype=float) * factor
    return float(h2) if h2.ndim == 0 else h2
