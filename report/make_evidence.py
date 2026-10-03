"""Regenerate the measured evidence behind report/phensim_report.tex.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python report/make_evidence.py

Writes report/evidence.json, report/tables/*.tex and report/figures/*.pdf.
Every draw is seeded; timings are wall-clock medians on the recording
machine and indicative only (the load average is recorded with them).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import phensim
from phensim._coalescent import simulate_dosages

HERE = Path(__file__).resolve().parent
TABLES, FIGURES = HERE / "tables", HERE / "figures"
EVIDENCE = {}


def _fmt(mean, sd, digits=3):
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def _write_table(name, header, rows, spec):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")


def _timed(fn, reps=3):
    fn()  # warm (JIT, caches)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# --------------------------------------------------------------------------- #
# 1. Phenotype simulators: realized versus target quantities
# --------------------------------------------------------------------------- #
def phenotype_targets(G):
    reps = 40
    rows, ev = [], {}
    for arch in ("mixed", "infinitesimal", "qtl"):
        bg, qtl, gen = [], [], []
        for r in range(reps):
            tr = phensim.simulate_trait(G, h2=0.5, n_causal=20, architecture=arch, seed=r)
            v = tr["liability"].var()
            bg.append(tr["u"].var() / v)
            qtl.append(tr["q"].var() / v)
            gen.append((tr["u"] + tr["q"]).var() / v)
        target = {"mixed": (0.25, 0.25), "infinitesimal": (0.5, 0.0), "qtl": (0.0, 0.5)}[arch]
        ev[arch] = dict(background=np.mean(bg), qtl=np.mean(qtl), genetic=np.mean(gen))
        rows.append([f"\\texttt{{simulate\\_trait}} ({arch})",
                     f"{target[0]:.2f} / {target[1]:.2f} / 0.50",
                     f"{_fmt(np.mean(bg), np.std(bg), 2)} / {_fmt(np.mean(qtl), np.std(qtl), 2)} / "
                     f"{_fmt(np.mean(gen), np.std(gen), 2)}"])
    K = phensim.grm(G)
    free = [phensim.simulate_trait(G, h2=0.5, architecture="infinitesimal", seed=r)["u"].var()
            for r in range(reps)]
    dense = [phensim.simulate_trait(G, h2=0.5, architecture="infinitesimal", K=K, seed=r)["u"].var()
             for r in range(reps)]
    ev["background_var"] = dict(matrix_free=np.mean(free), supplied_K=np.mean(dense),
                                expected=0.5 * (1 - 1 / G.shape[0]))
    rows.append(["background variance, $K$ omitted / supplied",
                 f"{0.5 * (1 - 1 / G.shape[0]):.3f}",
                 f"{_fmt(np.mean(free), np.std(free))} / {_fmt(np.mean(dense), np.std(dense))}"])
    for prev in (0.05, 0.2):
        frac = [phensim.simulate_binary_trait(G, prevalence=prev, h2=0.5, seed=r)["y"].mean()
                for r in range(reps)]
        ev[f"prevalence_{prev}"] = np.mean(frac)
        rows.append(["\\texttt{simulate\\_binary\\_trait} prevalence", f"{prev:.2f}",
                     _fmt(np.mean(frac), np.std(frac))])
    for rg in (0.0, 0.5, 0.9):
        real = []
        for r in range(reps):
            t = phensim.simulate_correlated_traits(G, rg=rg, n_causal=20, seed=r)
            real.append(np.corrcoef(t["g_a"], t["g_b"])[0, 1])
        ev[f"rg_{rg}"] = np.mean(real)
        rows.append(["\\texttt{simulate\\_correlated\\_traits} $r_g$", f"{rg:.2f}",
                     _fmt(np.mean(real), np.std(real))])
    inter = []
    for r in range(reps):
        t = phensim.simulate_gxe_trait(G, h2=0.5, interaction_h2=0.2, seed=r)
        inter.append(t["interaction"].var() / t["liability"].var())
    ev["gxe_interaction"] = np.mean(inter)
    rows.append(["\\texttt{simulate\\_gxe\\_trait} interaction share", "0.20",
                 _fmt(np.mean(inter), np.std(inter))])
    _write_table("phenotypes", ["Simulator and quantity", "Target", f"Realized mean (SD), {reps} draws"],
                 rows, "lll")
    EVIDENCE["phenotypes"] = ev


# --------------------------------------------------------------------------- #
# 2. Summary-statistic layer
# --------------------------------------------------------------------------- #
def _ar1(k, rho):
    return rho ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))


def sumstats_targets():
    rows, ev = [], {}
    k = 100
    blocks = [(_ar1(k, 0.9), np.arange(k)), (_ar1(k, 0.6), np.arange(k, 2 * k))]
    R = np.zeros((2 * k, 2 * k))
    for B, ix in blocks:
        R[np.ix_(ix, ix)] = B
    worst = 0.0
    for arch, kw in (("sparse", dict(n_causal=10)), ("polygenic", {}),
                     ("maf", dict(maf=np.linspace(0.01, 0.5, 2 * k))), ("equal", dict(n_causal=10))):
        beta = phensim.simulate_effects(blocks, h2=0.3, architecture=arch, seed=1, **kw)
        worst = max(worst, abs(beta @ R @ beta - 0.3))
    ev["effects_h2_error"] = worst
    rows.append(["\\texttt{simulate\\_effects}: $|\\beta^\\top R\\beta - h^2|$, 4 architectures", "0",
                 f"{worst:.1e}"])
    beta = phensim.simulate_effects(blocks, h2=0.3, n_causal=10, seed=2)
    ld = phensim.prepare_blocks(blocks)
    n, draws = 10_000, 4000
    noise = np.array([phensim.simulate_sumstats(beta, ld, n, seed=s) - R @ beta for s in range(draws)])
    err = np.linalg.norm(np.cov(noise.T) * n - R) / np.linalg.norm(R)
    # Wishart: E||S - R||_F^2 = (tr R^2 + (tr R)^2) / draws.
    expected = np.sqrt((np.sum(R * R) + np.trace(R) ** 2) / draws) / np.linalg.norm(R)
    ev["sumstats_cov_rel_error"] = err
    ev["sumstats_cov_rel_error_expected"] = expected
    rows.append([f"\\texttt{{simulate\\_sumstats}}: $\\|n\\,\\widehat{{\\mathrm{{Cov}}}} - R\\|_F/\\|R\\|_F$, {draws} draws",
                 f"$\\approx {expected:.3f}$", f"{err:.3f}"])
    pairs = [phensim.simulate_sumstats_pair(beta, beta, ld, n, noise_correlation=0.4, seed=s)
             for s in range(draws)]
    ea = np.array([a - R @ beta for a, _ in pairs])
    eb = np.array([b - R @ beta for _, b in pairs])
    corr = np.mean([np.corrcoef(ea[:, j], eb[:, j])[0, 1] for j in range(2 * k)])
    ev["pair_noise_correlation"] = corr
    rows.append(["\\texttt{simulate\\_sumstats\\_pair}: noise correlation", "0.40", f"{corr:.3f}"])
    a, b = phensim.simulate_effects_pair(blocks, 0.4, 0.4, 0.8, n_causal=(40, 40), n_shared=20, seed=3)
    rg_draws = [phensim.genetic_correlation(*phensim.simulate_effects_pair(
        blocks, 0.4, 0.4, 0.8, n_causal=(40, 40), n_shared=20, seed=s), blocks) for s in range(400)]
    ev["effects_pair_rg"] = float(np.mean(rg_draws))
    rows.append(["\\texttt{simulate\\_effects\\_pair}: mean realized $r_g$ (40/40 causal, 20 shared, $\\rho=0.8$)",
                 "0.40", _fmt(np.mean(rg_draws), np.std(rg_draws))])
    for n_ref in (100, 500, 2500):
        panels = phensim.shake_ld(ld, n_ref, seed=4)
        off = ~np.eye(k, dtype=bool)
        dev = np.mean([np.sqrt(np.mean((P - B)[off] ** 2)) for (P, _), (B, _) in zip(panels, blocks)])
        theory = np.mean([np.sqrt(np.mean(((1 - B ** 2) ** 2)[off]) / n_ref) for B, _ in blocks])
        ev[f"shake_rmse_{n_ref}"] = dev
        rows.append([f"\\texttt{{shake\\_ld}}: off-diagonal RMSE, $n_{{\\mathrm{{ref}}}}={n_ref}$",
                     f"$\\approx{theory:.3f}$", f"{dev:.3f}"])
    rng = np.random.default_rng(5)
    G = phensim.simulate_ar1_blocks(3000, [50] * 40, maf=0.3, rho=0.5, seed=5)[0].astype(float)
    G[rng.random(G.shape) < 0.05] = np.nan
    lams, z2 = [], []
    for r in range(40):
        z = phensim.gwas_scan(G, np.random.default_rng(100 + r).standard_normal(3000))["z"]
        lams.append(np.median(z ** 2) / 0.4549364231195724)
        z2.append(np.mean(z ** 2))
    ev["gwas_lambda_gc"] = dict(mean=np.mean(lams), sd=np.std(lams), mean_z2=np.mean(z2))
    rows.append(["\\texttt{gwas\\_scan}: $\\lambda_{\\mathrm{GC}}$, 40 null traits, 5\\% missing calls", "1",
                 _fmt(np.mean(lams), np.std(lams))])
    _write_table("sumstats", ["Check", "Expected", "Realized"], rows, "lll")
    EVIDENCE["sumstats"] = ev


# --------------------------------------------------------------------------- #
# 3. Pedigree: Mendelian draws reproduce A
# --------------------------------------------------------------------------- #
def pedigree_check():
    ids, fa, mo = phensim.simulate_pedigree(n_founder_pairs=20, gens=3, remarry=0.2, seed=1)
    A = phensim.kinship_from_pedigree(ids, fa, mo)
    rng = np.random.default_rng(2)
    draws = np.array([phensim.mendelian_draw(ids, fa, mo, innovations=rng.standard_normal(len(ids)))[0]
                      for _ in range(2000)])
    C = np.cov(draws.T)
    d = np.diag(A)
    standardized = (C - A) / np.sqrt((np.outer(d, d) + A ** 2) / (len(draws) - 1))
    EVIDENCE["pedigree"] = dict(n=len(ids), draws=len(draws),
                                standardized_rms_error=float(np.sqrt(np.mean(standardized ** 2))),
                                max_abs_cov_error=float(np.abs(C - A).max()),
                                diag_inbreeding_max=float(np.diag(A).max() - 1),
                                relationship_values=sorted({round(float(x), 4) for x in np.unique(A)})[:8])


# --------------------------------------------------------------------------- #
# 4. Coalescent backends: built-in Hudson engine versus msprime
# --------------------------------------------------------------------------- #
def _summaries(G, pos, bins):
    af = G.mean(0) / 2
    out = {"sites": G.shape[1], "diversity": float((2 * af * (1 - af)).sum())}
    keep = (af >= 0.05) & (af <= 0.95)
    X, p = G[:, keep].astype(float), pos[keep]
    X = (X - X.mean(0)) / X.std(0)
    i, j = np.triu_indices(X.shape[1], 1)
    r2 = (X.T @ X / X.shape[0])[i, j] ** 2
    d = p[j] - p[i]
    out["r2"] = [float(r2[(d >= lo) & (d < hi)].mean()) for lo, hi in zip(bins[:-1], bins[1:])]
    return out


def coalescent_comparison():
    import msprime
    n, L, seeds = 100, 200_000, 30
    bins = np.array([0, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000])
    res = {"numba": [], "msprime": []}
    for s in range(1, seeds + 1):
        G, pos, _ = simulate_dosages(n, L, recomb_rate=1e-8, mut_rate=1e-8, Ne=10_000, seed=s)
        res["numba"].append(_summaries(G, pos, bins))
        ts = msprime.sim_ancestry(n, ploidy=2, population_size=10_000, sequence_length=L,
                                  recombination_rate=1e-8, discrete_genome=False, random_seed=s)
        ts = msprime.sim_mutations(ts, rate=1e-8, discrete_genome=False, random_seed=s + 10_000,
                                   model=msprime.BinaryMutationModel())
        H = ts.genotype_matrix()
        res["msprime"].append(_summaries((H[:, ::2] + H[:, 1::2]).T, ts.tables.sites.position, bins))
    # Theory under the neutral coalescent: E[S] = theta a_n, E[pi] = theta (per sequence).
    theta = 4 * 10_000 * 1e-8 * L
    a_n = np.sum(1.0 / np.arange(1, 2 * n))
    rows, ev = [], {"theory_sites": theta * a_n, "theory_diversity": theta}
    for name in ("numba", "msprime"):
        S = np.array([r["sites"] for r in res[name]])
        P = np.array([r["diversity"] for r in res[name]])
        ev[name] = dict(sites=S.mean(), sites_se=S.std(ddof=1) / np.sqrt(seeds),
                        diversity=P.mean(), diversity_se=P.std(ddof=1) / np.sqrt(seeds),
                        r2=np.nanmean([r["r2"] for r in res[name]], axis=0).tolist(),
                        r2_se=(np.nanstd([r["r2"] for r in res[name]], axis=0, ddof=1)
                               / np.sqrt(seeds)).tolist())
        label = "built-in (Numba)" if name == "numba" else "msprime 1.4"
        rows.append([label, f"{S.mean():.0f} ({S.std(ddof=1) / np.sqrt(seeds):.0f})",
                     f"{P.mean():.0f} ({P.std(ddof=1) / np.sqrt(seeds):.0f})"])
    rows.append(["neutral expectation", f"{theta * a_n:.0f}", f"{theta:.0f}"])
    _write_table("coalescent", ["Backend", "Segregating sites (SE)", "Diversity $\\sum 2f(1-f)$ (SE)"],
                 rows, "lrr")
    diff = np.subtract(ev["numba"]["r2"], ev["msprime"]["r2"])
    se = np.hypot(ev["numba"]["r2_se"], ev["msprime"]["r2_se"])
    ev["r2_max_abs_z"] = float(np.max(np.abs(diff) / se))
    ev["bins"] = bins.tolist()
    ev["design"] = dict(n=n, L=L, seeds=seeds, Ne=10_000, recomb_rate=1e-8, mut_rate=1e-8)
    EVIDENCE["coalescent"] = ev

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mid = np.sqrt(np.maximum(bins[:-1], 500) * bins[1:]) / 1000
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for name, style in (("numba", dict(marker="o", color="#1f5f99")),
                        ("msprime", dict(marker="s", color="#c2571a", linestyle="--"))):
        ax.errorbar(mid, ev[name]["r2"], yerr=1.96 * np.asarray(ev[name]["r2_se"]), capsize=2,
                    label="built-in" if name == "numba" else "msprime", **style)
    ax.set_xscale("log")
    ax.set_xlabel("physical distance (kb)")
    ax.set_ylabel(r"mean $r^2$ (common SNPs)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "ld_decay.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. LD structure of the genotype models (figure)
# --------------------------------------------------------------------------- #
def ld_structure_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, k = 1000, 120
    panels = {
        "AR(1) blocks, $\\rho=0.9$": phensim.simulate_ar1_blocks(n, [60, 60], maf=0.3, rho=0.9, seed=1)[0],
        "founder haplotype blocks": phensim.simulate_haplotype_blocks(n, k, block_size=60, seed=2),
        "coalescent (built-in)": phensim.simulate_coalescent(n, k, 60, seed=3, backend="numba")[0],
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (title, G) in zip(axes, panels.items()):
        X = G.astype(float)
        sd = X.std(0)
        X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
        im = ax.imshow((X.T @ X / n) ** 2, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="$r^2$")
    fig.savefig(FIGURES / "ld_structure.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Indicative cost
# --------------------------------------------------------------------------- #
def timings():
    rows, ev = [], {}
    cases = [
        ("simulate\\_coalescent, $n=1000$, $m=10^4$, built-in",
         lambda: phensim.simulate_coalescent(1000, 10_000, 200, seed=1, backend="numba")),
        ("simulate\\_coalescent, $n=1000$, $m=10^4$, msprime",
         lambda: phensim.simulate_coalescent(1000, 10_000, 200, seed=1, backend="msprime")),
        ("simulate\\_ar1\\_blocks, $n=5000$, $m=2\\times10^4$, Cholesky",
         lambda: phensim.simulate_ar1_blocks(5000, [200] * 100, seed=1)),
        ("simulate\\_ar1\\_blocks, $n=5000$, $m=2\\times10^4$, scan",
         lambda: phensim.simulate_ar1_blocks(5000, [200] * 100, seed=1, method="scan")),
    ]
    G = phensim.simulate_ar1_blocks(3000, [100] * 50, maf=0.3, rho=0.7, seed=2)[0]
    K = phensim.grm(G)
    cases += [
        ("simulate\\_trait, $n=3000$, $m=5000$, $K$ omitted (matrix-free)",
         lambda: phensim.simulate_trait(G, seed=1)),
        ("simulate\\_trait, $n=3000$, $m=5000$, $K$ supplied (eigh)",
         lambda: phensim.simulate_trait(G, K=K, seed=1)),
        ("grm, $n=3000$, $m=5000$", lambda: phensim.grm(G)),
    ]
    blocks = [(_ar1(200, 0.8), np.arange(i * 200, (i + 1) * 200)) for i in range(250)]
    beta = np.zeros(50_000)
    ld = phensim.prepare_blocks(blocks)
    cases += [
        ("simulate\\_sumstats, $m=5\\times10^4$ (250 blocks of 200), raw blocks",
         lambda: phensim.simulate_sumstats(beta, blocks, 1e5, seed=1)),
        ("simulate\\_sumstats, same, prepared blocks",
         lambda: phensim.simulate_sumstats(beta, ld, 1e5, seed=1)),
        ("shake\\_ld, same, $n_{\\mathrm{ref}}=500$", lambda: phensim.shake_ld(ld, 500, seed=1)),
    ]
    for label, fn in cases:
        t = _timed(fn)
        ev[label] = t
        rows.append([label, f"{t:.2f}"])
    _write_table("timings", ["Call", "Seconds (median of 3)"], rows, "lr")
    EVIDENCE["timings"] = ev


def write_macros(ev):
    """``tables/macros.tex``: the numbers the report quotes in prose."""
    pv, ss, ph, co, pd = (ev[k] for k in ("provenance", "sumstats", "phenotypes", "coalescent", "pedigree"))
    macros = {
        "PhRevision": pv["revision"], "PhVersion": pv["phensim"], "PhNumpy": pv["numpy"],
        "PhNumba": pv["numba"], "PhMsprime": pv["msprime"], "PhPython": pv["python"],
        "PhMachine": pv["machine"], "PhLoad": f"{pv['load_average']:.1f}", "PhSeconds": f"{pv['seconds']:.0f}",
        "PhPedN": str(pd["n"]), "PhPedDraws": str(pd["draws"]),
        "PhPedRms": f"{pd['standardized_rms_error']:.3f}",
        "PhLambda": f"{ss['gwas_lambda_gc']['mean']:.3f}", "PhLambdaSd": f"{ss['gwas_lambda_gc']['sd']:.3f}",
        "PhEffectsErr": f"{ss['effects_h2_error']:.1e}",
        "PhBgFree": f"{ph['background_var']['matrix_free']:.3f}",
        "PhBgDense": f"{ph['background_var']['supplied_K']:.3f}",
        "PhCoalSeeds": str(co["design"]["seeds"]), "PhCoalN": str(co["design"]["n"]),
        "PhCoalKb": f"{co['design']['L'] // 1000}",
        "PhCoalMaxZ": f"{co['r2_max_abs_z']:.1f}",
    }
    tm = ev["timings"]
    free = next(v for k, v in tm.items() if "omitted" in k)
    dense = next(v for k, v in tm.items() if "supplied" in k)
    macros["PhTraitSpeedup"] = f"{dense / free:.0f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (TABLES / "macros.tex").write_text("\n".join(lines) + "\n")


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    t0 = time.time()
    G = phensim.simulate_coalescent(1500, 3000, 150, seed=11, backend="numba")[0]
    phenotype_targets(G)
    sumstats_targets()
    pedigree_check()
    coalescent_comparison()
    ld_structure_figure()
    timings()
    root = HERE.parent
    rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "phensim"],
                                capture_output=True, text=True).stdout.strip())
    import msprime
    import numba
    EVIDENCE["provenance"] = dict(
        phensim=phensim.__version__, revision=rev, phensim_dirty=dirty, numpy=np.__version__,
        numba=numba.__version__, msprime=msprime.__version__, python=platform.python_version(),
        machine=f"{platform.system()} {platform.machine()}",
        load_average=os.getloadavg()[0], seconds=round(time.time() - t0, 1),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"))
    (HERE / "evidence.json").write_text(json.dumps(EVIDENCE, indent=1, default=float) + "\n")
    write_macros(json.loads((HERE / "evidence.json").read_text()))
    print(json.dumps(EVIDENCE["provenance"]))


if __name__ == "__main__":
    main()
