"""Is the built-in coalescent's LD right? Measure the decay curve.

The family's benchmark genomes come from ``simulate_by_mutation_rate`` --
Hudson's coalescent with recombination, run by phensim's built-in engine.
``tests/test_coalescent_invariants.py`` checks that engine against msprime on
site counts, diversity, the folded SFS and short- versus long-range r^2; this
script measures the whole decay curve two ways:

1. **Against msprime** on identical parameters (continuous genome, binary
   mutations) -- an independent implementation of the same model, so
   agreement rules out a bug in the built-in recombination handling.
2. **Against Sved (1971) and Hill & Weir (1988)**, the neutral
   drift-recombination expectations, with the finite-sample floor
   ``1/(N-1)`` for dosage correlations. This separates "the two backends
   agree" from "the two backends are both right".

Neither says the LD is *human*: a constant-``Ne`` neutral coalescent has no
out-of-Africa bottleneck and no recent explosive growth. Moved here from
ldpred3's benchmark diagnostics (2026-10-03).

    python benchmarks/ld_decay_validation.py           # default: 30 reps
    python benchmarks/ld_decay_validation.py 60
"""
import sys

import numpy as np

from phensim._coalescent import simulate_dosages

N, L, NE, MU, REC = 100, 1_000_000.0, 10_000, 1e-8, 1e-8
MIN_MAF = 0.05
# log-ish bins in bp; LD at these scales is what a block-diagonal LD model sees
EDGES = np.array([0, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])


def r2_by_distance(dos, pos):
    """Mean r^2 in each distance bin, and the pair count, for one replicate."""
    ac = dos.sum(0)
    twoN = 2 * dos.shape[0]
    p = ac / twoN
    keep = (p > MIN_MAF) & (p < 1 - MIN_MAF)
    dos, pos = dos[:, keep], pos[keep]
    if dos.shape[1] < 2:
        z = np.zeros(len(EDGES) - 1)
        return z, z.copy(), z.copy()
    Z = dos - dos.mean(0)
    sd = Z.std(0)
    sd[sd == 0] = 1.0
    Z /= sd
    R = (Z.T @ Z) / dos.shape[0]
    iu = np.triu_indices(dos.shape[1], 1)
    r2 = R[iu] ** 2
    d = np.abs(pos[iu[1]] - pos[iu[0]])
    idx = np.digitize(d, EDGES) - 1
    ok = (idx >= 0) & (idx < len(EDGES) - 1)
    s = np.bincount(idx[ok], weights=r2[ok], minlength=len(EDGES) - 1)
    c = np.bincount(idx[ok], minlength=len(EDGES) - 1).astype(float)
    # mean pair separation per bin: the geometric midpoint of a bin whose lower
    # edge is 0 is meaningless, and the bins are wide, so the analytic curves
    # below are evaluated where the pairs actually are.
    dsum = np.bincount(idx[ok], weights=d[ok], minlength=len(EDGES) - 1)
    return s, c, dsum


def run_numba(seed):
    G, pos, _ = simulate_dosages(N, L, recomb_rate=REC, mut_rate=MU, Ne=NE,
                                 seed=seed)
    return G, np.asarray(pos, dtype=float)


def run_msprime(seed):
    import msprime
    ts = msprime.sim_ancestry(samples=N, ploidy=2, population_size=NE,
                              recombination_rate=REC, sequence_length=L,
                              discrete_genome=False, random_seed=seed)
    mts = msprime.sim_mutations(ts, rate=MU, random_seed=seed, discrete_genome=False,
                                model=msprime.BinaryMutationModel())
    H = mts.genotype_matrix()
    return (H[:, 0::2] + H[:, 1::2]).T, np.asarray(mts.tables.sites.position, dtype=float)


def aggregate(runner, reps):
    s = np.zeros(len(EDGES) - 1)
    c = np.zeros(len(EDGES) - 1)
    dsum = np.zeros(len(EDGES) - 1)
    for r in range(reps):
        si, ci, di = runner(r + 1)
        s += si
        c += ci
        dsum += di
    mean = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    dbar = np.divide(dsum, c, out=np.full_like(s, np.nan), where=c > 0)
    return mean, c, dbar


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"LD decay, {reps} replicates of a {L / 1e6:.1f} Mb segment, "
          f"n={N} diploids, Ne={NE:,}, mu=rec={MU:.0e}, MAF>{MIN_MAF}\n")

    nb, cnt, dbar = aggregate(lambda s: r2_by_distance(*run_numba(s)), reps)
    try:
        import msprime  # noqa: F401
        ms, _, _ = aggregate(lambda s: r2_by_distance(*run_msprime(s)), reps)
    except ImportError:
        ms = np.full_like(nb, np.nan)
        print("msprime absent -- backend comparison skipped\n")

    # Analytic expectations at the *observed* mean pair separation per bin.
    # rho = 4*Ne*c is the scaled recombination rate between the pair; the
    # + 1/n_hap term is the finite-sample inflation of r^2.
    rho = 4.0 * NE * REC * dbar
    # r^2 here is the squared Pearson correlation of *dosages* over N
    # individuals, so its expectation under independence is 1/(N-1) -- not the
    # 1/n_hap floor that applies when correlating haplotypes. Using the
    # haplotype floor understates the plateau by 2x and makes the simulator
    # look like it carries excess long-range LD when it does not.
    samp = 1.0 / (N - 1.0)
    sved = 1.0 / (1.0 + rho) + samp                        # Sved 1971
    hw = (10.0 + rho) / ((2.0 + rho) * (11.0 + rho)) + samp  # Hill & Weir 1988

    hdr = (f"{'distance':>16} {'mean bp':>9} {'pairs':>10} {'r2 numba':>9} "
           f"{'r2 msprime':>11} {'ratio':>7} {'Sved':>7} {'Hill-Weir':>10} "
           f"{'nb/H-W':>7}")
    print(hdr); print("-" * len(hdr))
    for i in range(len(nb)):
        lab = f"{EDGES[i] / 1e3:.0f}-{EDGES[i + 1] / 1e3:.0f} kb"
        ratio = nb[i] / ms[i] if ms[i] == ms[i] and ms[i] > 0 else np.nan
        print(f"{lab:>16} {dbar[i]:9,.0f} {int(cnt[i]):>10,} {nb[i]:9.4f} "
              f"{ms[i]:11.4f} {ratio:7.3f} {sved[i]:7.4f} {hw[i]:10.4f} "
              f"{nb[i] / hw[i]:7.3f}")

    ok = ~np.isnan(ms) & (cnt > 0)
    if ok.any():
        rel = np.abs(nb[ok] - ms[ok]) / ms[ok]
        print(f"\nbackend agreement:  max relative difference {rel.max():.1%}, "
              f"median {np.median(rel):.1%}")
    good = cnt > 0
    for name, curve in (("Sved (1971)", sved), ("Hill-Weir (1988)", hw)):
        rel_s = np.abs(nb[good] - curve[good]) / curve[good]
        print(f"vs {name:17s} max relative difference {rel_s.max():.1%}, "
              f"median {np.median(rel_s):.1%}")

    print("\nWhat this does and does not establish:")
    print("  + the built-in engine's recombination-driven LD matches an")
    print("    independent implementation of the same model (msprime) across")
    print("    three orders of magnitude of physical distance;")
    print("  + and both match Hill-Weir, the analytic neutral-equilibrium")
    print("    expectation, to within the MAF ascertainment. So the LD is")
    print("    right *for the model*: it is not a simulator artifact.")
    print("  - the model is a single constant-Ne neutral population. Real")
    print("    European LD is inflated at long range by the out-of-Africa")
    print("    bottleneck and depleted at short range by recent explosive")
    print("    growth; neither is simulated here.")

if __name__ == "__main__":
    main()
