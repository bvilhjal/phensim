# phensim guide

phensim simulates the data that statistical-genetics methods are tested on:
genotypes with realistic linkage disequilibrium (LD), phenotypes with known
genetic architecture, GWAS summary statistics under population LD, pedigrees,
kinship matrices and PLINK files. Every simulator is seeded, validates its
inputs, and returns the ground truth alongside the data. The core needs only
NumPy. [Numba](https://numba.pydata.org) speeds up the built-in coalescent,
and [msprime](https://tskit.dev/msprime) is an optional second coalescent
backend.

It is the shared simulator of the LDpred3 family (ldpred3, bipred, gwfm,
g2gfun, multipgs). Their benchmark genotype simulators re-export phensim,
and phensim reproduces their summary-statistic simulators bit for bit. For the models and their
mathematics see [technical.md](technical.md). For measured validation see
the report, `report/phensim_report.pdf`.

```sh
pip install -e "./phensim[fast,msprime]"   # fast = Numba, msprime = second backend
```

## What it simulates

| Task | Functions | Ground truth returned |
|---|---|---|
| Genotypes without LD | `simulate_independent`, `simulate_population_structure` | population labels (structure) |
| Genotypes with block LD | `simulate_ar1_blocks`, `simulate_haplotype_blocks`, `realistic_block_sizes` | block index arrays |
| Genotypes from the coalescent | `simulate_coalescent`, `simulate_by_mutation_rate` | contiguous LD blocks; columns in physical order |
| Several populations, admixture | `drift_frequencies`, `simulate_populations`, `simulate_admixed`, `simulate_split_coalescent` | per-population frequencies; labels; local ancestry |
| Independent chromosomes | `simulate_genome` | chromosome labels; exact zero-LD chromosome blocks |
| Quantitative traits | `simulate_trait` | `u`, `q`, `e`, `liability`, `causal`, `effects` |
| Case/control traits | `simulate_binary_trait`, `ascertain_case_control` | liability and case status; sampled indices |
| Confounding, GxE, two traits | `simulate_confounded_trait`, `simulate_gxe_trait`, `simulate_correlated_traits` | structure axis; interaction terms; `g_a`, `g_b` |
| GWAS summary statistics | `simulate_effects`, `simulate_effects_pair`, `simulate_sumstats`, `simulate_sumstats_pair`, `gwas_scan` | effects with `beta' R beta = h2`; realized `genetic_correlation` |
| Reference-panel LD | `shake_ld`, `prepare_blocks` | finite-panel noisy LD; validated, factored LD |
| Families | `simulate_pedigree`, `pedigree_birth_times`, `kinship_from_pedigree`, `mendelian_draw` | relationship matrix A; genetic values ~ N(0, A) |
| Kinship | `grm`, `ibs_kinship`, `loco_kinships`, `iter_loco_kinships`, `windowed_kinships` | — |
| Scale conversions | `h2_liability`, `n_eff_case_control` | — |
| Output | `write_plink` | `.bed/.bim/.fam` |

All genotype simulators return sample-major `int8` dosages in {0, 1, 2}. The
dosage counts the allele that `write_plink` labels allele 2.

## Choosing a genotype model

| Model | Use it when | Cost |
|---|---|---|
| `simulate_independent` | LD is irrelevant (a GWAS null, unit tests) | instant |
| `simulate_population_structure` | testing stratification control: K populations drifted by `fst` (normal approximation or exact Balding–Nichols) | instant |
| `simulate_ar1_blocks` | you want smooth, tunable LD decay (`rho`) and exact block boundaries; the family's default test genotypes | one pass |
| `simulate_haplotype_blocks` | you want haplotype sharing (a few founders per block) cheaply | one pass |
| `simulate_coalescent` | LD has to look real: recombination-driven decay, a realistic allele-frequency spectrum, a target SNP count | seconds |
| `simulate_by_mutation_rate` | scaling studies: the genealogy is fixed by the seed and the mutation rate sets SNP density | seconds |

```python
import numpy as np
import phensim

G, blocks = phensim.simulate_ar1_blocks(2000, [200] * 50, maf=0.3, rho=0.9, seed=1)
G, blocks = phensim.simulate_coalescent(2000, 10_000, 200, seed=1)        # built-in engine
G = phensim.simulate_by_mutation_rate(2000, 1e6, mut_rate=2e-8, seed=7,   # fixed 1 Mb segment
                                      backend="msprime")
```

Right-skewed block geometry is available through `realistic_block_sizes(m, n_blocks)`.
The coalescent has two backends: `"numba"` (built-in, also runs as pure
Python) and `"msprime"`. `"auto"` prefers Numba when it is installed. The
two backends draw from the same model but not the same events, so record
which one you used.

## Several populations and admixture

The controlled route draws per-population frequencies once and reuses them
for reference panels, GWAS samples and admixed targets. Each population
can have its own LD (`rho`, and block lengths in `simulate_populations`):

```python
freqs = phensim.drift_frequencies(4000, 2, fst=[0.05, 0.15], seed=1)    # (2, m)
G, pop = phensim.simulate_populations([3000, 3000], freqs, [50] * 80,
                                      rho=[0.9, 0.5], seed=2)
alpha = np.random.default_rng(3).uniform(0.05, 0.95, 800)
G_adm, local = phensim.simulate_admixed(800, freqs, [50] * 80,
                                        np.c_[alpha, 1 - alpha],
                                        generations=8, cm=0.01, rho=[0.9, 0.5], seed=4)
```

`local` is `(n, 2, m)`: the ancestry of each haplotype at each variant,
enough for per-haplotype genetic values such as
`sum_k (local == k) * beta_k` on phased output (`phased=True`). Tract
length is set by `generations` and the map: junctions occur at rate
`generations` per Morgan, so pass a realistic `cm` map (or spacing) and
`chromosome` labels.

For LD that comes from a genealogy, `simulate_split_coalescent` splits K
populations at the time that gives the requested F_ST and can add an
admixed population. It needs msprime; for any other demography, use
msprime directly.

```python
out = phensim.simulate_split_coalescent([2000, 2000], 10_000, 200, fst=0.12,
                                        admixed=1000, proportions=[0.8, 0.2],
                                        generations=10, seed=5)
out["G"], out["population"], out["local_ancestry"]   # admixed rows come last
```

## A genome of independent chromosomes

`simulate_genome` concatenates independently seeded coalescent chromosomes,
so cross-chromosome LD is exactly zero (not merely decaying) and each
chromosome keeps exactly its share of `m`:

```python
G, blocks, chrom = phensim.simulate_genome(4000, 22_000, 22, seed=1)
# one 1000-SNP block per chromosome: the partition for block consumers
```

Chromosome `c` (counting from 1) is drawn with seed
`seed * n_chromosomes + c`, so `seed=0` is valid -- the scheme of the
family's `genome(rep)` benchmarks, which `simulate_genome(min_maf=0.02,
seq_len=0.6e6, mut_rate=3e-8, seed=rep)` reproduces bit for bit. Pass its `chrom` on as the
`chromosome=` argument of `simulate_admixed`, `write_plink` or
`loco_kinships`.

## Meta-analysis summary statistics

`simulate_meta_sumstats` draws each population's GWAS under its own LD and
adds the fixed-effect meta-analysis. Sampling noise is independent across
populations (ancestry-stratified GWAS share no participants), and on the
standardized oracle scale `se^2 = 1/n`, so the inverse-variance weights are
`n_k / sum(n)`:

```python
meta = phensim.simulate_meta_sumstats([beta_eur, beta_afr], ld_pops,
                                      n=[250_000, 80_000], seed=3)
meta["bhat"]    # (2, m) per-population marginal effects
meta["meta"]    # (m,) inverse-variance weighted combination
meta["weights"] # (2, m)
```

`ld_pops[k]` is ancestry k's reference LD — one `(R, ix)` block list, e.g.
the sample correlations of a per-ancestry panel drawn by
`simulate_populations` with that ancestry's `rho`.

Per-population effects come from the caller: pass the same vector twice for
a shared architecture, or correlated vectors (e.g. `simulate_effects_pair`
under one population's LD) for ancestry-specific effects.

## Phenotypes with known truth

```python
tr = phensim.simulate_trait(G, h2=0.5, n_causal=20, architecture="mixed", seed=2)
tr["y"]          # standardized phenotype
tr["liability"]  # u + q + e on the raw scale
tr["u"], tr["q"] # infinitesimal background ~ N(0, h2/2 K) and QTL component
tr["causal"], tr["effects"]   # effects act on standardized causal columns

cc = phensim.simulate_binary_trait(G, prevalence=0.05, h2=0.5, seed=3)
study = phensim.ascertain_case_control(cc, n_cases=200, n_controls=800, seed=4)
pair = phensim.simulate_correlated_traits(G, h2_a=0.4, h2_b=0.6, rg=0.5, seed=5)
```

The architectures are:
- `mixed`: half of h2 from the kinship background, half from `n_causal` QTLs;
- `infinitesimal`: all of h2 from the background;
- `qtl`: all of h2 from the QTLs.

The background is drawn from the same scaled GRM a mixed model would fit.
Without a supplied `K` this needs no `n x n` matrix (see technical.md).
`simulate_confounded_trait` adds a population-structure axis, and
`simulate_gxe_trait` adds genotype-by-environment variance. Liability-scale
conversions follow Lee et al. (2011). `h2_liability` warns unless you pass the
study case fraction.

## GWAS summary statistics under LD

LD is a list of `(R, indices)` blocks that tile `0..m-1`. Each `R` must be a
dense correlation matrix. Effects are on the standardized-genotype scale.

```python
ld = phensim.prepare_blocks(blocks_R)            # validate and factor once
beta = phensim.simulate_effects(ld, h2=0.3, n_causal=50, seed=1)
beta = phensim.simulate_effects(ld, h2=0.3, p=0.01, seed=1)    # each variant causal w.p. p
bhat = phensim.simulate_sumstats(beta, ld, n=50_000, seed=2)   # R beta + N(0, R/n)

# two traits: shared causal variants with correlated effects, each at its own h2
b1, b2 = phensim.simulate_effects_pair(ld, 0.3, 0.5, rho=0.6, p=0.05, seed=3)
b1, b2 = phensim.simulate_effects_pair(ld, rho=0.8, n_causal=(400, 200), n_shared=100, seed=3)
phensim.genetic_correlation(b1, b2, ld)          # realized rg under the LD
bh1, bh2 = phensim.simulate_sumstats_pair(b1, b2, ld, 40_000, noise_correlation=0.1,
                                          n_b=25_000, seed=4)

ref = phensim.shake_ld(ld, n_ref=500, shrink=0.05, seed=5)     # finite reference panel
```

Across populations rather than traits, `simulate_meta_sumstats` (previous
section) draws each ancestry's GWAS under its own LD and meta-analyses them.
Noise options for awkward LD:
- `jitter=1e-4` draws the noise from the Cholesky factor of `R + 1e-4 I`.
- `factors=[...]` takes your own per-block factors, for example a clipped
  eigen-root of thresholded LD that is not positive semidefinite.

With these options phensim reproduces the family's benchmark draws bit for
bit (`tests/test_family_parity.py`), so a sibling can switch to it without
changing seeded results.

`simulate_effects(architecture="maf")` uses ldpred3's `alpha`: the per-allele
effect variance is proportional to `[2f(1-f)]^alpha`, so `alpha = -1` is flat
on the standardized scale.

`gwas_scan(G, y)` runs a marginal GWAS on individual-level data. It handles
missing calls per variant and returns NaN for untestable variants.

## Families and kinship

```python
ids, father, mother = phensim.simulate_pedigree(n_founder_pairs=150, gens=3, seed=1)
A = phensim.kinship_from_pedigree(ids, father, mother)     # 2 x kinship, with inbreeding
a, diagA = phensim.mendelian_draw(ids, father, mother, seed=2)   # a ~ N(0, A), O(n) storage
years = phensim.pedigree_birth_times(ids, father, mother)

K = phensim.grm(G)                                          # Yang-2010, mean diagonal 1
loco = phensim.loco_kinships(G, chromosomes)                # exact leave-one-chromosome-out
```

Pedigree ids may be any hashable values, including `0`. A parent of `None`,
`""`, NaN, or an unlisted `0` means "unknown". Any other unlisted parent is
also treated as unknown, with a warning.

## Writing PLINK files

```python
phensim.write_plink(G, "sim/cohort", chromosome=chrom, position=pos, sample_ids=ids)
```

Negative values and NaN are written as missing. Fractional dosages are
rejected, because PLINK 1 stores hard calls. Every check runs before any
file is opened.

## Reproducibility

- Every function takes `seed`: an integer, or a `numpy.random.Generator`
  whose stream continues across calls.
- Coalescent seeds must be integers in `[1, 2**31)`. The built-in kernel
  would otherwise alias large seeds. `simulate_genome`'s seed may be 0,
  since its chromosome seeds start at 1.
- Seeded outputs can change between versions; `CHANGELOG.md` lists every
  change. Record `phensim.__version__` and the git revision with your results.
- Sibling caches key on phensim's source:
  - bipred tags cached segments by a hash of `_coalescent.py`, `genotypes.py`
    and `_numba.py`, plus a hand-bumped msprime tag;
  - gwfm's simulation scope hashes those three plus `__init__.py`,
    `sumstats.py` and `_common.py`, so every phensim release (the version
    lives in `__init__.py`) moves it;
  - ldpred3's run archives hash every module.
- New features that do not touch genotypes therefore go into other modules,
  so bipred's cached segments are not invalidated needlessly.
