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
bhat = phensim.simulate_sumstats(beta, ld, n=50_000, seed=2)   # R beta + N(0, R/n)

# two traits: shared causal variants with correlated effects, each at its own h2
b1, b2 = phensim.simulate_effects_pair(ld, 0.3, 0.5, rho=0.6, p=0.05, seed=3)
b1, b2 = phensim.simulate_effects_pair(ld, rho=0.8, n_causal=(400, 200), n_shared=100, seed=3)
phensim.genetic_correlation(b1, b2, ld)          # realized rg under the LD
bh1, bh2 = phensim.simulate_sumstats_pair(b1, b2, ld, 40_000, noise_correlation=0.1,
                                          n_b=25_000, seed=4)

ref = phensim.shake_ld(ld, n_ref=500, shrink=0.05, seed=5)     # finite reference panel
```

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
  would otherwise alias large seeds.
- Seeded outputs can change between versions; `CHANGELOG.md` lists every
  change. Record `phensim.__version__` and the git revision with your results.
- Sibling caches key on phensim's source:
  - bipred tags cached segments by a hash of `_coalescent.py`, `genotypes.py`
    and `_numba.py`, plus a hand-bumped msprime tag;
  - gwfm fingerprints the same files;
  - ldpred3's run archives hash every module.
- New features that do not touch genotypes therefore go into other modules,
  so those caches are not invalidated needlessly.
