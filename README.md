# phensim

Genotype and phenotype simulators for genetic association studies.

phensim 1.0 is a complete revision of the 2019 snippet repo (preserved
under the `v0.1-legacy` git tag). It provides, in plain NumPy:

**Genotype simulators** (`phensim.genotypes`)

Table 1. Available genotype structures and their computational character.

| Function | Structure | Cost |
|---|---|---|
| `simulate_independent` | none; fixed or SFS-shaped frequencies (beta / uniform / rare / common) | instant |
| `simulate_population_structure` | diverged per-population frequencies (`fst`); normal or exact Balding–Nichols drift, K populations | instant |
| `simulate_haplotype_blocks` | founder-haplotype copying: genuine haplotypic LD within blocks | one pass |
| `simulate_ar1_blocks` | latent-Gaussian haplotypes with AR(1) decay `rho`, thresholded at the MAF quantile; right-skewed geometry via `realistic_block_sizes`; `method="scan"` opts into an O(nk) forward recursion instead of the per-block Cholesky | one pass |
| `simulate_coalescent` | coalescent with recombination: haplotype + recombination LD; target SNP count, contiguous LD blocks | seconds |
| `simulate_by_mutation_rate` | same, but fixed segment and mutation-rate density lever on a seed-fixed genealogy | seconds |

The coalescent family has two backends: the **msprime** C library
(`pip install phensim[msprime]`) and a **built-in Numba
coalescent-with-recombination** engine (Hudson ancestry with Fenwick-tree
lineage sampling, infinite-sites mutations, JIT end-to-end; correct
pure-Python fallback) extracted from the
[ldpred3](https://github.com/bvilhjal/ldpred3) benchmark suite. All
simulators return sample-major `int8` dosages in {0, 1, 2}, columns in
physical order.

**Phenotype simulators** (`phensim.phenotypes`) — all draw the
infinitesimal component as u ~ N(0, sigma2 K) on the empirical GRM, so
the data-generating covariance matches what a mixed model will fit.
`simulate_trait` (and the `simulate_binary_trait`/`simulate_gxe_trait`
wrappers delegating to it) and `simulate_correlated_traits` draw
matrix-free when no kinship is supplied: `m + 1` innovations through an
exact factor `F` of the scaled GRM (`F F' = K`), never an `n x n`
matrix or eigendecomposition. A supplied `K` keeps the classic
eigendecomposition draw. `simulate_confounded_trait` is the exception:
it still materializes the GRM for its leading axis, computing one
eigendecomposition shared by the structure axis and the background:

- `simulate_trait`: quantitative traits with `mixed` / `infinitesimal`
  / `qtl` architectures, h2 targeting, normal or equal effect sizes;
- `simulate_binary_trait`: liability-threshold case/control with a
  target prevalence;
- `simulate_confounded_trait`: structure-driven phenotypes on the
  leading kinship eigenvector (the genomic-control test scenario);
- `simulate_gxe_trait`: genotype-environment interaction variance;
- `simulate_correlated_traits`: bivariate traits with a target genetic
  correlation rg.

**Ascertainment and scale conversions** (`phensim.phenotypes`):
`ascertain_case_control` samples exact case/control counts from a
simulated trait (the register / balanced-cohort schemes); the trait dict
must carry an exact binary `case_control` vector and a matching finite
`liability`, and counts must be nonnegative integers.
`n_eff_case_control` and `h2_liability` (Lee 2011) convert between
observed and liability scales.

**Summary statistics** (`phensim.sumstats`) — the RSS layer on top of
block-diagonal population LD:

- `simulate_effects`: effect draws (sparse / polygenic / MAF-exponent /
  equal) pinned so `beta' R beta = h2`;
- `simulate_effects_pair`: two traits with correlated shared effects --
  a shared Bernoulli(`p`) causal set, or exact per-trait/shared counts
  (the MiXeR four-state truth) -- each pinned to its h2;
  `genetic_correlation` gives the realized rg under the LD;
- `simulate_sumstats`: the oracle `bhat = R beta + N(0, R/n)`, scalar or
  per-variant N;
- `simulate_sumstats_pair`: two GWAS with an explicit sampling-noise
  correlation and per-trait N (`n_b`);
- `gwas_scan`: marginal GWAS (beta / se / z / p) from individual-level
  genotypes and a phenotype;
- `shake_ld`: finite reference-panel LD noise (Wishart panels), optionally
  shrunk toward I (`shrink`), with opt-in `chunk_size` row-chunked
  accumulation for large `n_ref`;
- `prepare_blocks`: one-time LD-block validation and factorization into
  a read-only snapshot every block consumer accepts.

**Pedigrees** (`phensim.pedigree`): `simulate_pedigree` (multi-
generation trio columns with remarriage and half-sibs),
`pedigree_birth_times` (generation-coherent calendar years),
`kinship_from_pedigree` (dense additive relationship matrix A with
inbreeding, Henderson tabular recursion), and `mendelian_draw`
(genetic values with covariance A in O(n) storage, exact inbreeding
variances). Parent references that are not listed ids warn once per
call and are treated as unknown founders.

Plus `phensim.kinship` — `grm` (Yang-2010 called-only
standardization), `ibs_kinship`, exact leave-one-chromosome-out
`loco_kinships` and its lazy `iter_loco_kinships` generator, and
`windowed_kinships` local/global pairs — and
`phensim.io.write_plink` (PLINK 1 binary output for external tools).

## Documentation

- [docs/guide.md](docs/guide.md): what phensim simulates, how to pick a
  model, and recipes.
- [docs/technical.md](docs/technical.md): the models, the algorithms and
  the numerical contracts.
- [report/phensim_report.pdf](report/phensim_report.pdf): the technical
  report, with measured validation (targets versus realized,
  built-in coalescent versus msprime, null calibration) and indicative
  costs. Rebuild it with `python report/make_evidence.py` and then
  `tectonic report/phensim_report.tex`.

## Install

```sh
pip install -e "./phensim[fast,msprime,test,lint]"
```

Core dependency: NumPy only.

## Simulation truth and input contracts

Quantitative `y` is standardized; `liability`, `u`, `q` and `e` retain their
raw scale, with `liability = u + q + e`. Returned `effects` multiply centred,
unit-SD causal genotypes to recover `q`. Divide the effects by
`liability.std()` to express them on the standardized-y scale. Confounded
traits add `structure`; GxE traits add `interaction`, with matching
`interaction_effects`. GxE component targets are `h2-interaction_h2`,
`interaction_h2` and `1-h2`. Finite-sample covariance between components can
change their realized variance fractions. Trait inputs must be complete.

Bivariate `rg` targets the **total** genetic correlation, including QTLs;
`g_a` and `g_b` expose those genetic values. Finite-sample correlations
fluctuate, while `rg=+/-1` gives proportional genetic values. Binary traits
use a standard-normal liability threshold: the requested prevalence is
approximate when a sparse or structured liability is not normal.

Summary-statistic blocks are dense `(R, indices)` pairs covering every index
in `0..m-1` exactly once, in any block order. R must be finite, symmetric,
unit-diagonal and positive semidefinite. Singular LD is supported without
adding diagonal noise; only roundoff-sized negative eigenvalues are clipped
in its sampling factor, with roundoff judged at the input's precision
(float32 LD at float32). Validation uses bounded row tiles, as in LDpred3.
The `maf` effect architecture uses ldpred3's `alpha`: per-allele variance
proportional to `[2f(1-f)]^alpha`, flat on the standardized scale at `alpha=-1`.
Convert encoded/low-rank LDpred3 representations with `ldpred3.dense_ld`
before passing them; phensim itself does not depend on LDpred3.

The noise factor defaults to the exact PSD factor of each block; the
draws above also take `jitter=` (`chol(R + jitter I)`) or caller-made
`factors=` (e.g. a clipped root of thresholded LD, which need not be
PSD), while the signal always uses `R`. With these options the
siblings' benchmark draws -- ldpred3 `_metrics.sumstats` and
`panel_genome`, bipred `sim_effects`, `sumstats_pair`, `ref_panel` and
`_sim_mixture` -- are reproduced bit for bit (`tests/test_family_parity.py`).

`simulate_sumstats_pair(noise_correlation=rho)` controls the correlation
of GWAS sampling errors, not the fraction of shared participants. For the
equal-size conditional RSS model it is overlap fraction times standardized
residual correlation. The old `overlap` keyword preserves its numerical
meaning with a warning. `gwas_scan` fits each variant on its called samples;
its historical `z` field is the OLS t statistic and `p` is a large-sample
normal approximation. Untestable variants return NaN statistics.

When the same LD blocks feed several draws, `prepare_blocks(blocks)`
validates and factors them once; pass the result anywhere the raw
`(R, ix)` list is accepted — identical draws, no repeated work:

```python
ld = phensim.prepare_blocks(blocks)
bhat = phensim.simulate_sumstats(beta, ld, n, seed=1)
panel = phensim.shake_ld(ld, n_ref=2000, seed=2, chunk_size=500)
```

PLINK output counts BIM allele 2 (G); negative values and NaN denote missing
calls. Fractional dosages cannot be written as binary hard calls.
`write_plink` validates the whole genotype matrix and all metadata before
opening any file: sample ids must be unique nonempty whitespace-free
strings, chromosomes are nonnegative integer codes or the X/Y/XY/MT
labels, and positions are nonnegative integral base-pair values (0 means
unknown). Haplotype copying returns exactly the requested SNP count,
including partial blocks.

## Quickstart

```python
import numpy as np
import phensim

# LD-structured genotypes: 500 diploids, 10k common SNPs, 200-SNP blocks
G, blocks = phensim.simulate_coalescent(500, 10_000, 200, seed=1)

# a 60%-heritable trait, half infinitesimal, half 15 QTLs
tr = phensim.simulate_trait(G, h2=0.6, n_causal=15, seed=2)
y, causal = tr["y"], tr["causal"]

# case/control with 5% prevalence
bin = phensim.simulate_binary_trait(G, prevalence=0.05, h2=0.5, seed=3)
```

## Validation

Run `pytest -q -m "not slow"` for the core and regression tests. These include
independent BED decoding, direct OLS with missingness, total genetic
correlation endpoints, variance-component reconstruction, window-complement
GRMs and the coalescent's linked-segment/recombination-weight invariants.
Core tests run without msprime or Numba. `pytest -q` additionally compares
selected site-count, diversity and LD summaries with msprime over multiple
seeds when msprime is installed; these checks do not establish general
backend equivalence. Two scripts in `benchmarks/` measure more:
`coalescent_backend.py` compares speed, peak memory and the SFS across sizes,
and `ld_decay_validation.py` compares the full LD-decay curve with msprime
and with the Sved and Hill–Weir expectations.

The coalescent repair in `1.0.0.dev1` changes seeded recombining draws from
earlier versions. Record package version and source revision with benchmark
results; old results are not evidence for the corrected simulator.
