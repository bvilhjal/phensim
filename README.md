# phensim

phensim simulates genotypes, phenotypes, GWAS summary statistics and
pedigrees for statistical-genetics research. The core package depends only
on NumPy. Version 1.0 replaces the 2019 snippet repository, which is
preserved under the `v0.1-legacy` tag.

**Genotype simulators** (`phensim.genotypes`)

Table 1. Available genotype structures and their computational character.

| Function | Structure | Cost |
|---|---|---|
| `simulate_independent` | none; fixed or SFS-shaped frequencies (beta / uniform / rare / common) | instant |
| `simulate_population_structure` | normal or Balding–Nichols drift in K populations; optional AR(1) LD blocks; `fst=0` control | O(nm) |
| `simulate_haplotype_blocks` | founder-haplotype copying with LD within blocks | one pass |
| `simulate_ar1_blocks` | latent-Gaussian haplotypes with AR(1) decay `rho`, thresholded at the MAF quantile; right-skewed geometry via `realistic_block_sizes`; `method="scan"` opts into an O(nk) forward recursion instead of the per-block Cholesky | one pass |
| `simulate_coalescent` | coalescent with recombination: haplotype + recombination LD; target SNP count, contiguous LD blocks | seconds |
| `simulate_by_mutation_rate` | fixed segment with mutation-rate-controlled SNP density on a seed-fixed genealogy | seconds |
| `simulate_hapnest` / `iter_hapnest` | HAPNEST age-dependent copying from phased references; discrete populations; optional Numba, batched or disk-backed output | O(nm), bounded working memory |

The coalescent family has two backends: the **msprime** C library
(the `msprime` extra) and a **built-in Numba
coalescent-with-recombination** engine (Hudson ancestry with Fenwick-tree
lineage sampling, infinite-sites mutations, JIT end-to-end; correct
pure-Python fallback) extracted from the
[ldpred3](https://github.com/bvilhjal/ldpred3) benchmark suite. All
simulators return sample-major `int8` dosages in {0, 1, 2}, columns in
physical order.
Coalescent dosages count the derived allele without random flips;
`simulate_independent` randomly orients the counted allele at each site.

**Several populations and admixture** (`phensim.ancestry`)

Table 2. Multi-population simulators and the ancestry truth they return.

| Function | Structure | Truth returned |
|---|---|---|
| `drift_frequencies` | per-population frequencies drifted (Balding–Nichols or normal) from one ancestral spectrum; one F_ST per population | `(K, m)` frequencies |
| `simulate_populations` | discrete populations with exact sizes and their own AR(1) LD (`rho`, block lengths per population) | population labels |
| `simulate_admixed` | pulse-admixture mosaics: tracts along a genetic map, each following its ancestry's AR(1) model | local ancestry per haplotype and variant |
| `simulate_split_coalescent` | msprime split of K populations at a target F_ST, optionally with an admixture pulse | labels; local ancestry of admixed samples |
| `simulate_genome` | independently seeded coalescent chromosomes; zero population LD across chromosomes; bit-parity with bipred's block genome | chromosome labels; one LD block per chromosome |
| `simulate_meta_sumstats` | per-population GWAS under each ancestry's own LD plus the fixed-effect meta-analysis | per-population `bhat`, inverse-variance meta and weights |

**Phenotype simulators** (`phensim.phenotypes`) draw the infinitesimal
component as u ~ N(0, sigma2 K) on the empirical GRM. Without a supplied
kinship, `simulate_trait` and `simulate_correlated_traits` use `m + 1`
innovations and an exact factor of the scaled GRM, avoiding an `n x n`
matrix and its eigendecomposition. A supplied `K` uses the
eigendecomposition draw. `simulate_confounded_trait` also avoids forming
`K` when an explicit `environment` is given; otherwise one
eigendecomposition supplies both the structure axis and the background.

- `simulate_trait`: quantitative traits with `mixed` / `infinitesimal`
  / `qtl` architectures, h2 targeting, normal or equal effect sizes;
- `simulate_binary_trait`: liability-threshold case/control with a
  target prevalence;
- `simulate_confounded_trait`: an explicit environmental exposure or the
  leading kinship eigenvector, plus a chosen base genetic architecture;
- `simulate_gxe_trait`: genotype-environment interaction variance;
- `simulate_correlated_traits`: bivariate traits with a target genetic
  correlation rg.

For a matched structure/LD experiment, pass `model="balding-nichols"`,
`block_sizes=[50] * (m // 50)` and `rho=0.8` to
`simulate_population_structure`; use `fst=0` for the control. Block lengths
must sum to `m`. Pass the returned population labels, or another chosen
exposure, as `environment=` to `simulate_confounded_trait`. Its
`confounding_strength` sets a component target, not the realized fraction
of phenotype variance: genetic and environmental components can covary.

**Ascertainment and scale conversions** (`phensim.phenotypes`):
`ascertain_case_control` samples exact case/control counts from a
simulated trait (the register / balanced-cohort schemes); the trait dict
must carry an exact binary `case_control` vector and a matching finite
`liability`, and counts must be nonnegative integers.
`n_eff_case_control` and `h2_liability` (Lee et al. 2011) convert between
observed and liability scales.

**Summary statistics** (`phensim.sumstats`) — the RSS layer on top of
block-diagonal population LD:

- `simulate_effects`: effect draws (sparse / polygenic / MAF-exponent /
  equal) pinned so `beta' R beta = h2`; the causal set is `n_causal`
  variants or Bernoulli(`p`), with `p` scalar or per variant;
- `simulate_effects_pair`: two traits with correlated shared effects --
  a shared Bernoulli(`p`) causal set, or exact per-trait/shared counts
  (the MiXeR four-state truth) -- each pinned to its h2;
  `genetic_correlation` gives the realized rg under the LD;
- `simulate_sumstats`: the oracle `bhat = R beta + N(0, R/n)`, scalar or
  per-variant N;
- `simulate_sumstats_pair`: two GWAS with an explicit sampling-noise
  correlation and per-trait N (`n_b`);
- `simulate_meta_sumstats` (`phensim.ancestry`): K populations' GWAS under
  their own LD with independent noise, plus the fixed-effect
  meta-analysis (inverse-variance = sample-size weights on this scale);
- `gwas_scan`: marginal GWAS (beta / se / z / p) from individual-level
  genotypes and a phenotype;
- `shake_ld`: finite reference-panel LD noise (Wishart panels), optionally
  shrunk toward I (`shrink`), with opt-in `chunk_size` row-chunked
  accumulation for large `n_ref`;
- `prepare_blocks`: one-time LD-block validation and factorization into
  a read-only snapshot every block consumer accepts.
- `simulate_gaussian_individuals`: continuous Gaussian individuals from raw
  or prepared LD blocks, with population covariance R and no sample rescaling.
  `shake_ld(factors=...)` also accepts rectangular or eigenvalue-clipped roots;
  its finite panel samples from the supplied factor's covariance.

**Pedigrees** (`phensim.pedigree`): `simulate_pedigree` (multi-
generation trio columns with remarriage and half-sibs),
`pedigree_birth_times` (generation-coherent calendar years),
`kinship_from_pedigree` (dense additive relationship matrix A with
inbreeding, Henderson tabular recursion), and `mendelian_draw`
(genetic values with covariance A in O(n) storage, exact inbreeding
variances). Parent references that are not listed ids warn once per
call and are treated as unknown founders.
Birth-time placement additionally requires discrete generations: some valid
acyclic pedigrees, including uncle–niece matings, cannot satisfy this constraint.

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
- [docs/hapnest.md](docs/hapnest.md): large reference-based cohorts, mutation
  ages, model boundaries, and speed/memory controls.
- [report/phensim_report.pdf](report/phensim_report.pdf): the technical
  report, with current API contracts and historical dev1 measurements:
  targets versus realized, built-in coalescent versus msprime, null
  calibration and indicative costs. From the repo root, compile it with
  `tectonic report/phensim_report.tex`. To refresh the measurements first,
  run `python report/make_evidence.py`; this replaces the saved evidence.

## Install

```sh
pip install "phensim[fast,msprime]"                      # from PyPI
pip install -e "./phensim[fast,msprime,test,lint]"         # or from a checkout
pip install "phensim @ git+https://github.com/bvilhjal/phensim.git"   # or from GitHub
```

Core dependency: NumPy only.

## Output and input conventions

Discrete genotypes are sample-major `int8` dosages. Quantitative `y` is
standardized, while `liability`, `u`, `q` and `e` retain their raw scale and
satisfy `liability = u + q + e`. Returned QTL effects multiply centred,
unit-SD causal genotypes. Trait simulators require complete dosages and use
only varying columns for automatic causal selection. Bivariate `rg` targets
the total genetic correlation, including QTLs; binary prevalence is exact
only for a Gaussian liability.

LD is supplied as dense `(R, indices)` blocks covering `0..m-1` once. Each
`R` must be finite, symmetric, unit-diagonal and positive semidefinite.
Singular matrices are supported without diagonal jitter. Effects are on the
standardized-genotype scale and are scaled so `beta' R beta = h2`. Convert
encoded or low-rank LDpred3 representations with `ldpred3.dense_ld` first;
phensim does not depend on LDpred3.

When the same LD feeds several draws, validate and factor it once:

```python
ld = phensim.prepare_blocks(blocks)
bhat = phensim.simulate_sumstats(beta, ld, n, seed=1)
panel = phensim.shake_ld(ld, n_ref=2000, seed=2, chunk_size=500)
```

PLINK output counts BIM allele 2 (G); negative values and NaN are missing.
Name G in PLINK 1.9 `--score` input, and use `--recode-allele` with
`--recode A` to request G dosages. Fractional dosages cannot be written as
binary hard calls. Validation finishes before any output file is opened.

The [technical reference](docs/technical.md) gives the equations, draw
orders, validation tolerances and edge-case behavior. Seeded bit-parity with
historical sibling benchmark draws is pinned in `tests/test_family_parity.py`.

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

The coalescent repairs of the 1.0 rewrite change seeded recombining draws
from the legacy 2019 snippets (the `v0.1-legacy` tag). Record package
version and source revision with benchmark results; legacy results are not
evidence for the corrected simulator.

## Citation

If phensim contributes to your research, cite the technical report
([report/phensim_report.pdf](report/phensim_report.pdf)) and the version
you used:

```bibtex
@misc{phensim,
  title        = {phensim: genotype, phenotype and summary-statistic
                  simulation for statistical genetics},
  author       = {{The phensim authors}},
  year         = {2026},
  howpublished = {Version 1.0.1 technical report; source code version 1.0.2},
  url          = {https://github.com/bvilhjal/phensim}
}
```

## References

Primary sources for the models implemented; algorithmic sources are
listed in [docs/technical.md](docs/technical.md).

- Balding & Nichols (1995), *Genetica* 96:3–12 — F_ST frequency drift.
- Hudson (1983), *Theor. Popul. Biol.* 23:183–201 — the coalescent with recombination.
- Kelleher et al. (2016), *PLoS Comput. Biol.* 12:e1004842, and
  Baumdicker et al. (2022), *Genetics* 220:iyab229 — msprime.
- Wharrie et al. (2023), *Bioinformatics* 39:btad535 — HAPNEST reference-based copying.
- Yang et al. (2010), *Nat. Genet.* 42:565–569 — the called-standardized GRM.
- Kang et al. (2010), *Nat. Genet.* 42:348–354 — the EMMAX kinship scaling.
- Lee et al. (2011), *Am. J. Hum. Genet.* 88:294–305 — liability-scale h² conversion.
- Zhu & Stephens (2017), *Ann. Appl. Stat.* 11:1561–1592 — the RSS oracle.
- Frei et al. (2019), *Nat. Commun.* 10:2417 — the MiXeR bivariate overlap truth.
- Zeng et al. (2018), *Nat. Genet.* 50:746–753 — the MAF-exponent effect architecture (SBayesS `S`).
- Speed et al. (2012), *Am. J. Hum. Genet.* 91:1011–1021 — LDAK-style variant weighting.
- Henderson (1976), *Biometrics* 32:69–83 — the tabular relationship matrix.
- Sved (1971), *Theor. Popul. Biol.* 2:125–141, and Hill & Weir (1988),
  *Theor. Popul. Biol.* 33:54–78 — LD-decay expectations used by the benchmarks.
- Purcell et al. (2007), *Am. J. Hum. Genet.* 81:559–575, and Chang et al.
  (2015), *GigaScience* 4:7 — PLINK.

## AI-assisted development

Parts of phensim's code, tests, benchmarks, documentation and technical
report were developed with the assistance of AI tools under human
direction, and were reviewed and tested before release. The software and
documentation are provided as-is under the MIT license;
[CHANGELOG.md](CHANGELOG.md) records the development and review history.
