# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] (1.0.0.dev4)

### Added (2026-10-08 several populations and admixture)

- `phensim.ancestry`, for the multi-population simulators that ppb, PLDSC,
  multipgs, mixmogam and LDpred3-atw each wrote for themselves:
  `drift_frequencies` (Balding–Nichols or normal drift, one F_ST per
  population), `simulate_populations` (exact sizes, per-population AR(1)
  `rho` and block lengths), `simulate_admixed` (pulse tracts along a
  genetic map with per-haplotype local ancestry) and
  `simulate_split_coalescent` (msprime split at a target F_ST, optional
  admixture pulse, local ancestry from a census).
- Tests check the drift variance, the pulse admixture-LD curve and
  junction rate, tract frequencies and LD, and coalescent F_ST and local
  ancestry. `genotypes.py` is unchanged, so no existing draw or sibling
  cache key moves; sibling simulators are not migrated.

### Fixed (2026-10-07 simulation edge cases)

- Automatic causal selection excludes columns without observed variation;
  explicit constant causal columns raise an actionable error. Selection is
  capped at the number of variable columns, with O(m) eligibility workspace.
- Pin the default confounding axis so its largest-magnitude eigenvector
  loading is positive. This removes sign ambiguity in the environmental
  component, without promising cross-LAPACK reproducibility of the background.
- Evaluate inverse-normal survival probabilities directly in the small tail,
  removing cancellation and the 1e-12 cutoff. AR(1) frequencies 0 and 1 use
  infinite thresholds; binary traits use the same stable survival quantile.
- Clarify the GxE zero-causal error and distinguish ancestry cycles from
  pedigrees that cannot satisfy the birth-time model's discrete generations.
- Document supplied kinship scaling, derived-allele coalescent coding, and
  PLINK scoring/recode allele selection; retain those existing conventions.
  An optional PLINK integration regression checks G-allele scores and dosages.

These fixes change seeded draws when causal eligibility, the default
confounding-axis sign, or extreme-tail thresholds change. Record the version
and source revision with new simulations; historical benchmarks retain their
original provenance.

### Added (2026-10-03 matched GWAS simulations)

- HAPNEST-model genotype expansion from phased reference arrays, including
  discrete donor populations, mutation-age filtering and chromosome resets.
  Optional Numba fuses segment generation and copying; `iter_hapnest` streams
  sample batches and `simulate_hapnest` accepts a writable memory map.
  NumPy/Numba and batch/output modes have exact seeded agreement tests.
- Phased output from AR(1) and population generators supplies small reference
  panels without inventing phase from unphased dosages. This remains a
  controlled LD model, not a demographic or empirical human reference.
- Opt-in `genotype_block_size` for trait/background generation preserves
  innovations and GRM covariance while bounding float64 workspace. PLINK
  output now validates and encodes variant tiles rather than a full payload.
- Reproducible time/RSS experiments through 100,000 synthetic samples,
  source snapshots, and a guide separating model support from future ABC,
  admixture and empirical-reference validation.
- Population simulations accept `fst=0` as a control and optional AR(1)
  `block_sizes`/`rho` for LD within populations. Existing positive-Fst,
  independent-marker seeded draws are unchanged.
- Confounded traits accept an explicit environmental exposure and the base
  trait's architecture, causal indices and effect distribution. Explicit
  exposures avoid constructing a dense kinship merely to define an axis.
  Component targets are documented separately from realized variance fractions.

### Fixed (2026-10-03 HAPNEST release review)

- Binary `uint64` reference haplotypes now work in the NumPy HAPNEST backend,
  matching Numba and the existing integer types. Conversion is limited to the
  current copied segment, retaining bounded memory for mapped references.
- Non-native-byte-order references use NumPy under `backend="auto"` without
  a full reference copy. Explicit Numba requests raise an actionable input
  error instead of a compiler typing failure.

### Changed (2026-10-03 ldpred3 clean-up)

- `simulate_ar1_blocks` accepts counted-allele frequencies in `[0, 1]`
  (was `[0, 0.5]`); the threshold model is exact for any frequency, and
  the family's example data and ppb draw them in `(0.1, 0.9)`. Seeded
  draws for previously valid input are unchanged.
- The coalescent output-property tests ldpred3 carried for this kernel
  moved here: segregating sites in physical order, LD decay with
  distance, buffer growth under high recombination, memory proportional
  to the output, and a folded-SFS comparison with msprime on a
  continuous genome (slow leg).

### Fixed (2026-10-03 peak memory)

- One data draw at n = 4,000, m = 50,000 (`simulate_coalescent` with
  the msprime backend, then `simulate_trait`) peaked at an 8.5 GB
  footprint; it now peaks at 2.5 GB. The msprime backend decodes int8
  dosages site by site instead of building tskit's int32 genotype
  matrix over every site. `_called_standardized` works on one float64
  matrix in place instead of four temporaries. Trait simulators keep
  integer genotypes as they are and convert only what they read. All
  draws are bit-identical (`tests/test_memory_lean.py`).

### Added (2026-10-03 shared summary-statistic simulators)

- `simulate_effects_pair` (shared Bernoulli causal set, or exact
  per-trait/shared counts with correlated shared effects, each trait
  pinned to its h2) and `genetic_correlation` (realized rg under the LD).
- `jitter=` and `factors=` on `simulate_sumstats` and
  `simulate_sumstats_pair` (noise from `chol(R + jitter I)` or caller
  factors, which may come from non-PSD thresholded LD), and per-trait
  sample sizes via `n_b=`.
- `shake_ld(shrink=, jitter=)`: panels shrunk toward I, drawn from the
  jittered factor.
- These reproduce the sibling benchmark simulators bit for bit (ldpred3
  `_metrics.sumstats`, `_realistic_ld.panel_genome`; bipred
  `rg_architectures.sim_effects`/`sumstats_pair`/`ref_panel`,
  `mixer_overlap._sim_mixture`), pinned by `tests/test_family_parity.py`,
  so they can migrate without re-freezing results.

### Changed (2026-10-03 shared summary-statistic simulators)

- `shake_ld` panels standardize as the family's benchmarks do,
  `(X - mean) / sd`, and no longer reset the diagonal to exactly 1 (it is
  1 to roundoff). Seeded panels move by about 1e-15; other seeded outputs
  are unchanged.

### Fixed (2026-10-03 follow-up review)

- `simulate_effects(architecture="maf")` scaled standardized effects by
  `[2f(1-f)]^(alpha/2)`, dropping the `+1` of the ldpred3 extraction
  source: `alpha=-1` gave a MAF-0.01 variant about 24x the standardized
  variance of a MAF-0.4 one instead of being flat. It now uses `[2f(1-f)]^((1+alpha)/2)`
  (per-allele variance proportional to `[2f(1-f)]^alpha`, ldpred3's `alpha` and
  SBayesS `S`); seeded `maf` draws change.
- Pedigree ids may be `0`/`"0"` again (as in ltpred, where they came from):
  `np.arange(n)` ids were rejected as missing. Only `None`, `""` and NaN
  ids are refused; a listed `0` resolves as a parent.
- `gwas_scan` tests constancy exactly on each variant's called values.
  Constant fractional dosages with missing calls returned pseudo-random
  finite statistics from cancellation roundoff (|z| up to ~1), and
  constant called phenotypes returned tiny ones; both are now NaN.
- LD validation judges float32 input at float32 precision (symmetry,
  unit diagonal and range at `max(1e-7, 4 eps)`, PSD clipping up to
  `k eps`). Valid float32 LD -- a singular panel, or entries one ulp
  off -- was rejected. float64 behaviour and seeded draws are unchanged.
- The coalescent seed is `None` or an integer in `[1, 2**31)` on both
  backends. The built-in kernel masked seeds to 31 bits, so `s` and
  `s + 2**31` silently gave identical replicates, while msprime rejected
  0 and seeds of `2**32` or more.

### Fixed (2026-10-03 adversarial review)

- Isolate the built-in coalescent's pure-Python fallback from the caller's
  global NumPy RNG state (saved and restored around every kernel call,
  including on errors; seeded output is unchanged), and clamp each
  recombination breakpoint to the representable interior of the lineage
  span so an endpoint draw can no longer walk off the segment list and
  index the segment arrays at -1.
- Request a continuous genome (`discrete_genome=False`) on the msprime
  backend for both ancestry and mutations, matching the built-in backend;
  seeded msprime outputs change.
- Warn once per call when pedigree parent columns reference unlisted ids
  (treated as unknown founders) in `kinship_from_pedigree`,
  `mendelian_draw` and `pedigree_birth_times`; require `mendelian_draw`
  innovations to be a finite length-`n` vector; validate
  `simulate_pedigree`'s `n_founder_pairs`, `gens` and `remarry`.
- `write_plink` validates the whole genotype matrix and all metadata --
  unique nonempty whitespace/control-free string sample ids, nonnegative
  integer chromosome codes or the X/Y/XY/MT labels, and nonnegative
  integral positions -- before opening any output file.
- Validate genotype matrices and EMMAX scaling in `phensim.kinship`
  (square finite `K`, positive post-offset mean diagonal), require at
  least two chromosomes for LOCO kinships, and correct the `ibs_kinship`
  docstring: the unrelated baseline depends on the genotype/allele-
  frequency spectrum; identical complete calls give 1 before scaling.
- Validate simulator arguments across `phensim.genotypes`: positive
  integer `n`/`m`/`n_pops`/`block_size`, `maf` in `[0, 0.5]`, `rho` in
  `[-1, 1]`, finite nonnegative coalescent rates, positive finite `Ne`,
  `min_maf` in `[0, 0.5)`, and `seq_len >= 1`. `simulate_coalescent`
  rejects `mut_rate=0` and `block_size > m` up front.
- `ascertain_case_control` requires a trait dict with both
  `case_control` and `liability`, an exact binary (0/1) case vector, a
  matching finite liability vector, and nonnegative integer counts; no
  rows are silently excluded or coerced.
- `write_plink` keeps the genotype matrix and positions in their
  original dtypes through validation: integer positions above `2**53`
  are preserved exactly instead of being rounded by a float64
  roundtrip, and integer positions that do not fit int64 are rejected.
- `iter_loco_kinships` forms each chromosome's Gram once (not twice)
  and rejects non-finite chromosome labels instead of silently dropping
  NaN-labelled variants from every leave-out set.
- `simulate_ar1_blocks` rejects block lengths, or a total length, that
  overflow the addressable size rather than wrapping to a negative
  allocation.
- `simulate_trait` raises a clear `ValueError` when a positive-QTL
  architecture is given zero causal variants (`n_causal=0` or an empty
  `causal`); zero remains valid there for infinitesimal and `h2=0`
  targets. `simulate_correlated_traits` always draws unit-variance QTL
  components, so `n_causal=0` raises unconditionally there.

### Added (2026-10-03 adversarial review)

- `simulate_ar1_blocks(method="scan")`: an opt-in O(nk) AR(1) forward
  recursion sampling the unjittered AR(1) covariance -- identical to the
  default's latent-Gaussian distribution up to its 1e-8 diagonal
  jitter -- with the same RNG call order but no per-block Cholesky
  factorization; deterministic at `rho = +/-1`. The default
  `method="cholesky"` is bit-identical.
- `phensim.kinship.iter_loco_kinships`: a lazy `(chrom, K_loco)`
  generator; `loco_kinships` is a dict over it and no longer retains one
  Gram matrix per chromosome.
- Matrix-free genetic backgrounds: `simulate_trait` (and the
  binary/GxE wrappers delegating to it) and `simulate_correlated_traits`
  draw `u ~ N(0, sigma2 K)` from `m + 1` innovations through an exact
  factor of the scaled GRM when `K` is omitted -- no `n x n` matrix or
  eigendecomposition. A supplied `K` keeps the eigendecomposition draw,
  and `simulate_confounded_trait` now computes that eigendecomposition
  once and shares it between the structure axis and the background.
- `prepare_blocks(blocks)`: validates and factors LD blocks once into
  a read-only snapshot; all four block consumers (`simulate_effects`,
  `simulate_sumstats`, `simulate_sumstats_pair`, `shake_ld`) accept it
  and skip re-validation and re-factorization. Raw `(R, ix)` lists keep
  working on every call.
- `shake_ld(chunk_size=...)`: opt-in accumulation of the Wishart
  reference panel in bounded row chunks (a centered one-pass moment
  update, `O(chunk * k + k^2)` storage). `None` or
  `chunk_size >= n_ref` is the bit-identical full-panel path; chunked
  draws consume the same RNG stream and can differ at roundoff.

### Changed (2026-10-03 adversarial review)

- `write_plink` encodes the BED payload with a vectorized 2-bit packing;
  output bytes are unchanged.
- `gwas_scan` computes p-values through a JIT-accelerated
  `_normal_pvalues` loop, bit-identical to the `math.erfc` comprehension
  it replaces (pure-Python fallback preserved).
- Seeded outputs change for phenotype draws that omit `K`:
  `simulate_trait`, `simulate_binary_trait`, `simulate_gxe_trait` and
  `simulate_correlated_traits` use the new matrix-free innovation
  layout. Supplied-`K` and `simulate_confounded_trait` seeded streams
  are preserved bit for bit. The default `simulate_ar1_blocks` stream
  and ordinary built-in-coalescent seeded draws are unchanged; the
  msprime continuous-genome change and the recombination endpoint clamp
  above are the documented exceptions.
- Provenance note for callers and sibling suites: callers passing their
  own `K` to the phenotype API see no drift; the sibling benchmark
  genotype/coalescent shims (ldpred3, gwfm, g2gfun, bipred, multipgs)
  re-export phensim genotypes, so only their *msprime-backend* seeded
  draws differ from earlier versions (continuous genome). Record the
  phensim version and source revision with benchmark results; historical
  seeded outputs are not evidence for the current simulator.

### Fixed (2026-10-02 adversarial review)

- Repair the built-in coalescent's right-lineage tail after recombination.
  Seeded recombining simulations change; historical benchmark outputs need
  their original version/source provenance and cannot be relabelled as current.
- Encode PLINK heterozygous, homozygous and missing calls correctly; reject
  fractional calls and mismatched metadata before opening output files.
- Correlate both genetic components in bivariate traits, and allocate GxE
  residual variance as `1-h2` with an independent RNG stream continuation.
- Return scaled QTL effects and inspectable liability components, including
  the scaling applied by confounded and GxE wrappers.
- Use per-variant complete-case OLS in `gwas_scan`; preserve infinite statistics
  for perfect association and mark untestable variants as NaN. P-values retain
  the documented large-sample normal approximation.
- Compute window-complement kinships from the full genotype cross-product;
  overlapping/gapped windows are correct and window matrices are streamed.
- Require exact LD index coverage and finite symmetric correlation blocks.
  Use a singular PSD eigenfactor without diagonal jitter, and reject materially
  indefinite LD. Validation follows LDpred3's bounded-memory conventions.
- Rename paired-GWAS noise control to `noise_correlation`; the old `overlap`
  keyword warns that it denotes noise correlation, not participant overlap.
- Preserve the final partial haplotype block, reject impossible block counts,
  and reject self-parent/co-parent generation cycles in birth-time assignment.
- Keep core tests active without msprime; add independent regression oracles,
  coalescent event invariants, selected msprime LD comparisons, and a NumPy-only
  Python/NumPy-floor CI job.

### Added (2026-10-02 cross-repo extraction)

- `phensim.genotypes.simulate_ar1_blocks` + `realistic_block_sizes`:
  the AR(1) latent-Gaussian block-LD simulator replicated across the
  ldpred3 / gwfm / g2gfun / ppb / PLDSC / aadgen suites, with its RNG
  call order preserved so ldpred3-lineage seeds reproduce bit for bit.
- `simulate_population_structure(model="balding-nichols")`: the exact
  BN drift distribution (previously the normal approximation only).
- `phensim.sumstats`: `simulate_effects` (sparse/polygenic/maf/equal
  architectures pinned to `beta'R beta = h2`), `simulate_sumstats`
  (the `R beta + N(0, R/n)` RSS oracle, scalar or per-variant N),
  `simulate_sumstats_pair` (sample-overlap-correlated noise),
  `gwas_scan` (marginal GWAS from genotypes) and `shake_ld` (Wishart
  reference-panel LD noise) — consolidated from ten-plus per-repo
  reimplementations.
- `phensim.phenotypes.ascertain_case_control` (exact-count case/control
  sampling) plus `n_eff_case_control` / `h2_liability` (Lee 2011),
  extracted from ldpred3.
- `phensim.pedigree`: `simulate_pedigree` (O(n) parent lookup; the
  ltpred original rescanned the id list per mating), 
  `pedigree_birth_times`, `kinship_from_pedigree` (dense A with
  inbreeding) and `mendelian_draw` (O(n)-storage draw with covariance
  A), extracted from ltpred.
- `phensim.kinship`: `ibs_kinship`, exact `loco_kinships` (additive
  subtraction), `windowed_kinships`, extracted from mixmogam.
- Floors relaxed to `python>=3.9` / `numpy>=1.20` so the extracted
  benchmark-suite code keeps running on ldpred3's floor CI leg, where it
  has always run as `benchmarks/_coalescent.py` + `simulate.py`.

### Changed

- Complete revision of the 2019 snippet repo (preserved under the
  `v0.1-legacy` tag) into an installable package.
- Genotype simulators: independent (fixed/SFS-shaped frequencies),
  population-structure (fst-diverged frequencies), haplotype-block
  founder copying, and a coalescent family with two backends (msprime;
  built-in Numba coalescent-with-recombination extracted from the
  ldpred3 benchmark suite, with a pure-Python fallback).
- Phenotype simulators: model-consistent quantitative traits (mixed /
  infinitesimal / QTL architectures through the GRM eigendecomposition),
  liability-threshold binary traits, structure-confounded traits, GxE
  interaction traits, and bivariate traits with target genetic
  correlation.
- `grm` kinship (Yang-2010 called-only standardization) and a PLINK 1
  binary writer.
- Test suite covering every simulator contract across backends; the old
  snippets had none.
