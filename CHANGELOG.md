# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] (1.0.0.dev1)

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
