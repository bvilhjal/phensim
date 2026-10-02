# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] (1.0.0.dev0)

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
