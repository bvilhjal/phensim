# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] (1.0.0.dev0)

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
