# phensim technical reference

This reference defines the models, algorithms and numerical conventions.
See [guide.md](guide.md) for examples and `report/phensim_report.pdf` for
measured validation and costs. The report's measurements retain their
labeled dev1 provenance; they were not rerun after later implementation
changes.

## Conventions

- Discrete genotypes are sample-major `(n, m)` `int8` dosages in {0, 1, 2}, with
  columns in physical order. Kinship and GWAS inputs may mark missing calls
  as negative values or NaN; trait simulators require complete dosages.
- Effects act on standardized genotypes (each column centred, unit SD), so
  the genetic variance of `beta` under LD `R` is $\beta^\top R \beta$.
- LD is a list of dense `(R, ix)` blocks that tile `0..m-1` exactly once,
  in any order. Blocks are consumed in list order, one RNG draw per block.
- `seed` is an integer, `None`, or a `numpy.random.Generator`; a Generator
  continues its stream, so composed simulations stay reproducible.

## Genotype models

**Independent SNPs.** The frequency is either fixed (`maf`) or drawn from a
named spectrum: `beta` is $0.5\,\mathrm{Beta}(0.35, 1)$, `uniform` is
$U(0.01, 0.5)$, `rare` is $\mathrm{Exp}(0.02)$ clipped to $[10^{-4}, 0.5]$,
and `common` is $U(0.1, 0.5)$. Each site's minor allele is the counted
allele with probability 1/2. Dosages are $\mathrm{Bin}(2, p_j)$.

**Population structure.** Base frequencies are
$p_j = \mathrm{clip}(\text{maf} + N(0, 0.05^2), 0.05, 0.95)$. Population
$k$ draws either $p_{jk} \sim N(p_j, F p_j(1-p_j))$ (`normal`) or the exact
Balding–Nichols $p_{jk} \sim \mathrm{Beta}(p_j(1-F)/F, (1-p_j)(1-F)/F)$.
Both are clipped to $[0.01, 0.99]$. Individuals get uniform population labels.

**AR(1) blocks.** Within a block of $k$ SNPs each person has two latent
haplotypes $z \sim N(0, C)$ with $C_{ij} = \rho^{|i-j|}$. A haplotype
carries the allele where $z_j > \Phi^{-1}(1 - f_j)$; the dosage is the sum
of the two haplotypes. `method="cholesky"` draws $z = L\varepsilon$ with
$LL^\top = C + 10^{-8} I$ (the family's historical stream). `method="scan"`
uses the $O(nk)$ recursion
$z_j = \rho z_{j-1} + \sqrt{1-\rho^2}\,\varepsilon_j$. Both consume the same
normals, two $(n, k)$ draws per block. LD is smooth within a block and
zero between blocks. `realistic_block_sizes` draws log-normal block lengths
with coefficient of variation `cv`, repaired to sum exactly to $m$.
Extreme finite `cv` uses scaled log-weights to avoid overflow; ordinary draws
retain their historical RNG and rounding protocol.

**Founder haplotype blocks.** Each block has `n_founders` Bernoulli(0.3)
haplotypes. Each individual copies two founders, and every site flips with
probability `mutation_rate`.

**Coalescent.** `simulate_coalescent` grows a segment until it holds at
least $m$ common SNPs (MAF above `min_maf`), keeps the first $m$, and cuts
them into contiguous blocks of `block_size`, dropping any partial final block.
`simulate_split_coalescent` uses the same target and rounding rule. The first segment is
$\max(1, m/1200)$ Mb long; each retry multiplies it by 1.8, with a fresh
replicate seed, for up to 7 tries. `simulate_by_mutation_rate` fixes the
segment instead. The seed fixes the genealogy and `mut_rate` sets the
density, so the same seed at a higher rate gives the same chromosome with
more variants.
Both coalescent APIs count the derived allele without random flips. The
MAF filter uses the smaller of the derived and ancestral frequencies;
the retained dosages can therefore count either the minor or major allele.

**Reference-based copying.** `simulate_hapnest` and `iter_hapnest` implement
the HAPNEST copying model described in [hapnest.md](hapnest.md): a segment age
is Gamma(shape=2, scale=Ne/reference diploids), its length is exponential with
rate `2 * age * rho` per cM, and only donor alleles older than the segment are
copied. Chromosome starts reset the process; numeric labels must be finite.
NumPy and optional Numba consume the same Generator draws in the same order,
independent of sample batch size. Storage is bounded by the sample batch,
with disk-backed reference and output supported. Donor populations are discrete.

## Several populations and admixture (`phensim.ancestry`)

**Drifted frequencies.** `drift_frequencies` draws the ancestral spectrum
$p_j \sim U(0.05, 0.95)$ unless one is supplied, then per population $k$
the same `normal` or Balding–Nichols draw as the structured simulator, with
its own $F_k$ (a constant $F$ reproduces that stream). $F_k = 0$ copies
$p_j$. Results are clipped to `[min_freq, 1 - min_freq]`.

**Discrete populations.** `simulate_populations` draws each population in
turn with `simulate_ar1_blocks` (its frequencies, `rho` and block lengths),
continuing one random stream, and concatenates the rows. Sizes are exact
and labels sorted, unlike `simulate_population_structure`'s random labels.

**Admixture mosaics.** For each haplotype of an individual with
proportions $\alpha$, tract junctions form a Poisson process of rate $T$
per Morgan (`generations`), so the probability of a junction between
adjacent variants $d$ cM apart is $1 - e^{-Td/100}$; each chromosome starts
with a junction. At a junction the tract's ancestry is redrawn from
$\alpha$ (it may repeat). Inside a tract of ancestry $k$ the latent
haplotype follows the AR(1) scan $z_j = \rho_k z_{j-1} +
\sqrt{1-\rho_k^2}\,\varepsilon_j$, restarted at junctions and block edges,
and carries the allele where $z_j > \Phi^{-1}(1 - f_{kj})$. A junction
starts a new founder haplotype, so LD across it is zero; ancestry shared
along a tract gives admixture LD. Between variants with frequency
difference $\delta_j$ and one-SNP blocks, the haplotype correlation is
$\delta_i \delta_j\,\alpha(1-\alpha)\,e^{-Td/100} / \sqrt{\bar p_i \bar q_i
\bar p_j \bar q_j}$, which the tests check. Per block the stream draws
junction uniforms, ancestry uniforms and normals, each `(2n, k)`.

**Split coalescent.** `simulate_split_coalescent` builds an msprime
demography: K populations of size $N_e$ split together from an ancestor
of size $N_e$ at $t = 2N_eF/(1-F)$, so the expected Hudson
$F_{ST} = t/(t + 2N_e) = F$. An admixed population, if requested, forms at
`generations` $< t$ as one pulse with the given proportions. A census
between the pulse and the split records each lineage's ancestor; the local
ancestry of an admixed haplotype at a site is that ancestor's population
(`tskit` `link_ancestors`). Common SNPs are filtered on the pooled sample.

**A genome of independent chromosomes.** `simulate_genome` concatenates
$n_c$ segments of `simulate_by_mutation_rate`, chromosome $c$ (from 1)
seeded with $\text{seed} \cdot n_c + c$ -- bipred's `_block_genome` scheme,
which this reproduces bit for bit at its constants (`seq_len` $= 0.6$ Mb,
`mut_rate` $= 3\times10^{-8}$, `min_maf` $= 0.02$, one block per
chromosome). Recombination never links the segments, so the realized
cross-chromosome $r^2$ is pure sampling noise; the tests check it sits at
$1/(n-1)$ while adjacent within-chromosome pairs share genealogical LD.
$m$ is split as evenly as possible; each segment is redrawn with the
mutation rate raised $1.6\times$ (at most four draws, one genealogy) until
it covers its share, and extra columns are dropped.

**Meta-analysis summary statistics.** `simulate_meta_sumstats` draws
population $k$'s GWAS as `simulate_sumstats` under that population's own
LD and sample size, streaming one generator through the populations in
order, with independent sampling noise. The meta-analysis is the
inverse-variance weighted combination; on the standardized oracle scale
$\mathrm{se}_k^2 = 1/n_k$ exactly, so the weights $w_k = n_k/\sum_j n_j$
collapse inverse-variance, fixed-effect sample-size and beta-based
weighting into the same estimator, whose variance under zero effects is
$\sum_k w_k^2/n_k = 1/\sum_k n_k$ (checked by the tests).

## The built-in coalescent engine (`phensim._coalescent`)

The engine implements Hudson's coalescent with recombination for one
constant-size diploid population. Time is in generations.
1. **Ancestry.** The simulation starts from $2n$ haploid lineages, each
   carrying $[0, L)$. With $k$ lineages, coalescence occurs at total rate
   $\binom{k}{2}/(2N_e)$. Recombination occurs at rate $r\sum_i \ell_i$,
   where $\ell_i$ is the span from a lineage's leftmost to rightmost
   ancestral position. A Fenwick tree samples a lineage in proportion to
   $\ell_i$ in $O(\log k)$. The breakpoint is uniform on that span (gaps
   included) and clamped to the representable interior.
2. **Merging.** Merging two lineages walks their segment lists. An
   overlapping interval creates a node and two edges. Each segment
   tracks how many samples descend from it; an interval whose count reaches
   $2n$ has found its most recent common ancestor and is dropped, which is
   what ends the process.
3. **Mutations** follow infinite sites: each edge draws a
   $\mathrm{Poisson}(\mu \cdot \text{branch length} \cdot \text{span})$
   number of mutations at uniform positions on the branch.
4. **Densification** sweeps the marginal trees left to right. Edges are
   inserted in (left, parent time) order and removed in (right, −parent
   time) order, tskit style. Each mutation adds one to every sample below
   its node; haplotypes $2i$ and $2i+1$ form individual $i$.

All buffers are preallocated from expected event counts. On overflow the
engine doubles the buffer and reruns with the same seed, which reproduces
the same events, so callers never see partial results. Seeds are masked to
31 bits; the mutation stream uses $(s \cdot 2654435761) \bmod 2^{31}$. The
pure-Python fallback draws from NumPy's global RNG, which it saves and
restores around each kernel call, including when `NUMBA_DISABLE_JIT=1`.
The msprime backend calls `sim_ancestry`
and `sim_mutations` on a continuous genome with the binary mutation model,
using the same seed for both.

## Kinship

`grm` standardizes each variant over its called samples (Yang et al. 2010),
giving missing calls zero weight, forms $ZZ^\top/m$, and applies EMMAX
scaling: subtract the mean off-diagonal, then divide by the mean diagonal.
Columns whose called values have an exactly zero range contribute zero, including
fractional constants with centering roundoff; small real variation is retained.
`loco_kinships` subtracts each chromosome's Gram from the full Gram, with
the same global standardization, so the leave-one-chromosome-out matrix is
exact up to floating-point arithmetic. `windowed_kinships` splits the Gram
the same way into a window and its complement. A complement with no varying
called columns returns exact zero with `scale=False` and raises with scaling;
subtraction roundoff must not supply a diagonal scale.
`ibs_kinship` is the per-pair fraction of identical called
genotypes.

## Phenotype models

`simulate_trait` builds $\text{liability} = u + q + e$.

**Background.** The background is $u \sim N(0, \sigma^2 K)$, with $K$
the EMMAX-scaled GRM. With $Z$ the called-standardized genotypes, every
column of $Z$ sums to zero. With $S = \sum_{ij} Z_{ij}^2$, the scaled GRM
is therefore exactly

$$K = \frac{n-1}{S} Z Z^\top + \frac{1}{n} J.$$

It factors as $K = FF^\top$ with $F = [\sqrt{(n-1)/S}\,Z,\ \mathbf{1}/\sqrt{n}]$.
So $u = \sigma(\sqrt{(n-1)/S}\,Zw + c/\sqrt{n})$ with $m+1$ standard
normals: no $n \times n$ matrix and no eigendecomposition. A supplied $K$
is drawn through its eigendecomposition, with negative eigenvalues set to
zero. No rescaling is applied to a supplied positive-semidefinite $K$:
mean diagonal one gives average marginal background variance $\sigma^2$.
Multiplying $K$ by three multiplies this variance by three; realized
sample variance still varies from draw to draw.

**QTL component.** The causal effects for $q$ are $N(0,1)$ (`normal`) or
random signs (`equal`). They act on the standardized causal columns and are
rescaled so that $\mathrm{Var}(q)$ equals the QTL share of $h^2$ exactly in
the sample. `effects` reports the rescaled values.
Automatic selection samples only columns with observed variation, capped
at the number available. Explicit constant causal columns are rejected.
Eligibility uses column minima and maxima, requiring $O(m)$ workspace.

**Residual and architectures.** $e \sim N(0, 1-h^2)$, and `y` is the
standardized liability. The shares of $h^2$ are:
- `mixed`: half to $u$, half to $q$;
- `infinitesimal`: all to $u$;
- `qtl`: all to $q$.

**Binary traits** threshold the liability at $\Phi^{-1}(1-K)$. This hits
prevalence $K$ when the liability is close to standard normal.
The threshold is evaluated as $-\Phi^{-1}(K)$ to retain small-tail
precision, without the former probability truncation at $10^{-12}$.

**Confounded traits.** $\text{liability} = \sqrt{s}\,v + \sqrt{1-s}\,\ell$,
where $v$ is the standardized leading eigenvector of $K$ and $\ell$ a
`simulate_trait` liability that reuses the same eigendecomposition.
Before standardization, the leading eigenvector's largest-magnitude loading
is made positive. This fixes the axis's sign, but tied eigenvalues and the
background eigenbasis still preclude a general cross-LAPACK seed guarantee.

**Gene–environment interaction.** The interaction term is the standardized
causal genotypes times the standardized $E$, with random-sign effects, and
is scaled to `interaction_h2`. The additive part targets
$h^2 - h^2_{\mathrm{int}}$ and the residual $1-h^2$.

**Correlated traits.** Trait A is
$g_a = \sqrt{h_a^2/2}\,(u_1 + q_1)$. Trait B combines a shared draw with an
independent one in both components,
$g_b = \sqrt{h_b^2/2}\,(r_g u_1 + \sqrt{1-r_g^2}\,u_2 + r_g q_1 + \sqrt{1-r_g^2}\,q_2)$,
where $u_1,u_2$ are independent $N(0,K)$ draws without sample-SD rescaling,
and $q_2$ uses fresh effects on the same causal set. `u_a` and `u_b` retain
these Gaussian backgrounds; their sample variances fluctuate. This targets the
total genetic correlation $r_g$ and is exact at $r_g = \pm 1$.

**Ascertainment.** `ascertain_case_control` samples exact case and control
counts without replacement. `h2_liability` applies Lee et al. (2011),
$h^2_l = h^2_o K^2(1-K)^2 / (z^2 P(1-P))$, where $z = \phi(\Phi^{-1}(1-K))$
and $P$ is the study case fraction. `n_eff_case_control` is
$4/(1/N_{\mathrm{case}} + 1/N_{\mathrm{control}})$.

## Summary statistics (the RSS layer)

**Model.** Marginal standardized effects are drawn from
$\hat\beta = R\beta + \varepsilon$, with $\varepsilon \sim N(0, R/n)$. With
per-variant $N$ the noise covariance is $DRD$, where $D_{jj} = n_j^{-1/2}$.
The draw is $\varepsilon = F z / \sqrt{n}$, one $z \sim N(0, I_r)$ per block,
using one of three noise factors $F$:
- by default, the exact factor of $R$: Cholesky, or $V\Lambda^{1/2}$ when
  $R$ is singular (no diagonal noise is added);
- with `jitter`, the Cholesky factor of $R + \epsilon I$;
- with `factors`, a factor you supply, so $R$ itself need not be PSD.

The signal always uses $R$.

**Two GWAS.** `simulate_sumstats_pair` draws $z_a$ and then $z_b$ in each
block, setting $\varepsilon_b = F(\rho z_a + \sqrt{1-\rho^2} z_b)/\sqrt{n_b}$.
The cross-covariance is $\rho R / \sqrt{n_a n_b}$. $\rho$ is the
sampling-noise correlation, not the fraction of overlapping participants.

**Effects.** `simulate_effects` draws a shape (`sparse`, `polygenic`,
`equal` or `maf`) and rescales it so that $\beta^\top R \beta = h^2$
exactly. The `sparse` and `equal` causal set is either exactly
$\min(n_\text{causal}, m)$ variants or independent Bernoulli($p_j$)
indicators; if the indicators select nothing, one variant with $p_j > 0$
is forced. The `maf` shape uses
$\beta_j \propto [2f_j(1-f_j)]^{(1+\alpha)/2}$, ldpred3's $\alpha$.
`simulate_effects_pair` draws shared effects
$N(0, [[1, \rho], [\rho, 1]])$ and scales each trait to its own $h^2$.
There are two layouts:
- a shared Bernoulli(`p`) causal set, with at least one causal variant;
- exact counts $n_a$, $n_b$ and $n_{ab}$ (the MiXeR four-state truth),
  with target $r_g = \rho\, n_{ab}/\sqrt{n_a n_b}$.

`genetic_correlation` returns
$\beta_a^\top R \beta_b / \sqrt{\beta_a^\top R\beta_a \cdot \beta_b^\top R\beta_b}$.

**Reference panels.** `shake_ld` draws $X = ZF^\top$ with `n_ref` rows,
standardizes the columns as $(X - \bar X)/\mathrm{sd}(X)$, and returns
$(1-s)X^\top X/n_{\mathrm{ref}} + sI$. The diagonal is 1 to roundoff.
`chunk_size` accumulates the same draws in row chunks with a Chan
one-pass update, bounding memory at $O(\text{chunk} \cdot k + k^2)$.

**Gaussian individuals.** `simulate_gaussian_individuals` draws each block as
independent standard-normal rows multiplied by the transpose of its covariance
factor. The returned float64 matrix has population covariance `R`; it is not
empirically centered or standardized. Singular LD uses its exact PSD root.
Raw blocks are validated and factored one at a time; prepared blocks reuse
their stored factors and produce identical seeded draws. The output costs
O(nm) storage, with only the current block's factor and innovations transient.

**Validation.** `prepare_blocks` validates and factors the blocks once,
storing read-only copies. Every consumer otherwise validates on each call,
in tiles of 256 rows. The checks are coverage, finiteness, symmetry, range
$[-1, 1]$, unit diagonal and PSD. Tolerances follow the input's precision:

Table 1. LD validation tolerances.

| Check | Tolerance |
|---|---|
| Symmetry, range and diagonal | relative `max(1e-7, 4 eps)` |
| Negative eigenvalues (PSD) | down to `k * max(64 eps64 max(1, lambda_max), eps_in)` |

Here `eps_in` is the unit roundoff of the input's dtype, so float32 LD is
judged at float32 precision.

**`gwas_scan`.** Each variant is fitted on its called samples. With $r$ the
Pearson correlation, the outputs are `beta` $= r$,
`se` $= \sqrt{(1-r^2)/(n_{\mathrm{called}}-2)}$, and `z` $= r/\mathrm{se}$
(the OLS $t$ statistic). `p` is $\mathrm{erfc}(|z|/\sqrt{2})$. Variants
with fewer than 3 calls, or with constant called genotypes or phenotypes
(tested exactly), return NaN.

## Pedigrees

`kinship_from_pedigree` orders people parents-first (Kahn's algorithm) and
fills the relationship matrix with Henderson's tabular method:
$A_{kj} = (A_{s_k j} + A_{d_k j})/2$ and $A_{kk} = 1 + A_{s_k d_k}/2$,
with unknown parents contributing zero. The cost is $O(n^2)$; the result is
exactly symmetric and includes inbreeding.

`mendelian_draw` produces $a \sim N(0, A)$ without forming $A$:
$a_i = \sum_{\text{known } p} a_p/2 + \sqrt{w_i}\,z_i$, with
$w_i = 1 - \sum_{\text{known } p} A_{pp}/4$. This is the Mendelian sampling
variance $(1 - (F_s + F_d)/2)/2$ when both parents are known. Its diagonals
come from a memoized pair recursion with an iterative stack and an LRU
bound.

`pedigree_birth_times` solves exact relative generation constraints along
parent-child edges in both directions, in O(n) time and storage. Each connected
pedigree is shifted so its earliest generation starts at the base year;
unrelated founders can enter later. Children are exactly one generation later
than their recorded parents. Some valid acyclic pedigrees, such as
uncle–niece matings, cannot meet these discrete-generation constraints; the
error distinguishes this limitation from an ancestry cycle. Kinship and
Mendelian draws remain valid for such acyclic pedigrees.

## PLINK output

`write_plink` writes SNP-major BED. The 2-bit codes are 00 for homozygous
allele 1, 10 for heterozygous, 11 for homozygous allele 2 and 01 for
missing, packed little-endian four samples per byte with zero-padded final
bytes. The BIM alleles are `A G`, and the dosage counts `G`. Every check
runs before any file is opened.
PLINK 1.9 [`--score`](https://www.cog-genomics.org/plink/1.9/score) counts the
allele named in its score file, so use G for these dosages. `--recode A`
counts A1, which may change when PLINK loads the data. To reproduce G
dosages, supply a two-column variant-ID/G file with
[`--recode-allele`](https://www.cog-genomics.org/plink/1.9/data#recode);
`--keep-allele-order --recode A` instead counts the BIM's A allele and
returns two minus the nonmissing input dosage.

## Test coverage

Table 2. Main regression suites and the behavior they pin.

| Test file | What it pins |
|---|---|
| `test_phensim.py`, `test_regressions.py` | contracts and independent oracles: a BED decoder, direct OLS with missing calls, variance-component reconstruction, $r_g$ endpoints, window-complement GRMs |
| `test_coalescent_invariants.py` | the segment-list and Fenwick invariants at every event; msprime agreement on site count, diversity and short- versus long-range $r^2$ over 75 seeds (slow leg) |
| `test_family_parity.py` | bit parity with the ldpred3 and bipred benchmark simulators |
| `test_ancestry.py` | drift variance $F p(1-p)$; populations as sequential AR(1) draws; the pulse admixture-LD curve and junction rate; tract frequencies and LD; split-coalescent $F_{ST}$ and local ancestry (with msprime); the independent-chromosome genome (cross-chromosome $r^2$ at the $1/(n-1)$ noise floor, bit parity with bipred's `_block_genome`, both backends); meta sumstats (stream parity with sequential `simulate_sumstats`, IVW weights, the $1/\sum_k n_k$ noise moment) |
| `test_review_fixes.py`, `test_review_optimizations.py` | the adversarial-review regressions; that the optimized paths match the reference paths |

## Known limitations

- The built-in coalescent models one constant-size population.
  `simulate_split_coalescent` adds one simultaneous split and an optional
  admixture pulse through msprime, with equal population sizes; use msprime
  directly for other demography, migration or selection.
- Admixture mosaics use the AR(1) model inside tracts, so their LD stops at
  block edges like the source populations'.
- AR(1) and haplotype-block LD stop at block edges. Only the coalescent
  produces LD that decays with physical distance.
- Binary prevalence is exact only for a Gaussian liability. Sparse or
  structured liabilities shift the case fraction.
- The RSS oracle assumes summary statistics from a single homogeneous
  sample. Per-variant $N$ is modelled as $DRD$, not as arbitrary
  missingness. `simulate_meta_sumstats` meta-analyses K such samples with
  independent noise; it does not model cross-population effect
  heterogeneity (the caller supplies the per-population `betas`) nor
  random-effects meta-analysis.
- The independent-chromosome genome gives each chromosome the same
  `seq_len`, `Ne` and rates; chromosome lengths differ only in SNP count.
  It returns chromosome labels, not base-pair positions.
- No age-of-onset or follow-up models yet. Those still live in ltpred
  (`simulate_followup_records`) and aadgen.

## References

Primary sources for the models and algorithms described above. Measured
evidence and the original API contracts are in `report/phensim_report.pdf`,
whose bibliography overlaps this list.

- Balding, D. J. and Nichols, R. A. (1995). A method for quantifying
  differentiation between populations at multi-allelic loci and its
  implications for investigating identity and paternity. *Genetica*
  96:3–12.
- Baumdicker, F. et al. (2022). Efficient ancestry and mutation simulation
  with msprime 1.0. *Genetics* 220:iyab229.
- Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983). Algorithms for
  computing the sample variance: analysis and recommendations. *The
  American Statistician* 37:242–247.
- Chang, C. C. et al. (2015). Second-generation PLINK: rising to the
  challenge of larger and richer datasets. *GigaScience* 4:7.
- Fenwick, P. M. (1994). A new data structure for cumulative frequency
  tables. *Software: Practice and Experience* 24:327–336.
- Frei, O. et al. (2019). Bivariate causal mixture model quantifies
  polygenic overlap between complex traits beyond genetic correlation.
  *Nature Communications* 10:2417.
- Henderson, C. R. (1976). A simple method for computing the inverse of a
  numerator relationship matrix used in prediction of breeding values.
  *Biometrics* 32:69–83.
- Hill, W. G. and Weir, B. S. (1988). Variances and covariances of squared
  linkage disequilibria in finite populations. *Theoretical Population
  Biology* 33:54–78.
- Hudson, R. R. (1983). Properties of a neutral allele model with
  intragenic recombination. *Theoretical Population Biology* 23:183–201.
- Kang, H. M. et al. (2010). Variance component model to account for sample
  structure in genome-wide association studies. *Nature Genetics* 42:348–354.
- Kelleher, J., Etheridge, A. M. and McVean, G. (2016). Efficient coalescent
  simulation and genealogical analysis for large sample sizes. *PLoS
  Computational Biology* 12:e1004842.
- Lee, S. H., Wray, N. R., Goddard, M. E. and Visscher, P. M. (2011).
  Estimating missing heritability for disease from genome-wide association
  studies. *American Journal of Human Genetics* 88:294–305.
- Purcell, S. et al. (2007). PLINK: a tool set for whole-genome association
  and population-based linkage analyses. *American Journal of Human
  Genetics* 81:559–575.
- Speed, D., Hemani, G., Johnson, M. R. and Balding, D. J. (2012).
  Improved heritability estimation from genome-wide SNPs. *American Journal
  of Human Genetics* 91:1011–1021.
- Sved, J. A. (1971). Linkage disequilibrium and homozygosity of chromosome
  segments in finite populations. *Theoretical Population Biology* 2:125–141.
- Wharrie, S. et al. (2023). HAPNEST: simulating biobank-scale genomic and
  phenotypic data. *Bioinformatics* 39:btad535.
- Yang, J. et al. (2010). Common SNPs explain a large proportion of the
  heritability for human height. *Nature Genetics* 42:565–569.
- Zeng, J. et al. (2018). Signatures of negative selection in the genetic
  architecture of human complex traits. *Nature Genetics* 50:746–753.
- Zhu, X. and Stephens, M. (2017). Bayesian large-scale multiple regression
  with summary statistics from genome-wide association studies. *Annals of
  Applied Statistics* 11:1561–1592.
