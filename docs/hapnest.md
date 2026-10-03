# Large cohorts by haplotype copying

`simulate_hapnest` implements the genotype model of
[Wharrie et al. (2023), HAPNEST](https://doi.org/10.1093/bioinformatics/btad535).
It constructs new chromosomes from a phased reference panel. The number of
synthetic individuals can greatly exceed the reference size. This makes
large computational experiments feasible; it does not create independent
information about diversity absent from the reference.

Let a reference population contain N diploid individuals. For each copied
segment, draw its age T and its genetic length L:

Equation 1. Segment-age and segment-length distributions.

\[
T\sim\operatorname{Gamma}(\text{shape}=2,\ \text{scale}=N_e/N),\qquad
L\mid T\sim\operatorname{Exp}(\text{rate}=2T\rho).
\]

Map positions and L are in centimorgans; ages are in generations. A donor
is uniform within the assigned population. A reference allele coded 1 is
copied only when its mutation age exceeds T; otherwise it becomes 0.
Consequently, finite mutation ages can change allele frequencies. The method
does not introduce new variant sites. Do not use arbitrary reference/alternate
coding when mutation ages refer to derived alleles: encode 1 consistently
with the variant whose age is supplied.

The input is `(reference individuals, 2, variants)` with binary integer
haplotypes. Population labels refer to diploids, not haplotypes. The public
API supports discrete donor populations and restarts copying at chromosome
boundaries. It does not implement admixture times, demographic inference,
upstream phenotype generation, or HAPNEST's ABC parameter selection. Choose
Ne and rho against independent reference data when fidelity matters; the
default values are convenient model parameters, not a universal human fit.

```python
import numpy as np
import phensim

# Controlled synthetic reference: 600 people, three populations, 12,000 SNPs.
H, reference_pop = phensim.simulate_population_structure(
    600, 12_000, fst=.05, model="balding-nichols",
    block_sizes=[50]*240, rho=.8, phased=True, seed=10)
chrom = np.repeat(np.arange(6), 2000)
cm = np.tile(np.arange(2000)*.001, 6)
ages = np.full(12_000, 1000.)
G = np.lib.format.open_memmap("cohort.npy", mode="w+", dtype="int8",
                             shape=(50_000, 12_000))
phensim.simulate_hapnest(
    H, 50_000, cm, ages, reference_populations=reference_pop,
    sample_populations=np.arange(50_000) % 3, chromosome=chrom,
    ne=10_000, rho=.7, out=G, seed=11)
G.flush()
trait = phensim.simulate_trait(G, seed=12, genotype_block_size=128)
phensim.write_plink(G, "cohort", chromosome=chrom)
```

This example uses controlled AR(1) LD and synthetic mutation ages. It does
not reproduce human LD or a fitted demographic history. The same API accepts
empirical phased arrays and calibrated map/age inputs, including read-only
memory maps. Supplying infinite ages explicitly disables allele erosion.

For minimum memory use, consume `iter_hapnest` batches instead of retaining
the entire cohort. A materialized int8 result needs n*m bytes. A memory map
avoids a second full allocation, but resident file pages can still raise RSS;
it is not a promise of constant resident memory. The iterator needs only a
sample batch plus the reference and O(n) population labels. It never stores
all segment records or both full synthetic haplotype matrices. PLINK writing
validates and packs bounded variant tiles. `genotype_block_size` similarly
avoids the full float64 genotype factor during phenotype generation.

`backend="auto"` uses optional Numba (`pip install -e '.[fast]'`) for references
in native byte order. Other byte orders fall back to NumPy without copying
the full reference; explicit `backend="numba"` requires native byte order.
`backend="numpy"` retains a readable segment-wise oracle. The compiled path
fuses random draws, binary map searches, age filtering and dosage addition.
It uses no fast-math approximation. Backends and batch sizes consume the same
NumPy Generator stream; seeded output agreement is tested. Compilation/cache
loading is a one-time cost and is reported separately from warm timings.

The implementation follows two conventions checked against upstream commit
`ba52da1a63cf609306ea92540b3d130fa1efd213`: phase 1 copies reference phase 1
(and likewise phase 2), and each segment includes the first marker beyond
its sampled endpoint. The latter matters at sparse marker densities. This
is an independent Python implementation of the model, not an executed or
bitwise-validated port of the Julia program. Seed identities are local to
phensim and its recorded NumPy/Numba versions.

Reproduce the performance experiment with `python benchmarks/hapnest_scaling.py
--out NEW_DIRECTORY`, with BLAS/OpenMP/Numba thread counts fixed to one.
The frozen design is in [hapnest_plan.md](../benchmarks/hapnest_plan.md).
Tests include backend/batch/output equality, chromosome and phase boundaries,
the Gamma survival probability, invalid inputs, and tiled versus dense
phenotype/PLINK oracles. Validation on real reference panels should additionally
check allele frequencies, LD decay, population PCs, and excessive sharing
with donors using data held out from any parameter tuning.
