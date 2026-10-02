"""phensim: genotype and phenotype simulators for genetic studies."""

__version__ = "1.0.0.dev0"

__all__ = [
    "simulate_independent",
    "simulate_population_structure",
    "simulate_haplotype_blocks",
    "simulate_coalescent",
    "simulate_by_mutation_rate",
    "simulate_trait",
    "simulate_binary_trait",
    "simulate_confounded_trait",
    "simulate_gxe_trait",
    "simulate_correlated_traits",
    "grm",
    "write_plink",
    "HAVE_NUMBA",
    "__version__",
]

_MAP = {
    "simulate_independent": ("phensim.genotypes", "simulate_independent"),
    "simulate_population_structure": ("phensim.genotypes", "simulate_population_structure"),
    "simulate_haplotype_blocks": ("phensim.genotypes", "simulate_haplotype_blocks"),
    "simulate_coalescent": ("phensim.genotypes", "simulate_coalescent"),
    "simulate_by_mutation_rate": ("phensim.genotypes", "simulate_by_mutation_rate"),
    "simulate_trait": ("phensim.phenotypes", "simulate_trait"),
    "simulate_binary_trait": ("phensim.phenotypes", "simulate_binary_trait"),
    "simulate_confounded_trait": ("phensim.phenotypes", "simulate_confounded_trait"),
    "simulate_gxe_trait": ("phensim.phenotypes", "simulate_gxe_trait"),
    "simulate_correlated_traits": ("phensim.phenotypes", "simulate_correlated_traits"),
    "grm": ("phensim.kinship", "grm"),
    "write_plink": ("phensim.io", "write_plink"),
    "HAVE_NUMBA": ("phensim._numba", "HAVE_NUMBA"),
}


def __getattr__(name):
    if name in _MAP:
        import importlib

        module_name, attr = _MAP[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
