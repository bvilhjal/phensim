"""Pedigree simulator and pedigree-kinship tests."""

import numpy as np
import pytest

from phensim.pedigree import (simulate_pedigree, pedigree_birth_times,
                              kinship_from_pedigree, mendelian_draw)


def test_simulate_pedigree_contract():
    ids, father, mother = simulate_pedigree(n_founder_pairs=40, gens=3,
                                            remarry=0.2, seed=1)
    assert len(ids) == len(father) == len(mother)
    assert len(set(ids)) == len(ids)
    index = set(ids)
    for f, m in zip(father, mother):
        assert (f is None) == (m is None)  # founder or fully parented
        assert f is None or f in index
        assert m is None or m in index
    assert father[0] is None and mother[0] is None  # founders lead
    # same seed -> same pedigree
    ids2, *_ = simulate_pedigree(n_founder_pairs=40, gens=3, remarry=0.2, seed=1)
    assert ids == ids2


def test_simulate_pedigree_matches_ltpred_draws():
    ltpred_sim = pytest.importorskip("ltpred.simulate")
    a = simulate_pedigree(n_founder_pairs=25, gens=3, remarry=0.15, seed=11)
    b = ltpred_sim.simulate_pedigree(
        np.random.default_rng(11), n_founder_pairs=25, gens=3, remarry=0.15)
    assert a == b  # identical RNG call order reproduces ltpred pedigrees


def test_pedigree_values_match_ltpred_copies():
    # ltpred keeps its own copies (it stays free of a phensim dependency), so
    # pin them where ltpred is installed (the family env; CI skips): same
    # birth years on simulated pedigrees -- phensim alone refuses
    # generation-skipping matings -- and the same Mendelian recursion.
    ltpred_sim = pytest.importorskip("ltpred.simulate")
    for seed in (3, 11):
        ped = simulate_pedigree(n_founder_pairs=30, gens=4, remarry=0.2, seed=seed)
        np.testing.assert_array_equal(
            pedigree_birth_times(*ped, base_birth_year=1900.0, generation_years=28.0),
            ltpred_sim.pedigree_birth_times(*ped, base_birth_year=1900.0, generation_years=28.0))
        z = np.random.default_rng(seed).standard_normal(len(ped[0]))
        (a, diag), (a_lt, diag_lt) = (mendelian_draw(*ped, innovations=z),
                                      ltpred_sim._mendelian_draw(*ped, z))
        np.testing.assert_array_equal(a, a_lt)
        np.testing.assert_array_equal(diag, diag_lt)


def test_pedigree_birth_times_generations():
    ids, father, mother = simulate_pedigree(n_founder_pairs=20, gens=3, seed=2)
    years = pedigree_birth_times(ids, father, mother)
    pos = {pid: i for i, pid in enumerate(ids)}
    for i, pid in enumerate(ids):
        for parent in (father[i], mother[i]):
            if parent is not None:
                assert years[i] == years[pos[parent]] + 30  # one generation later
    # the founder couples lead, and later-generation remarriage mates are
    # new founders placed in their partner's (later) generation
    assert (years[:40] == 1920.0).all()
    assert set(np.unique(years)) <= {1920.0, 1950.0, 1980.0, 2010.0}


def test_pedigree_birth_times_cycle_raises():
    ids = ["a", "b"]
    with pytest.raises(ValueError, match="ancestry cycle"):
        pedigree_birth_times(ids, father=["b", "a"], mother=[None, None])


def test_birth_times_distinguishes_generation_skipping_from_ancestry_cycle():
    # Uncle and niece mate: valid ancestry, incompatible equal-generation
    # co-parents. Their child has inbreeding F = 1/8.
    ids = ["gf", "gm", "uncle", "mother", "father", "niece", "child"]
    father = [None, None, "gf", "gf", None, "father", "uncle"]
    mother = [None, None, "gm", "gm", None, "mother", "niece"]
    A = kinship_from_pedigree(ids, father, mother)
    assert A[-1, -1] == 1.125
    _, diagonal = mendelian_draw(ids, father, mother, seed=1)
    np.testing.assert_allclose(diagonal, np.diag(A))
    with pytest.raises(ValueError, match="generation-skipping matings") as err:
        pedigree_birth_times(ids, father, mother)
    assert "cycle" not in str(err.value)


def _nuclear_family():
    """Father f, mother m, two full children c1/c2, half-sib h (f x m2)."""
    ids = ["f", "m", "m2", "c1", "c2", "h"]
    father = [None, None, None, "f", "f", "f"]
    mother = [None, None, None, "m", "m", "m2"]
    return ids, father, mother


def test_kinship_relationship_degrees():
    ids, father, mother = _nuclear_family()
    A = kinship_from_pedigree(ids, father, mother)
    np.testing.assert_allclose(np.diag(A)[:3], 1.0)  # non-inbred founders
    np.testing.assert_allclose(A[0, 3], 0.5)  # parent-offspring
    np.testing.assert_allclose(A[3, 4], 0.5)  # full sibs
    np.testing.assert_allclose(A[3, 5], 0.25)  # half sibs
    np.testing.assert_allclose(A[4, 5], 0.25)
    # first cousins: m and u are full siblings; c1 = (f, m), v = (founder, u)
    ids = ["gf", "gm", "f", "m", "u", "c1", "v"]
    father = [None, None, None, "gf", "gf", "f", None]
    mother = [None, None, None, "gm", "gm", "m", "u"]
    A = kinship_from_pedigree(ids, father, mother)
    np.testing.assert_allclose(A[3, 4], 0.5)  # m and u full sibs
    np.testing.assert_allclose(A[5, 6], 0.125)  # c1 vs v: first cousins
    np.testing.assert_allclose(A, A.T, atol=1e-14)


def test_kinship_inbreeding_and_validation():
    # child of full siblings: F = 0.25 -> A_ii = 1.25
    ids = ["gf", "gm", "f", "g", "inbred"]
    father = [None, None, "gf", "gf", "f"]
    mother = [None, None, "gm", "gm", "g"]
    A = kinship_from_pedigree(ids, father, mother)
    np.testing.assert_allclose(A[2, 3], 0.5)  # full sibs
    np.testing.assert_allclose(A[4, 4], 1.25)
    with pytest.raises(ValueError, match="unique"):
        kinship_from_pedigree(["a", "a"], [None, None], [None, None])
    with pytest.raises(ValueError, match="own parent"):
        kinship_from_pedigree(["a", "b"], ["a", None], [None, None])
    with pytest.raises(ValueError, match="cycle"):
        kinship_from_pedigree(["a", "b", "c"], ["b", None, None],
                              [None, "a", "b"])


def test_mendelian_draw_matches_dense_A():
    ids, father, mother = simulate_pedigree(n_founder_pairs=15, gens=3,
                                            remarry=0.15, seed=3)
    A = kinship_from_pedigree(ids, father, mother)
    rng = np.random.default_rng(4)
    reps = np.stack(
        [mendelian_draw(ids, father, mother, innovations=rng.standard_normal(len(ids)))[0]
         for _ in range(2000)]
    )
    emp = np.cov(reps.T)
    scale = np.sqrt(np.outer(np.diag(A), np.diag(A)))
    np.testing.assert_allclose(emp / scale, A / scale, atol=0.15)
    # diagonals come from the exact recursion, not estimation
    _, diagonal = mendelian_draw(ids, father, mother, seed=5)
    np.testing.assert_allclose(diagonal, np.diag(A), atol=1e-12)


def test_mendelian_draw_inbreeding_variance():
    ids = ["gf", "gm", "f", "g", "inbred"]
    father = [None, None, "gf", "gf", "f"]
    mother = [None, None, "gm", "gm", "g"]
    A = kinship_from_pedigree(ids, father, mother)
    rng = np.random.default_rng(6)
    reps = np.stack(
        [mendelian_draw(ids, father, mother, innovations=rng.standard_normal(5))[0]
         for _ in range(4000)]
    )
    np.testing.assert_allclose(reps.var(0)[4], A[4, 4], rtol=0.08)
