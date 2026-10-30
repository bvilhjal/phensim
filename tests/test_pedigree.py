"""Pedigree simulator and pedigree-kinship tests."""

import hashlib
import warnings

import numpy as np
import pytest

import phensim

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


def test_simulate_pedigree_draws_are_frozen():
    # ltpred's simulate_pedigree delegates here, and its committed register
    # benchmarks were drawn by the original ltpred code this reproduces.
    ped = simulate_pedigree(n_founder_pairs=25, gens=3, remarry=0.15, seed=11)
    assert len(ped[0]) == 500
    assert hashlib.sha256(repr(ped).encode()).hexdigest()[:16] == "27390326dc87ff51"


def test_pedigree_birth_times_match_ltpred_copy():
    # ltpred keeps its own permissive copy for real pedigrees (it places
    # generation-skipping matings, which phensim refuses). Pin the two to the
    # same years on simulated pedigrees where ltpred is installed (CI skips).
    ltpred_sim = pytest.importorskip("ltpred.simulate")
    for seed in (3, 11):
        ped = simulate_pedigree(n_founder_pairs=30, gens=4, remarry=0.2, seed=seed)
        np.testing.assert_array_equal(
            pedigree_birth_times(*ped, base_birth_year=1900.0, generation_years=28.0),
            ltpred_sim.pedigree_birth_times(*ped, base_birth_year=1900.0, generation_years=28.0))


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


def test_birth_times_allow_founders_to_enter_later_generations():
    ids = ["gf", "gm", "dad", "late", "c2", "c1", "kid", "isolated"]
    father = [None, None, "gf", None, "dad", "late", "c1", None]
    mother = [None, None, "gm", None, None, None, "c2", None]
    expected = 1920 + 30 * np.array([0, 0, 1, 1, 2, 2, 3, 0])
    np.testing.assert_array_equal(pedigree_birth_times(ids, father, mother), expected)
    order = [6, 3, 7, 5, 0, 4, 2, 1]
    np.testing.assert_array_equal(pedigree_birth_times(
        [ids[i] for i in order], [father[i] for i in order], [mother[i] for i in order]), expected[order])


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
    assert (np.diag(A) > 1).any()  # inbred people are exercised
    # the draw is linear in the innovations, so unit innovations give the
    # factor F of a ~ N(0, A) exactly: F F' must equal A, not just estimate it
    n = len(ids)
    F = np.column_stack([mendelian_draw(ids, father, mother, innovations=e)[0]
                         for e in np.eye(n)])
    np.testing.assert_allclose(F @ F.T, A, rtol=0, atol=1e-12)
    # diagonals come from the exact recursion, not estimation
    _, diagonal = mendelian_draw(ids, father, mother, seed=5)
    np.testing.assert_allclose(diagonal, np.diag(A), atol=1e-12)
    # seeded values are frozen: ltpred's register simulation draws through here
    a, d = mendelian_draw(ids, father, mother,
                          innovations=np.random.default_rng(3).standard_normal(n))
    assert hashlib.sha256(a.tobytes() + d.tobytes()).hexdigest()[:16] == "76f610d85fe50826"


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


def test_birth_times_reject_self_parent_and_incompatible_generations():
    with pytest.raises(ValueError, match="own parent"):
        phensim.pedigree_birth_times(["a"], ["a"], [None])
    # a parents b; a and b are also co-parents of c. Union-find must not
    # erase the impossible a -> b edge when it merges the co-parents.
    with pytest.raises(ValueError, match="discrete generations"):
        phensim.pedigree_birth_times(["a", "b", "c"], [None, "a", "a"], [None, None, "b"])


def test_pedigree_accepts_zero_as_a_listed_id():
    A = phensim.kinship_from_pedigree([0, 1, 2], [None, None, 0], [None, None, 1])
    np.testing.assert_allclose(A[2, :2], 0.5)
    A = phensim.kinship_from_pedigree(["0", "1", "2"], [None, None, "0"], [None, None, "1"])
    np.testing.assert_allclose(A[2], [0.5, 0.5, 1])
    for missing in (None, "", np.nan):
        with pytest.raises(ValueError, match="missing"):
            phensim.kinship_from_pedigree(["a", missing], [None, None], [None, None])


def _typo_pedigree():
    """``'ax'`` is an unlisted father reference for child ``'c'``."""
    ids = ["a", "b", "c"]
    return ids, [None, None, "ax"], [None, None, "b"]


def test_unlisted_parent_warns_once_with_count():
    ids, father, mother = _typo_pedigree()
    with pytest.warns(UserWarning) as record:
        A = kinship_from_pedigree(ids, father, mother)
    assert len(record) == 1
    message = str(record[0].message)
    assert "1" in message and "unlisted parent" in message
    assert "unknown founders" in message
    np.testing.assert_allclose(A[1, 2], 0.5)  # c is still b's child


def test_unlisted_parent_warns_in_all_parent_consumers():
    ids, father, mother = _typo_pedigree()
    consumers = [
        lambda: kinship_from_pedigree(ids, father, mother),
        lambda: mendelian_draw(ids, father, mother, seed=1),
        lambda: pedigree_birth_times(ids, father, mother),
    ]
    for consume in consumers:
        with pytest.warns(UserWarning, match="unlisted parent"):
            consume()


def test_listed_parents_do_not_warn():
    ids = ["a", "b", "c"]
    father, mother = [None, None, "a"], [None, None, "b"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A = kinship_from_pedigree(ids, father, mother)
        mendelian_draw(ids, father, mother, seed=1)
        pedigree_birth_times(ids, father, mother)
    np.testing.assert_allclose(A[0, 2], 0.5)  # a is c's father


@pytest.mark.parametrize("marker", [None, 0, "0", "", np.nan])
def test_missing_parent_sentinels_do_not_warn(marker):
    ids = ["a", "b"]
    father, mother = [marker, None], [None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kinship_from_pedigree(ids, father, mother)
        mendelian_draw(ids, father, mother, seed=1)
        pedigree_birth_times(ids, father, mother)


def test_unlisted_parents_keep_external_parent_semantics():
    ids = ["f", "c", "c2"]
    with pytest.warns(UserWarning, match="unlisted parent"):
        A = kinship_from_pedigree(ids, [None, "fx", "f"], [None, None, "m"])
    # 'fx' and 'm' are unlisted: c is a founder, c2 has one known parent.
    expected = np.array([[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])
    np.testing.assert_allclose(A, expected)


def test_mendelian_innovations_must_be_finite_vector():
    ids = ["a", "b", "c"]
    with pytest.raises(ValueError, match="innovations"):
        mendelian_draw(ids, [None] * 3, [None] * 3,
                       innovations=np.ones((3, 1)))
    with pytest.raises(ValueError, match="innovations"):
        mendelian_draw(ids, [None] * 3, [None] * 3,
                       innovations=[1.0, np.inf, 0.0])
    with pytest.raises(ValueError, match="innovations"):
        mendelian_draw(ids, [None] * 3, [None] * 3,
                       innovations=np.ones(4))


@pytest.mark.parametrize("kwargs", [
    dict(n_founder_pairs=0), dict(n_founder_pairs=-1),
    dict(n_founder_pairs=2.5), dict(n_founder_pairs=True),
    dict(gens=-1), dict(gens=1.5), dict(gens=True),
    dict(remarry=-0.1), dict(remarry=1.5), dict(remarry=np.nan),
])
def test_simulate_pedigree_validates_geometry(kwargs):
    with pytest.raises(ValueError):
        simulate_pedigree(**kwargs)


def test_simulate_pedigree_zero_gens_is_founders_only():
    ids, father, mother = simulate_pedigree(
        n_founder_pairs=2, gens=0, seed=1)
    assert ids == ["p0", "p1", "p2", "p3"]
    assert father == [None] * 4 and mother == [None] * 4
