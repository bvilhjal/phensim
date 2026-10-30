"""PLINK encoding and external allele checks (PHENSIM_PLINK or PATH)."""

import os
import shutil
import subprocess

import numpy as np
import pytest

import phensim

from phensim.io import write_plink


@pytest.mark.external
@pytest.mark.parametrize("keep_order", [False, True])
def test_plink_recoding_and_named_allele_scores(tmp_path, keep_order):
    plink = (os.environ.get("PHENSIM_PLINK") or shutil.which("plink")
             or shutil.which("plink1.9") or shutil.which("plink19"))
    if plink is None:
        pytest.skip("PLINK 1.9 executable not available")

    # Five samples exercise BED padding; G is minor at one SNP and major
    # at another, so default PLINK loading changes only some A1/A2 labels.
    G = np.array([[0, 2, 2], [0, np.nan, 1], [1, 2, 0],
                  [0, 1, -1], [2, 2, 0]], dtype=float)
    called = np.where((G < 0) | np.isnan(G), np.nan, G)
    ids = [f"sample{i}" for i in range(len(G))]
    variants = [f"sim_1_{j + 1}" for j in range(G.shape[1])]
    prefix = tmp_path / "genotypes"
    write_plink(G, prefix, sample_ids=ids)
    alleles = tmp_path / "alleles.txt"
    alleles.write_text("".join(f"{variant} G\n" for variant in variants))
    weights = np.array([0.25, -0.5, 2.0])
    scores = tmp_path / "scores.txt"
    scores.write_text("".join(
        f"{variant} G {weight}\n" for variant, weight in zip(variants, weights)))

    def run_plink(name, *options):
        output = tmp_path / name
        command = [plink, "--bfile", str(prefix), "--out", str(output),
                   "--threads", "1", "--memory", "64"]
        if keep_order:
            command.append("--keep-allele-order")
        result = subprocess.run(command + list(options), capture_output=True,
                                text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return output

    def check_recoded(output, allele, expected):
        rows = output.with_suffix(".raw").read_text().splitlines()
        assert rows[0].split()[6:] == [f"{variant}_{allele}" for variant in variants]
        assert [line.split()[1] for line in rows[1:]] == ids
        actual = np.loadtxt(output.with_suffix(".raw"), dtype=str, skiprows=1)[:, 6:]
        actual = np.where(actual == "NA", "nan", actual).astype(float)
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    output = run_plink("recode_g", "--recode", "A", "--recode-allele", str(alleles))
    check_recoded(output, "G", called)
    if keep_order:
        output = run_plink("recode_a", "--recode", "A")
        check_recoded(output, "A", 2 - called)

    # --score uses the named G allele irrespective of internal A1/A2;
    # PLINK mean-imputes missing genotypes and 'sum' returns the dot product.
    output = run_plink("score_g", "--score", str(scores), "1", "2", "3", "sum")
    profile = np.genfromtxt(output.with_suffix(".profile"), names=True, dtype=None,
                            encoding="utf-8")
    assert profile["IID"].tolist() == ids
    expected = np.where(np.isnan(called), np.nanmean(called, axis=0), called) @ weights
    np.testing.assert_allclose(profile["SCORESUM"], expected, rtol=1e-5)


def _decode_bed_bytes(bed, n, m):
    """Independent SNP-major BED decoder; also checks zero padding bits."""
    assert bed[:3] == b"\x6c\x1b\x01"
    stride = (n + 3) // 4
    assert len(bed) == 3 + stride * m
    # Counting BIM allele 2: 00,01,10,11.
    lookup = np.array([0, -1, 1, 2])
    decoded = np.empty((n, m))
    for j in range(m):
        for i in range(n):
            code = (bed[3 + j * stride + i // 4] >> (2 * (i % 4))) & 3
            decoded[i, j] = lookup[code]
        if n % 4:
            assert bed[3 + (j + 1) * stride - 1] >> (2 * (n % 4)) == 0
    return decoded


@pytest.mark.parametrize("n", range(1, 10))
@pytest.mark.parametrize("dtype", ["int8", "float64"])
def test_write_plink_byte_oracle(tmp_path, n, dtype):
    # All four states and every padding width; also covers the former
    # test_plink_decodes_all_four_states_and_padding (n = 7, float64).
    m = 6
    states = np.tile(np.array([0, 1, 2, -1]), n * m // 4 + 1)[:n * m]
    G = states.reshape(n, m).astype(dtype)
    if dtype == "float64":
        G[n - 1, m - 1] = np.nan  # NaN missing alongside the -1 codes
    prefix = str(tmp_path / "oracle")
    phensim.write_plink(G, prefix)
    decoded = _decode_bed_bytes((tmp_path / "oracle.bed").read_bytes(), n, m)
    np.testing.assert_array_equal(
        decoded, np.where(np.isnan(G) | (G < 0), -1, G))


def test_tiled_bed_payload_matches_full_encoding(tmp_path):
    from phensim.io import _encode_bed, _plink_genotypes
    G = np.random.default_rng(15).integers(-1, 3, (13, 77)).astype(float)
    G[2, 3] = np.nan
    checked, missing = _plink_genotypes(G)
    expected = b"\x6c\x1b\x01" + _encode_bed(missing, checked, *G.shape)
    for block in [1, 11, 200]:
        prefix = tmp_path / f"tile{block}"
        phensim.write_plink(G, prefix, block_size=block)
        assert prefix.with_suffix(".bed").read_bytes() == expected
    G[-1, -1] = .5
    with pytest.raises(ValueError):
        phensim.write_plink(G, tmp_path / "invalid", block_size=1)
    assert not (tmp_path / "invalid.bed").exists()


@pytest.mark.parametrize("bad", [0.5, 3, np.inf])
def test_plink_rejects_unrepresentable_calls_before_writing(tmp_path, bad):
    with pytest.raises(ValueError, match="genotypes"):
        phensim.write_plink(np.array([[bad]]), str(tmp_path / "bad"))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("meta", [
    dict(sample_ids=["a", "b", "c"]),           # wrong count
    dict(sample_ids=["a", "a", "b", "c"]),      # duplicates
    dict(sample_ids=["a ", "b", "c", "d"]),     # whitespace
    dict(sample_ids=["a\tb", "b", "c", "d"]),   # control/whitespace
    dict(sample_ids=["", "b", "c", "d"]),       # empty
    dict(sample_ids=["a", None, "c", "d"]),     # None
    dict(chromosome=np.array([22.5, 1.0, 1.0])),
    dict(chromosome=np.array([-1, 1, 1])),
    dict(chromosome=np.array([np.inf, 1, 1])),
    dict(chromosome=["2 2", "1", "1"]),         # whitespace token
    dict(chromosome=np.array([[1, 1, 1]])),     # not 1-D
    dict(chromosome=np.array([1, 1])),          # wrong length
    dict(position=np.array([-1, 5, 9])),
    dict(position=np.array([1.5, 5, 9])),
    dict(position=np.array([np.inf, 5, 9])),
    dict(position=np.array([1, 5])),            # wrong length
    dict(position=[True, True, False]),         # bool
])
def test_write_plink_rejects_bad_metadata(tmp_path, meta):
    G = np.tile(np.array([0, 1, 2]), (4, 1))  # (4, 3) valid dosages
    # Sentinels at the *requested* prefix: a bad call must not create or
    # truncate any of the three outputs.
    for suffix in (".bed", ".bim", ".fam"):
        (tmp_path / f"bad{suffix}").write_bytes(b"sentinel")
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(ValueError):
        phensim.write_plink(G, str(tmp_path / "bad"), **meta)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    for suffix in (".bed", ".bim", ".fam"):
        assert (tmp_path / f"bad{suffix}").read_bytes() == b"sentinel"


def test_write_plink_preserves_large_integer_positions(tmp_path):
    G = np.tile(np.array([0, 1, 2]), (4, 1))
    positions = np.array([2**53 + 1, 2**62 + 7, 9], dtype=np.int64)
    phensim.write_plink(G, str(tmp_path / "big"), position=positions)
    rows = [line.split() for line in (tmp_path / "big.bim").read_text().splitlines()]
    assert [int(row[3]) for row in rows] == [2**53 + 1, 2**62 + 7, 9]


@pytest.mark.parametrize("position", [
    np.array([2**63, 1, 1], dtype=np.uint64),   # does not fit int64
    np.array([2**64 - 1, 1, 1], dtype=np.uint64),
])
def test_write_plink_rejects_overflowing_positions(tmp_path, position):
    G = np.tile(np.array([0, 1, 2]), (4, 1))
    with pytest.raises(ValueError, match="base-pair"):
        phensim.write_plink(G, str(tmp_path / "bad"), position=position)
    assert list(tmp_path.iterdir()) == []


def test_write_plink_accepts_labels_and_integral_floats(tmp_path):
    G = np.tile(np.array([0, 1, 2, 0]), (4, 1))  # (4, 4)
    prefix = str(tmp_path / "labels")
    phensim.write_plink(
        G, prefix,
        chromosome=["X", "MT", 22.0, 0],
        position=np.array([0.0, 5, 9, 12]),
        sample_ids=["s1", "s2", "s3", "s4"])
    bim = [line.split() for line in (tmp_path / "labels.bim").read_text().splitlines()]
    assert [row[0] for row in bim] == ["X", "MT", "22", "0"]
    assert [row[3] for row in bim] == ["0", "5", "9", "12"]
    fam = [line.split() for line in (tmp_path / "labels.fam").read_text().splitlines()]
    assert [row[1] for row in fam] == ["s1", "s2", "s3", "s4"]
