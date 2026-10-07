"""External PLINK 1.9 allele checks; set PHENSIM_PLINK or put plink on PATH."""

import os
import shutil
import subprocess

import numpy as np
import pytest

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
