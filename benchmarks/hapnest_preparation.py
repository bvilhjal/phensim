"""Dense-oracle versus tiled phenotype/PLINK preparation after HAPNEST."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import phensim
from coalescent_backend import self_current_rss_bytes
from hapnest_scaling import power_guard


def worker(args):
    ref = np.load(args.reference)
    G = phensim.simulate_hapnest(ref["H"], 10000, ref["cm"], np.full(12000, 1000.),
          reference_populations=ref["pops"], sample_populations=np.arange(10000)%3,
          chromosome=ref["chrom"], seed=3400+args.rep)
    base, stop = self_current_rss_bytes(), threading.Event()
    peak = [base]
    def poll():
        while not stop.wait(.002):
            peak[0] = max(peak[0], self_current_rss_bytes())
    monitor = threading.Thread(target=poll); monitor.start()
    name = f"{args.mode}-{args.rep}"
    before = time.perf_counter()
    if args.mode.startswith("trait"):
        tr = phensim.simulate_trait(G, seed=73,
                 genotype_block_size=128 if args.mode.endswith("tiled") else None)
    else:
        prefix = args.out/name
        if args.mode.endswith("tiled"):
            phensim.write_plink(G, prefix)
        else:
            # Prior whole-payload implementation; metadata cost is omitted,
            # making this a conservative baseline for the full tiled writer.
            from phensim.io import _plink_genotypes, _encode_bed
            checked, missing = _plink_genotypes(G)
            payload = _encode_bed(missing, checked, *G.shape)
            prefix.with_suffix(".bed").write_bytes(b"\x6c\x1b\x01"+payload)
    elapsed = time.perf_counter()-before
    peak[0] = max(peak[0], self_current_rss_bytes())
    stop.set(); monitor.join()
    result = dict(mode=args.mode, rep=args.rep, n=10000, m=12000,
                  wall_seconds=elapsed, baseline_rss_bytes=base, sampled_peak_rss_bytes=peak[0],
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024))
    if args.mode.startswith("trait"):
        np.savez(args.out/f"{name}.npz", **tr)
    else:
        result["bed_sha256"] = hashlib.sha256(prefix.with_suffix(".bed").read_bytes()).hexdigest()
        for path in args.out.glob(name+".*"):
            path.unlink()
    (args.out/f"{name}.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode")
    ap.add_argument("--rep", type=int)
    args = ap.parse_args()
    args.out, args.reference = args.out.resolve(), args.reference.resolve()
    if args.worker:
        worker(args); return
    power_guard(); args.out.mkdir(parents=True, exist_ok=False)
    src = args.out/"source"
    shutil.copytree(Path(phensim.__file__).parent, src/"phensim", ignore=shutil.ignore_patterns("__pycache__"))
    for name in [Path(__file__).name,"hapnest_scaling.py","coalescent_backend.py"]:
        shutil.copy(Path(__file__).with_name(name),src/name)
    shutil.copy(Path(__file__).with_name("hapnest_plan.md"),args.out/"plan.md")
    (args.out/"inputs.json").write_text(json.dumps(dict(
        reference=str(args.reference), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob("*.py")}),indent=2)+"\n")
    modes = ["trait-dense","trait-tiled","plink-dense","plink-tiled"]
    for rep in range(1,4):
        for mode in modes[rep-1:]+modes[:rep-1]:
            power_guard()
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--out",str(args.out),
                "--reference",str(args.reference),"--worker","--mode",mode,"--rep",str(rep)],check=True)
        a,b = [np.load(args.out/f"trait-{mode}-{rep}.npz") for mode in ("dense","tiled")]
        for key in a.files:
            np.testing.assert_allclose(a[key],b[key],rtol=2e-12,atol=2e-12)
        hashes = [json.loads((args.out/f"plink-{mode}-{rep}.json").read_text())["bed_sha256"] for mode in ("dense","tiled")]
        assert hashes[0] == hashes[1]
    (args.out/"validation.json").write_text(json.dumps(dict(
        phenotype_oracle="all components agree within 2e-12 across three seeds",
        plink_oracle="identical BED hashes across three seeds"),indent=2)+"\n")


if __name__ == "__main__":
    main()
