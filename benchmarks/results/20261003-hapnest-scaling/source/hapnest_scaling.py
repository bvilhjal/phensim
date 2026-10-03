"""Reproducible speed / resident-memory measurements; see hapnest_plan.md."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import phensim
from coalescent_backend import self_current_rss_bytes

ROOT = Path(__file__).resolve().parents[1]


def power_guard():
    if sys.platform == "darwin":
        if "AC Power" not in subprocess.check_output(["pmset", "-g", "batt"], text=True) or re.search(
                r"lowpowermode\s+1", subprocess.check_output(["pmset", "-g"], text=True)):
            raise RuntimeError("formal timings require AC power and Low Power Mode off")


def worker(args):
    data = np.load(args.out / "reference.npz")
    H, pops, cm, chrom = (data[k] for k in ["H", "pops", "cm", "chrom"])
    kw = dict(reference_populations=pops, sample_populations=np.arange(args.n) % 3,
              chromosome=chrom, seed=20261005 + args.rep, ne=10000, rho=.7,
              batch_size=256, backend=args.backend)
    ages = np.full(cm.size, 1000.)
    before = time.perf_counter()
    phensim.simulate_hapnest(H, 1, cm, ages, **dict(kw, sample_populations=[0]))
    warm = time.perf_counter() - before
    base = self_current_rss_bytes()
    peak, stop = [base], threading.Event()
    def poll():
        while not stop.wait(.002):
            peak[0] = max(peak[0], self_current_rss_bytes())
    monitor = threading.Thread(target=poll)
    monitor.start()
    before = time.perf_counter()
    digest = hashlib.sha256()
    if args.mode == "batches":
        for batch in phensim.iter_hapnest(H, args.n, cm, ages, **kw):
            digest.update(memoryview(batch))
    else:
        mmap_path = args.out / f"temporary-{os.getpid()}.bin"
        out = np.memmap(mmap_path, mode="w+", shape=(args.n, cm.size), dtype="int8") if args.mode == "memmap" else None
        G = phensim.simulate_hapnest(H, args.n, cm, ages, out=out, **kw)
        for start in range(0, args.n, 256):
            digest.update(memoryview(G[start:start+256]))
        if args.mode == "memmap":
            G.flush()
    elapsed = time.perf_counter() - before
    peak[0] = max(peak[0], self_current_rss_bytes())
    stop.set(); monitor.join()
    result = dict(n=args.n, m=cm.size, backend=args.backend, mode=args.mode, rep=args.rep,
                  wall_seconds=elapsed, first_call_seconds=warm, baseline_rss_bytes=base,
                  sampled_peak_rss_bytes=peak[0],
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024),
                  sha256=digest.hexdigest())
    (args.out / f"{args.n}-{args.backend}-{args.mode}-{args.rep}.json").write_text(json.dumps(result, indent=2)+"\n")
    if args.mode == "memmap":
        del G, out
        mmap_path.unlink()
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--backend", default="numba")
    ap.add_argument("--mode", default="batches")
    ap.add_argument("--rep", type=int, default=1)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.worker:
        worker(args); return
    power_guard()
    args.out.mkdir(parents=True, exist_ok=False)
    src = args.out / "source"
    shutil.copytree(ROOT / "phensim", src / "phensim", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(__file__, src)
    shutil.copy(Path(__file__).with_name("coalescent_backend.py"), src)
    shutil.copy(Path(__file__).with_name("hapnest_plan.md"), args.out / "plan.md")
    import numba
    meta = dict(python=sys.version, numpy=np.__version__, numba=numba.__version__,
                phensim=phensim.__version__, platform=platform.platform(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_sha256={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob("*.py")})
    (args.out / "environment.json").write_text(json.dumps(meta, indent=2)+"\n")
    H, pops = phensim.simulate_population_structure(600, 12000, fst=.05, model="balding-nichols",
                   block_sizes=[50]*240, rho=.8, phased=True, seed=20261005)
    np.savez_compressed(args.out / "reference.npz", H=H, pops=pops,
                         cm=np.tile(np.arange(2000)*.001, 6), chrom=np.repeat(np.arange(6), 2000))
    jobs = [(2000,"numpy","batches"), (2000,"numba","batches"),
            (10000,"numba","batches"), (50000,"numba","batches"),
            (50000,"numba","array"), (50000,"numba","memmap"),
            (100000,"numba","batches")]
    for rep in range(1,4):
        for n, backend, mode in jobs[rep-1:]+jobs[:rep-1]:
            power_guard()
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(args.out),
                            "--n", str(n), "--backend", backend, "--mode", mode, "--rep", str(rep)], check=True)
    rows = [json.loads(p.read_text()) for p in sorted(args.out.glob("[0-9]*.json"))]
    for n in [2000,50000]:
        for rep in range(1,4):
            assert len({r["sha256"] for r in rows if r["n"] == n and r["rep"] == rep}) == 1
    with open(args.out / "resources.csv", "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
