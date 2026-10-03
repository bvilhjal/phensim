"""Built-in coalescent versus msprime: speed, peak memory and selected summaries.

phensim has two coalescent backends: the built-in Hudson engine
(``phensim._coalescent``, Numba-compiled when available) and msprime. This
script compares them on the same model:

* **Speed.** Wall-clock for the msprime ``sim_ancestry`` + ``sim_mutations`` +
  genotype-matrix pipeline versus the built-in backend, across sample size
  and segment length.
* **Peak memory.** Resident-set growth per backend and size, each measured in
  a fresh subprocess.
* **Selected summaries.** Segregating-site count, nucleotide diversity
  (``theta = 4*Ne*mu``) and the folded site-frequency spectrum, averaged over
  replicates. The report (``report/make_evidence.py``) adds LD decay with
  distance; neither establishes general equivalence.

Moved here from ldpred3's benchmark suite (2026-10-03), where it measured
this same kernel before it was extracted.

    python benchmarks/coalescent_backend.py            # speed, memory, summaries
    python benchmarks/coalescent_backend.py --csv coalescent_backend.csv

The largest memory cells (20,000 diploids) need a few GB under msprime;
pass ``--no-mem`` on a small machine.
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np

from phensim.genotypes import _coalescent_dosages


def self_current_rss_bytes():
    """This process's current resident memory in bytes (Linux or macOS)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, FileNotFoundError, IndexError, OSError, ValueError):
        pass
    if sys.platform != "darwin":
        raise RuntimeError("cannot read current process RSS on this platform")
    import ctypes

    class ProcTaskInfo(ctypes.Structure):
        _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
                    ("total_user", ctypes.c_uint64), ("total_system", ctypes.c_uint64),
                    ("threads_user", ctypes.c_uint64), ("threads_system", ctypes.c_uint64),
                    ("detail", ctypes.c_int32 * 12)]

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                     ctypes.c_void_p, ctypes.c_int]
    libproc.proc_pidinfo.restype = ctypes.c_int
    info = ProcTaskInfo()
    if libproc.proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(info),
                            ctypes.sizeof(info)) != ctypes.sizeof(info):  # PROC_PIDTASKINFO
        raise OSError("proc_pidinfo did not return PROC_PIDTASKINFO")
    return int(info.resident_size)


Ne, MU, REC = 10_000, 1e-8, 1e-8


# --------------------------------------------------------------------------- #
# Peak-RSS measurement. Peak resident memory is the only fair cross-backend
# number: msprime allocates its tree-sequence tables and haplotype matrix in C,
# the built-in backend allocates numpy buffers -- both show up in RSS but not in
# tracemalloc. Each (backend, size) runs in a fresh subprocess and a poller
# thread samples the process working set around the simulation call.
# --------------------------------------------------------------------------- #
def _measure_peak(backend, n, seq_len):
    peak = [0]
    base = self_current_rss_bytes()
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            r = self_current_rss_bytes()
            if r > peak[0]:
                peak[0] = r
            time.sleep(0.002)

    if backend == "numba":
        # Warm up the JIT *and* grow Numba's runtime memory pool at this exact
        # size (one discarded run), so we measure the steady-state per-simulation
        # working set, not a one-time pool-growth spike.
        _coalescent_dosages(n, seq_len, recomb_rate=REC, mut_rate=MU, Ne=Ne,
                            seed=1, backend="numba")
    import gc
    gc.collect()
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    dos, _ = _coalescent_dosages(n, seq_len, recomb_rate=REC, mut_rate=MU,
                                 Ne=Ne, seed=1, backend=backend)
    stop.set()
    t.join()
    return (peak[0] - base) / 1e6, dos.shape[1], dos.nbytes / 1e6


def _time_backend(backend, n, seq_len, seed):
    t = time.time()
    dos, af = _coalescent_dosages(n, seq_len, recomb_rate=REC, mut_rate=MU,
                                  Ne=Ne, seed=seed, backend=backend)
    return time.time() - t, dos.shape[1]


def run_speed(args):
    # Warm up the JIT so compilation doesn't pollute the first timed row.
    _coalescent_dosages(10, 1e5, recomb_rate=REC, mut_rate=MU, Ne=Ne, seed=1,
                        backend="numba")

    have_ms = True
    try:
        import msprime  # noqa: F401
    except ImportError:
        have_ms = False

    grid = [(500, 3e5), (1000, 1e6), (2000, 1e6), (5000, 1e6),
            (10_000, 1e6), (2000, 8e6)]
    print(f"Coalescent backend speed  (Ne={Ne}, mu={MU}, rec={REC})")
    head = (f"{'n':>7} {'seq(bp)':>9} | {'msprime(s)':>11} {'numba(s)':>9} "
            f"{'speedup':>8} | {'sites_ms':>8} {'sites_nb':>8}")
    print(head)
    print("-" * len(head))

    rows = []
    for n, seq in grid:
        t_nb, s_nb = _time_backend("numba", n, seq, seed=1)
        if have_ms:
            t_ms, s_ms = _time_backend("msprime", n, seq, seed=1)
            sp = f"{t_ms / t_nb:>7.2f}x"
        else:
            t_ms, s_ms, sp = float("nan"), 0, "     n/a"
        print(f"{n:>7} {seq:>9.0e} | {t_ms:>11.3f} {t_nb:>9.3f} {sp:>8} | "
              f"{s_ms:>8} {s_nb:>8}")
        rows.append({"n": n, "seq_len": seq, "t_msprime_s": round(t_ms, 4),
                     "t_numba_s": round(t_nb, 4), "sites_msprime": s_ms,
                     "sites_numba": s_nb})
    if not have_ms:
        print("\n(msprime not installed -- showing built-in backend only)")
    return rows


def _summ(dos):
    twoN = 2 * dos.shape[0]
    ac = dos.sum(0)
    p = ac / twoN
    pi = float(np.sum(2 * p * (1 - p) * twoN / (twoN - 1)))
    mac = np.minimum(ac, twoN - ac)
    sfs = np.bincount(mac, minlength=twoN // 2 + 1)[1:twoN // 2 + 1].astype(float)
    return dos.shape[1], pi, sfs


def run_memory(args):
    """Peak RSS per backend per size, each in its own subprocess."""
    have_ms = True
    try:
        import msprime  # noqa: F401
    except ImportError:
        have_ms = False

    # Start where the output is non-trivial: below ~1 MB output both backends
    # are dominated by fixed process / JIT-runtime overhead and the RSS delta is
    # noise. The point is the *slope* -- msprime's peak grows with the 2n-wide
    # haplotype matrix + tree-sequence tables, the built-in backend's with the
    # far smaller diploid dosage output.
    grid = [(2000, 1e6), (5000, 1e6), (10_000, 1e6), (2000, 8e6),
            (20_000, 1e6), (20_000, 4e6)]
    print(f"\nPeak memory (RSS delta)  (Ne={Ne}, mu={MU}, rec={REC})")
    head = (f"{'n':>7} {'seq(bp)':>9} | {'msprime(MB)':>12} {'numba(MB)':>10} "
            f"{'ratio':>6} | {'output(MB)':>10}")
    print(head)
    print("-" * len(head))
    rows = []
    for n, seq in grid:
        nb_mb, sites, out_mb = _run_mem_child("numba", n, seq)
        if have_ms:
            ms_mb, _, _ = _run_mem_child("msprime", n, seq)
            ratio = f"{ms_mb / nb_mb:>5.1f}x"
        else:
            ms_mb, ratio = float("nan"), "   n/a"
        print(f"{n:>7} {seq:>9.0e} | {ms_mb:>12.1f} {nb_mb:>10.1f} {ratio:>6} | "
              f"{out_mb:>10.1f}")
        rows.append({"n": n, "seq_len": seq, "peak_msprime_mb": round(ms_mb, 1),
                     "peak_numba_mb": round(nb_mb, 1), "output_mb": round(out_mb, 1)})
    if not have_ms:
        print("\n(msprime not installed -- showing built-in backend only)")
    return rows


def _run_mem_child(backend, n, seq_len):
    """Run one peak-RSS measurement in a clean subprocess; parse its one line."""
    env = dict(os.environ, COALESCENT_MEM_CHILD="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    out = subprocess.run([sys.executable, os.path.abspath(__file__),
                          backend, str(n), repr(seq_len)],
                         env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"memory child failed:\n{out.stderr}")
    mb, sites, out_mb = out.stdout.split()
    return float(mb), int(sites), float(out_mb)


def run_equivalence(args):
    try:
        import msprime  # noqa: F401
    except ImportError:
        print("msprime not installed -- skipping the equivalence check.")
        return

    n, L, reps = 60, 2e5, args.reps
    agg = {}
    for backend in ("msprime", "numba"):
        Ss, pis, sfs = [], [], None
        for r in range(reps):
            dos, _ = _coalescent_dosages(n, L, recomb_rate=REC, mut_rate=MU,
                                         Ne=Ne, seed=r + 1, backend=backend)
            S, pi, s = _summ(dos)
            Ss.append(S)
            pis.append(pi)
            sfs = s if sfs is None else sfs + s
        agg[backend] = (np.mean(Ss), np.mean(pis), sfs / sfs.sum())

    theta_L = 4 * Ne * MU * L
    print(f"\nSelected summary comparison  (n={n}, L={L:.0e}, {reps} reps, "
          f"theta*L={theta_L:.1f})")
    print(f"{'backend':>8} | {'seg.sites':>9} {'pi':>7} | folded SFS (first 6)")
    print("-" * 60)
    for backend in ("msprime", "numba"):
        S, pi, sfs = agg[backend]
        print(f"{backend:>8} | {S:>9.1f} {pi:>7.2f} | "
              + " ".join(f"{x:.3f}" for x in sfs[:6]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=30,
                    help="replicates for the equivalence check")
    ap.add_argument("--no-mem", action="store_true",
                    help="skip the (subprocess-based) peak-memory section")
    ap.add_argument("--csv", default=None, help="write speed rows to this CSV")
    args = ap.parse_args(argv)

    rows = run_speed(args)
    mem_rows = [] if args.no_mem else run_memory(args)
    run_equivalence(args)

    if args.csv:
        import csv
        mem_by_key = {(r["n"], r["seq_len"]): r for r in mem_rows}
        for r in rows:
            m = mem_by_key.get((r["n"], r["seq_len"]))
            if m:
                r["peak_msprime_mb"] = m["peak_msprime_mb"]
                r["peak_numba_mb"] = m["peak_numba_mb"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved results to {args.csv}")


if __name__ == "__main__":
    # Subprocess child mode for peak-RSS measurement: "<backend> <n> <seq_len>".
    if os.environ.get("COALESCENT_MEM_CHILD") == "1" and len(sys.argv) == 4:
        _backend, _n, _seq = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
        _mb, _sites, _out = _measure_peak(_backend, _n, _seq)
        print(f"{_mb:.1f} {_sites} {_out:.1f}")
        sys.exit(0)
    main()
