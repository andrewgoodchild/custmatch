"""Where does memory peak at ten million records? custmatch_scale.py's steps, sampled by step.

The steps of research/scripts/custmatch_scale.py for K disjoint copies of the synthetic file (load,
its timing pass over profiling, blocking and features, the labels, the run), with the process's
physical footprint sampled every 0.2 s and labelled by step and by the pipeline's log lines. The
footprint (proc_pid_rusage, macOS only) counts compressed pages, which resident memory does not,
so it holds steady when other work presses on the machine. --no-diagnostics skips the timing pass,
as a plain run would. Run with MallocLargeCache=0, as ncvr_memory.py's cache-off figures are.

    MallocLargeCache=0 python research/scripts/scale_memory.py 500 [--no-diagnostics]

Prints each step's peak and the lifetime peak; the results are in research/results/scale_large.json
("scale_10m_local_footprint").
"""
import ctypes
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custmatch_scale as CS  # noqa: E402
import custmatch_validate as V  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import block, features, profile_contacts  # noqa: E402

NAMES = ("user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
         "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time",
         "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
         "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default",
         "cpu_time_qos_maintenance", "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
         "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive", "billed_system_time",
         "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint")


class RUsageInfoV4(ctypes.Structure):                     # <sys/resource.h>, RUSAGE_INFO_V4 (first fields)
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in NAMES] + [("rest", ctypes.c_uint64 * 6)]


def footprint():
    """(current, lifetime peak) physical footprint of this process, GB."""
    r = RUsageInfoV4()
    ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(r))
    return r.phys_footprint / 2 ** 30, r.lifetime_max_phys_footprint / 2 ** 30


def main():
    k, diagnostics = int(sys.argv[1]), "--no-diagnostics" not in sys.argv
    step, peaks, done = ["start"], {}, threading.Event()

    def sample():
        while not done.is_set():
            peaks[step[0]] = max(peaks.get(step[0], 0), footprint()[0])
            time.sleep(0.2)
    threading.Thread(target=sample, daemon=True).start()

    def at(s):
        step[0] = s
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        at("scaled")
        cfg = CS.scaled(k, tmp)
        at("load")
        df, _ = io.load(cfg)
        if diagnostics:                                  # custmatch_scale.py's timing pass
            at("diag profile")
            prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
            at("diag block")
            pairs, _ = block(df, prof)
            at("diag features")
            features(df, pairs, prof)
        at("labels")
        lp = tmp / "labels.csv"
        labels.sample(cfg, 1000, lp, seed=1, stratified=False)
        lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
        gold = {f"{key.split(':', 1)[0]}:{key.split(':', 1)[1]}-x{c}": f"{s}-x{c}"
                for c in range(k) for key, (s, _) in V.truth().items()}
        lab["label"] = [int(gold[a] == gold[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(lp, index=False)
        t0 = time.time()
        at("run: start")
        pipeline.run(cfg, lp, tmp / "out", log=lambda *a: at("run: " + " ".join(map(str, a))[:70]))
        print("run seconds", round(time.time() - t0, 1))
    done.set()
    for s, g in peaks.items():
        print(f"{g:6.2f} GB  {s}")
    print(f"lifetime peak footprint {footprint()[1]:.2f} GB")


if __name__ == "__main__":
    main()
