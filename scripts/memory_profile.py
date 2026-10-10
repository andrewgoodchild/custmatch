"""Where does custmatch's memory go? Peak memory per pipeline stage on an NCVR sample.

A thread samples the process's resident memory every 50 ms while `custmatch run` works; the
pipeline's stage messages mark where each stage starts, so each stage's peak is reported along
with the memory before the run (the loaded data) and the candidate pairs.

    python scripts/memory_profile.py [--people 200000]
"""
import argparse
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncvr_leipzig as N  # noqa: E402
from custmatch import labels, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=200000)
    a = ap.parse_args()
    proc = psutil.Process()
    samples, marks, stop = [], [("start", time.time())], threading.Event()

    def sample():
        while not stop.is_set():
            samples.append((time.time(), proc.memory_info().rss))
            time.sleep(0.05)
    sub = N.sample(N.load(), a.people, 1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = dict(N.VARIANTS["postcode_initial_keys"], sources=[])
        for s, g in sub.groupby("source"):
            g = g.reset_index(drop=True).assign(rid=lambda d: d.recid + "_" + d.index.astype(str))
            g.to_parquet(tmp / f"{s}.parquet", index=False)
            cfg["sources"].append({"name": s, "path": f"{s}.parquet", "columns": {
                "record_id": "rid", "first_name": "givenname", "last_name": "surname",
                "town": "suburb", "postcode": "postcode"}})
        cp = tmp / "config.json"
        cp.write_text(json.dumps(cfg))
        lp = tmp / "labels.csv"
        labels.sample(cp, 1000, lp, seed=1)
        lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
        person = lambda k: k.split(":")[1].rsplit("_", 1)[0]  # noqa: E731
        lab["label"] = [int(person(x) == person(y)) for x, y in zip(lab.a, lab.b)]
        lab.to_csv(lp, index=False)
        del sub
        base = proc.memory_info().rss
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        t0 = time.time()
        s = pipeline.run(cp, lp, tmp / "out", seed=1, log=lambda m: marks.append((str(m)[:60], time.time())))
        stop.set()
        th.join()
    marks.append(("end", time.time()))
    gb = 1024 ** 3
    print(f"records {s['records']:,}  candidate pairs {s['candidate_pairs']:,}  run {time.time() - t0:.0f} s")
    print(f"memory before the run (python, libraries, labels): {base / gb:.2f} GB")
    peak = max(m for _, m in samples)
    print(f"peak during the run: {peak / gb:.2f} GB  ({(peak - base) / s['candidate_pairs']:.0f} bytes per candidate pair above the baseline)")
    for (name, t), (_, t_next) in zip(marks, marks[1:]):
        inside = [m for ts, m in samples if t <= ts < t_next]
        if inside:
            print(f"  {max(inside) / gb:6.2f} GB peak  {t_next - t:7.1f} s  until: {name}")


if __name__ == "__main__":
    main()
