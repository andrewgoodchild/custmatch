"""How far does custmatch scale? 2, 5 and 10 million records, and the full NCVR benchmark, on Modal.

  scale_2m / 5m / 10m   scripts/custmatch_scale.py with 100 / 250 / 500 disjoint copies of the
                        synthetic file: time per stage and accuracy as the file grows. Copies get
                        copy-specific surnames and contacts, so blocks do not grow as in a real
                        file - this measures the machinery, not harder matching.
  ncvr_all              scripts/ncvr_leipzig.py on all five sources (5,000,000 records, 3,500,840
                        people; real names, so blocks grow as they would), one label draw.

Each job also reports the peak memory of the run (the largest child process). One container per
job, sized for 10 million records; nothing is kept on Modal.

    modal run scripts/modal_scale.py [--only scale_10m]

Writes results/scale_large.json.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/scale_large.json"
JOBS = {
    "scale_2m": (["scripts/custmatch_scale.py", "100"], "custmatch_scale.json"),
    "scale_5m": (["scripts/custmatch_scale.py", "250"], "custmatch_scale.json"),
    "scale_10m": (["scripts/custmatch_scale.py", "500"], "custmatch_scale.json"),
    "ncvr_all": (["scripts/ncvr_leipzig.py", "--people", "3500840", "--seeds", "1",
                  "--variants", "postcode_initial_keys"], "ncvr_leipzig.json"),
}

app = modal.App("custmatch-scale")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "httpx", "certifi", "pyarrow",
                      "phonenumbers", "anyascii", "networkx")
         .env({"PYTHONPATH": "/repo:/repo/scripts"}))
for d in ("custmatch", "scripts", "examples", "results"):
    image = image.add_local_dir(ROOT / d, f"/repo/{d}", ignore=["**/__pycache__"])
for d in ("customer", "ncvr_leipzig"):
    image = image.add_local_dir(ROOT / "data" / d, f"/repo/data/{d}")


@app.function(image=image, cpu=16, memory=131072, timeout=8 * 3600)
def run(name, cmd, results_file):
    import resource
    import subprocess
    import time
    path = Path("/repo/results") / results_file
    before = json.loads(path.read_text()) if path.exists() else None
    t0 = time.time()
    p = subprocess.run(["python", *cmd], cwd="/repo", capture_output=True, text=True)
    out = {"command": " ".join(cmd), "wall_seconds": round(time.time() - t0, 1),
           "peak_memory_gb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2, 2),
           "machine": "Modal, 16 cores, 128 GiB", "log_tail": (p.stdout + p.stderr).strip().splitlines()[-12:]}
    if p.returncode != 0:
        return name, out | {"failed": p.returncode}
    after = json.loads(path.read_text())
    if isinstance(after, list):                # custmatch_scale.json: one entry per size run
        out["result"] = after[-1]
    else:                                      # ncvr_leipzig.json: the entry this run wrote
        out["result"] = {k: v for k, v in after.items() if (before or {}).get(k) != v}
    return name, out


@app.local_entrypoint()
def main(only: str = ""):
    names = only.split(",") if only else list(JOBS)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, out in run.starmap([(n, *JOBS[n]) for n in names]):
        res[name] = out
        OUT.write_text(json.dumps(res, indent=1, default=float))
        print(f"=== {name}: {'FAILED' if 'failed' in out else 'ok'}  wall {out['wall_seconds']} s  "
              f"peak {out['peak_memory_gb']} GB", flush=True)
        print("\n".join(out["log_tail"][-4:]), flush=True)
