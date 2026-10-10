"""Rerun experiments on Modal (modal.com), one container per command, in parallel.

Each job runs one experiment command in a copy of the repo (code, examples, results and the
generated data it reads) and returns the top-level entries of the results file it changed; they
are merged into the local file, so parallel jobs that write different entries of one file do not
overwrite each other. A job marked "replace" starts from an empty results file and returns all of
it (for files whose old entries must not survive, e.g. removed methods).

    modal run research/scripts/modal_rerun.py                      # the country-profile set below
    modal run research/scripts/modal_rerun.py --only cookies,scenarios

The jobs are the published experiments that set a country profile, rerun after the Wikidata name
list was removed in 0.1.0. Data uploaded: the generated customer files (ours) and the Leipzig
NCVR benchmark (CC BY 4.0), both from the gitignored data/ folder; nothing is kept on Modal.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
DATA = ["customer", "names", "au_customer", "au_customer_s1", "au_customer_s2", "au_customer_noshared",
        "au_customer_script", "au_customer_dated", "au_customer_cdp", "au_customer_s1_cdp",
        "au_customer_s2_cdp", "au_customer_noshared_cdp", "ncvr_leipzig"]
NCVR20 = "default_keys,postcode_initial_keys,postcode_initial_keys_unique_sources,lsh_blocking,lsh_blocking_unique_sources"

# name: (results file, command, replace?)
JOBS = {
    "cdp_comparison": ("cdp_comparison.json", ["research/scripts/cdp_comparison.py"], False),
    **{f"reassess_{d}": ("cdp_reassess.json", ["research/scripts/cdp_reassess.py", "--seeds", "1", "--data", d], False)
       for d in ("au_customer", "au_customer_s1", "au_customer_s2", "au_customer_noshared")},
    **{f"separation_{d}": ("separation.json", ["research/scripts/separation_research.py", "--seeds", "1", "--data", d], False)
       for d in ("au_customer", "au_customer_s1", "au_customer_s2")},
    "cookies": ("cookies.json", ["research/scripts/cookie_research.py"], False),
    "record_dates": ("record_dates.json", ["research/scripts/record_dates_research.py", "--data-seeds", "1,2,3"], False),
    "scenarios": ("scenarios.json", ["research/scripts/scenarios.py", "score"], False),
    "order_dependence": ("order_dependence.json", ["research/scripts/order_dependence.py", "--seeds", "1,2"], True),
    "order_dependence_couples": ("order_dependence.json",
                                 ["research/scripts/order_dependence.py", "--couples", "--methods",
                                  "rerun,rerun_fixed_ids,rerun_hyst05,add_fixed", "--seeds", "1,2"], False),
    **{f"ncvr20k_{v}": ("ncvr_leipzig.json", ["research/scripts/ncvr_leipzig.py", "--people", "20000", "--seeds", "1,2,3",
                                              "--variants", v], False) for v in NCVR20.split(",")},
    "ncvr200k": ("ncvr_leipzig.json", ["research/scripts/ncvr_leipzig.py", "--people", "200000", "--seeds", "1,2,3",
                                       "--variants", "postcode_initial_keys", "--splink"], False),
    **{f"ncvr200k_{v}": ("ncvr_leipzig.json", ["research/scripts/ncvr_leipzig.py", "--people", "200000", "--seeds", "1",
                                               "--variants", v], False)
       for v in ("default_keys", "postcode_initial_keys_unique_sources", "lsh_blocking", "lsh_blocking_unique_sources")},
}
# the couples run adds entries to the file the first order_dependence job rebuilds: run it after
AFTER = {"order_dependence_couples": "order_dependence"}

app = modal.App("custmatch-rerun")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "httpx", "certifi", "pyarrow",
                      "phonenumbers", "anyascii", "networkx", "splink==4.0.17", "pseudopeople")
         .env({"PYTHONPATH": "/repo:/repo/research/scripts"}))
for d in ("custmatch", "research/scripts", "examples", "research/results"):
    image = image.add_local_dir(ROOT / d, f"/repo/{d}", ignore=["**/__pycache__"])
for d in DATA:
    if (ROOT / "data" / d).exists():
        image = image.add_local_dir(ROOT / "data" / d, f"/repo/data/{d}")


@app.function(image=image, cpu=8, memory=32768, timeout=6 * 3600)
def run(name, results_file, cmd, replace, base=None):
    """Run one experiment command; return (name, changed top-level entries, whole file?, log tail)."""
    import subprocess
    path = Path("/repo/research/results") / results_file
    if replace:
        path.unlink(missing_ok=True)
    if base is not None:                       # entries another job just wrote, to build on
        path.write_text(json.dumps(base))
    before = json.loads(path.read_text()) if path.exists() else {}
    p = subprocess.run(["python", *cmd], cwd="/repo", capture_output=True, text=True)
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
    if p.returncode != 0:
        return name, None, replace, f"FAILED ({p.returncode})\n{tail}"
    after = json.loads(path.read_text()) if path.exists() else {}
    if isinstance(after, list) or replace:
        return name, after, True, tail
    return name, {k: v for k, v in after.items() if before.get(k) != v}, False, tail


def _merge(results_file, entries, whole):
    path = ROOT / "research/results" / results_file
    if whole:
        path.write_text(json.dumps(entries, indent=1, default=float))
        return
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(entries)
    path.write_text(json.dumps(cur, indent=1, default=float))


@app.local_entrypoint()
def main(only: str = ""):
    names = only.split(",") if only else list(JOBS)
    first = [n for n in names if n not in AFTER]
    later = [n for n in names if n in AFTER]
    done = {}
    for name, entries, whole, tail in run.starmap([(n, *JOBS[n]) for n in first]):
        print(f"=== {name}\n{tail}\n", flush=True)
        if entries is not None:
            _merge(JOBS[name][0], entries, whole)
            done[name] = entries
    for n in later:                            # builds on the file its predecessor rebuilt
        base = json.loads((ROOT / "research/results" / JOBS[n][0]).read_text())
        name, entries, whole, tail = run.remote(n, *JOBS[n], base=base)
        print(f"=== {name}\n{tail}\n", flush=True)
        if entries is not None:
            _merge(JOBS[name][0], entries, whole)
    failed = [n for n in names if n not in done and n not in later]
    print("failed:", failed or "none")
