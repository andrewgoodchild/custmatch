"""Was removing the Wikidata name list right? A controlled A/B on Modal.

Runs the code from just before the removal (commit 8c117bc, checked out at OLD) twice: with the
cross-language name list (name_variants.csv) in custmatch's data folder, and without it. Same
code, so the list is the only difference. Experiments: the CDP reassessment on four generated
populations and the sparse Australian validation, three label draws each.

    git worktree add /tmp/custmatch_8c117bc 8c117bc
    OLD_TREE=/tmp/custmatch_8c117bc NAME_LIST=~/.cache/custmatch/name_variants.csv modal run research/scripts/modal_name_list_ab.py

(name_variants.csv was built by research/scripts/build_name_variants.py, removed in 0.1.0; check out 8c117bc
to rebuild it.) Result: no measurable difference - see docs/report/11-adopted-and-dropped.md "Removed in 0.1.0".

Writes research/results/name_list_ab.json.
"""
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/name_list_ab.json"
POPS = ("au_customer", "au_customer_s1", "au_customer_s2", "au_customer_noshared")
app = modal.App("custmatch-name-list-ab")


def image_for(old, name_list):
    img = (modal.Image.debian_slim(python_version="3.12")
           .pip_install("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "httpx", "certifi", "pyarrow",
                        "phonenumbers", "anyascii", "networkx", "duckdb", "jellyfish")
           .env({"PYTHONPATH": "/repo:/repo/research/scripts", "CUSTMATCH_DATA": "/custmatch_data"}))
    for d in ("custmatch", "research/scripts", "examples", "research/results"):
        img = img.add_local_dir(Path(old) / d, f"/repo/{d}", ignore=["**/__pycache__"])
    for d in ("au_customer", *POPS[1:], *[p + "_cdp" for p in POPS], "customer"):
        if (ROOT / "data" / d).exists():
            img = img.add_local_dir(ROOT / "data" / d, f"/repo/data/{d}")
    return img.add_local_file(name_list, "/name_list/name_variants.csv")


OLD = os.environ.get("OLD_TREE", "")
LIST = os.path.expanduser(os.environ.get("NAME_LIST", "~/.cache/custmatch/name_variants.csv"))


@app.function(image=image_for(OLD, LIST), cpu=8, memory=32768, timeout=4 * 3600)
def run(arm, cmd, results_file):
    import shutil
    import subprocess
    Path("/custmatch_data").mkdir(exist_ok=True)
    if arm == "with_list":
        shutil.copy("/name_list/name_variants.csv", "/custmatch_data/name_variants.csv")
    path = Path("/repo/research/results") / results_file
    before = json.loads(path.read_text()) if path.exists() else {}
    p = subprocess.run(["python", *cmd], cwd="/repo", capture_output=True, text=True)
    if p.returncode:
        return arm, cmd, {"failed": (p.stdout + p.stderr)[-1500:]}
    after = json.loads(path.read_text())
    return arm, cmd, {k: v for k, v in after.items() if before.get(k) != v}


@app.local_entrypoint()
def main():
    jobs = []
    for arm in ("with_list", "without_list"):
        for d in POPS:
            jobs.append((arm, ["research/scripts/cdp_reassess.py", "--seeds", "1,2,3", "--data", d], "cdp_reassess.json"))
        jobs.append((arm, ["research/scripts/au_validate.py", "country_au", "country_au_active"], "au_validation.json"))
    res = {}
    for arm, cmd, out in run.starmap(jobs):
        res.setdefault(arm, {})[" ".join(cmd[1:])] = out
        print(arm, " ".join(cmd), "FAILED" if "failed" in out else "ok", flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
