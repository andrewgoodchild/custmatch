"""Which pairwise classifier, under 0.1.0's defaults - run on Modal (modal.com), in parallel.

Should custmatch keep its optional models (CatBoost; TabPFN, which needs torch), and is there
something better? Every candidate trains on exactly the same labels: for each dataset and label
draw one CPU container builds 1,000 random labels and 300 random + 700 active ones (chosen by
the default model, as a user would) and stores them on a Modal volume; then one CPU container
per (dataset, draw) runs every CPU model on them, and GPU containers run the GPU models.

  shipped     forest (default), hgb, catboost, tabpfn
  candidates  extratrees (scikit-learn), lightgbm, xgboost - lighter installs than CatBoost -
              and tabicl, an open tabular foundation model like TabPFN (GPU)

The candidates are defined here, not in custmatch, until one earns a place. The review that
removed hgb, CatBoost and TabPFN (results/model_review.json) ran while the match weight was on by
default, with the middle-of-ties threshold rule; TabPFN with its default 4 ensemble members (the
earlier bake-off used 1 to fit a laptop). Now the weight is opt-in and runs as its own model name.

    modal run scripts/modal_models.py                      # everything
    modal run scripts/modal_models.py --datasets febrl3 --seeds 1 --models forest,hgb   # smoke test
    modal run scripts/modal_models.py --models forest,forest_match_weight --out results/match_weight.json

Data: the generated customer files (data/customer, data/au_customer) are uploaded with the
code; the public benchmarks download inside the container. Writes results/model_review.json.
Cost at Modal's prices (2026-10): about $0.047 per core-hour, $0.80 per L4 GPU-hour.
"""
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/model_review.json"
DATASETS = ["febrl3", "febrl4", "fake_1000", "synthetic", "au", "historical_50k"]
CPU_MODELS = ["forest", "forest_match_weight", "hgb", "extratrees", "lightgbm", "xgboost", "catboost",
              "forest_catboost"]
# "forest_catboost": shipped in custmatch (opt-in), the forest's and CatBoost's P(match) averaged
# "forest_match_weight": the forest with "match_weight_feature": true (custmatch/matchweight.py)
GPU_MODELS = ["tabpfn", "tabicl"]

app = modal.App("custmatch-model-review")
vol = modal.Volume.from_name("custmatch-model-review", create_if_missing=True)
_base = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "httpx", "certifi", "pyarrow",
                      "phonenumbers", "anyascii", "networkx", "duckdb", "jellyfish", "splink==4.0.17",
                      "catboost", "lightgbm", "xgboost")
         .env({"PYTHONPATH": "/repo:/repo/scripts"}))
_gpu = _base.pip_install("torch", "tabpfn", "tabicl")


def _code(image):
    for d in ("custmatch", "scripts", "examples"):
        image = image.add_local_dir(ROOT / d, f"/repo/{d}", ignore=["**/__pycache__"])
    for d in ("data/customer", "data/au_customer"):
        image = image.add_local_dir(ROOT / d, f"/repo/{d}")
    return image


cpu_image, gpu_image = _code(_base), _code(_gpu)


# ------------------------------------------------------------------ candidate models
class _Balanced:
    """XGBoost has no class_weight: weight the matches up to the non-matches at fit time."""
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        import numpy as np
        pos = max(int(np.sum(y)), 1)
        self.model.set_params(scale_pos_weight=(len(y) - pos) / pos)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class Chunked:
    """Predicts in slices: TabPFN and TabICL attend over all test rows at once and ran out of GPU
    memory on 185,675 candidate pairs."""
    CHUNK = 20000

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        import numpy as np
        if len(X) <= self.CHUNK:
            return self.model.predict_proba(X)
        return np.vstack([self.model.predict_proba(X[s:s + self.CHUNK]) for s in range(0, len(X), self.CHUNK)])


def _install_candidates():
    """Teach custmatch.models.factory the candidate names (the pipeline looks it up per run).
    hgb, CatBoost and TabPFN shipped in custmatch until this review removed them; they are defined
    here, as they shipped, so the comparison can be rerun."""
    from custmatch import models
    shipped = models.factory

    def factory(cfg, seed=0):
        name = cfg.get("matcher_model", "forest")
        if name == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                          class_weight="balanced", random_state=seed)
        if name == "extratrees":
            from sklearn.ensemble import ExtraTreesClassifier
            return lambda: ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                                class_weight="balanced_subsample")
        if name == "lightgbm":
            from lightgbm import LGBMClassifier
            return lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                                          random_state=seed, verbose=-1)
        if name == "xgboost":
            from xgboost import XGBClassifier
            return lambda: _Balanced(XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                   random_state=seed, n_jobs=-1, tree_method="hist"))
        if name == "catboost":
            from catboost import CatBoostClassifier
            return lambda: CatBoostClassifier(iterations=500, auto_class_weights="Balanced",
                                              random_seed=seed, verbose=False, thread_count=-1,
                                              allow_writing_files=False)
        if name == "tabpfn":
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            n = int(cfg.get("_tabpfn_estimators", 4))
            return lambda: Chunked(TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, device="cuda", n_estimators=n, random_state=seed,
                ignore_pretraining_limits=True))
        if name == "tabicl":
            from tabicl import TabICLClassifier
            return lambda: Chunked(TabICLClassifier(device="cuda", random_state=seed))
        return shipped(cfg, seed)
    models.factory = factory
    models.MODELS = models.MODELS + ("hgb", "extratrees", "lightgbm", "xgboost", "catboost", "tabpfn", "tabicl")
    models.REMOVED = ()


# ------------------------------------------------------------------ jobs
def _setup(ds, tmp):
    import sys
    sys.path.insert(0, "/repo/scripts")
    import matcher_bakeoff as MB
    return MB.setup(ds, tmp)


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=3 * 3600, volumes={"/vol": vol})
def build_labels(ds, seed):
    """The label files every model of this (dataset, draw) trains on, saved to the volume."""
    import shutil
    import sys
    import tempfile
    sys.path.insert(0, "/repo/scripts")
    import matcher_bakeoff as MB
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config, truth = _setup(ds, tmp)
        files = MB.label_files(config, truth, tmp, f"{ds}_{seed}", seed)
        Path("/vol/labels").mkdir(parents=True, exist_ok=True)
        for lt, p in files.items():
            shutil.copy(p, f"/vol/labels/{ds}_{seed}_{lt}.csv")
    vol.commit()
    return ds, seed


def _run_models(ds, seed, model_names, extra=None):
    import sys
    import tempfile
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/repo/scripts")
    import splink_datasets_benchmark as B
    from custmatch import pipeline
    _install_candidates()
    vol.reload()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config, truth = _setup(ds, tmp)
        for m in model_names:
            cfg = json.loads(Path(config).read_text())
            for s in cfg["sources"]:
                s["path"] = str((Path(config).parent / s["path"]).resolve())
            cp = tmp / f"{m}.json"
            model = {"matcher_model": "forest", "match_weight_feature": True} if m == "forest_match_weight" \
                else {"matcher_model": m}
            cp.write_text(json.dumps(cfg | model | (extra or {})))
            for lt in ("random_1000", "active_300_700"):
                t0 = time.time()
                try:
                    o = tmp / f"{m}_{lt}"
                    pipeline.run(cp, f"/vol/labels/{ds}_{seed}_{lt}.csv", o, seed=seed, log=lambda *_: None)
                    c = pd.read_csv(o / "clusters.csv", dtype=str)
                    t = np.array([truth[f"{a}:{b}"] for a, b in zip(c.source, c.record_id)])
                    r = {"f1": B.clustered_f1(c.customer_id.to_numpy(), t)["f1"]}
                except Exception as e:        # one model failing must not lose the others
                    r = {"error": f"{type(e).__name__}: {e}"[:300]}
                out[(m, lt)] = r | {"seconds": round(time.time() - t0, 1)}
                print(ds, seed, m, lt, out[(m, lt)], flush=True)
    return ds, seed, {f"{m}|{lt}": v for (m, lt), v in out.items()}


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=4 * 3600, volumes={"/vol": vol})
def run_cpu(ds, seed, model_names):
    return _run_models(ds, seed, model_names)


@app.function(image=gpu_image, gpu="L4", cpu=4, memory=32768, timeout=6 * 3600, volumes={"/vol": vol})
def run_gpu(ds, seed, model_names):
    return _run_models(ds, seed, model_names, {"_tabpfn_estimators": 4})


@app.local_entrypoint()
def main(datasets: str = ",".join(DATASETS), seeds: str = "1,2,3", models: str = "", out: str = ""):
    """out: a results file other than results/model_review.json (e.g. results/match_weight.json)."""
    global OUT
    if out:
        OUT = ROOT / out
    wanted = models.split(",") if models else CPU_MODELS + GPU_MODELS
    cpu = [m for m in wanted if m in CPU_MODELS]
    gpu = [m for m in wanted if m in GPU_MODELS]
    jobs = [(ds, int(s)) for ds in datasets.split(",") for s in seeds.split(",")]
    list(build_labels.starmap(jobs))
    print(f"labels built for {len(jobs)} (dataset, draw) pairs", flush=True)
    calls = ([run_cpu.spawn(ds, s, cpu) for ds, s in jobs] if cpu else []) + \
            ([run_gpu.spawn(ds, s, [m]) for ds, s in jobs for m in gpu])
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for call in calls:
        ds, seed, r = call.get()
        for key, v in r.items():
            m, lt = key.split("|")
            res.setdefault(ds, {}).setdefault(m, {})[f"{lt}/seed{seed}"] = v
        OUT.write_text(json.dumps(res, indent=1))
    summary(res)


def summary(res):
    import numpy as np
    for ds, by in res.items():
        base = by.get("forest", {})
        for m, runs in by.items():
            for lt in ("random_1000", "active_300_700"):
                f = [v["f1"] for k, v in runs.items() if k.startswith(lt) and "f1" in v]
                b = [v["f1"] for k, v in base.items() if k.startswith(lt) and "f1" in v]
                err = sum("error" in v for k, v in runs.items() if k.startswith(lt))
                if f:
                    print(f"{ds:15s} {m:11s} {lt:15s} {np.mean(f):6.2f} (n={len(f)})"
                          + (f"  vs forest {np.mean(f) - np.mean(b):+.2f}" if b and m != "forest" else "")
                          + (f"  {err} failed" if err else ""))
