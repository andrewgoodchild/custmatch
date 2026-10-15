"""manifest.json: what produced a run's outputs, so a run can be shown to be the one it claims.

Since 0.1.1 the same records, labels, config and seed give the same customers. The manifest
records each of those, by SHA-256, with the versions of custmatch, Python and the libraries that
decide the numbers, and the SHA-256 of every output file. To check that an output folder is what a
given set of inputs produces, rerun with those inputs and compare the two manifests
(`custmatch verify DIR --against OTHER`, or compare(a, b)): identical inputs and versions must
give identical outputs. `custmatch verify DIR` alone checks that nothing in the folder, and none
of the input files, has changed since the run. Run times are recorded but never compared
(summary.json is hashed without its stage timings, history.csv without event times).

Nothing secret is written: the hashing key of "hash_fields" stays in its environment variable,
and the config is fingerprinted, not copied. Records are fingerprinted as files - a source file
re-exported with the same rows in another order is a different input, as it is to a run.
(jlink's run provenance - keltokhy/jlink, docs/run-provenance.md - is where the idea came from.)
"""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "custmatch-manifest-1"
NAME = "manifest.json"
# libraries whose versions can change a run's numbers
LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "pyarrow", "phonenumbers", "anyascii",
             "networkx", "catboost")
# clock readings vary from run to run by design: summary.json is hashed without its stage
# timings, history.csv without the time of each event
TIMINGS = ("stage_seconds",)
TIME_COLUMN = {"history.csv": "time"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _output_sha256(path):
    if path.name == "summary.json":
        summary = {k: v for k, v in json.loads(path.read_text()).items() if k not in TIMINGS}
        return hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
    if path.name in TIME_COLUMN:            # stewardship.HISTORY_COLUMNS: the time comes first, ISO, unquoted
        lines = path.read_text().splitlines()
        return hashlib.sha256("\n".join(line.split(",", 1)[-1] for line in lines).encode()).hexdigest()
    return file_sha256(path)


def config_sha256(cfg):
    """The config as custmatch read it, canonical JSON (sorted keys): key order and spacing in the
    file do not change it, a changed setting does. Comments ("#..." keys) are left out."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("#")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for name in ("custmatch",) + LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


def _input(path, base=None, rows=None):
    p = Path(path) if base is None else (Path(base) / path).resolve()
    out = {"path": str(path), "sha256": file_sha256(p), "bytes": p.stat().st_size}
    if rows is not None:
        out["records"] = int(rows)
    return out


def build(command, config_path, cfg, df, seed=0, labels_path=None, started=None, out_dir=None, **extra):
    """The manifest for a run (or add) that wrote out_dir. cfg: the config file's own contents;
    df: the loaded records, for the per-source counts. extra: other inputs by name (a reused
    model's or a previous run's manifest, an erasure list)."""
    config_path = Path(config_path) if config_path is not None else None
    counts = df.source.value_counts().to_dict() if df is not None and len(df) else {}
    m = {"format": FORMAT, "command": command, "seed": seed,
         "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "versions": versions(), "platform": f"{platform.system()} {platform.machine()}",
         "config": None, "sources": [], "labels": None, "other_inputs": {}, "outputs": {}}
    if config_path is not None:
        m["config"] = {"path": str(config_path), "sha256": config_sha256(cfg),
                       "file_sha256": file_sha256(config_path)}
        for src in cfg.get("sources", []):
            if "path" in src:
                m["sources"].append({"name": src.get("name"), **_input(src["path"], config_path.parent,
                                                                       counts.get(src.get("name"), 0))})
    if labels_path is not None:
        m["labels"] = _input(labels_path)
    for name, path in extra.items():
        if path is not None:
            p = Path(path)
            m["other_inputs"][name] = _input(p / NAME) if p.is_dir() and (p / NAME).exists() else \
                (_input(p) if p.is_file() else {"path": str(p)})
    if out_dir is not None:
        out = Path(out_dir)
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.name != NAME:
                m["outputs"][str(f.relative_to(out))] = _output_sha256(f)
    return m


def write(out_dir, manifest):
    (Path(out_dir) / NAME).write_text(json.dumps(manifest, indent=1))
    return manifest


def read(out_dir):
    p = Path(out_dir) / NAME
    if not p.exists():
        from .io import ConfigError
        raise ConfigError(f"no {NAME} in {out_dir}: it was written by an older custmatch, or is not an output folder")
    return json.loads(p.read_text())


def compare(a, b):
    """Differences between two manifests (dicts or output folders) that matter: inputs, versions,
    settings and outputs. [] means the two runs had the same inputs and gave the same outputs."""
    a = a if isinstance(a, dict) else read(a)
    b = b if isinstance(b, dict) else read(b)
    diffs = []
    for key in ("command", "seed"):
        if a.get(key) != b.get(key):
            diffs.append(f"{key}: {a.get(key)} vs {b.get(key)}")
    for key in sorted(set(a["versions"]) | set(b["versions"])):
        if a["versions"].get(key) != b["versions"].get(key):
            diffs.append(f"version of {key}: {a['versions'].get(key)} vs {b['versions'].get(key)}")
    if (a.get("config") or {}).get("sha256") != (b.get("config") or {}).get("sha256"):
        diffs.append("config settings differ")
    sa = {s["name"]: s["sha256"] for s in a.get("sources", [])}
    sb = {s["name"]: s["sha256"] for s in b.get("sources", [])}
    for name in sorted(set(sa) | set(sb)):
        if sa.get(name) != sb.get(name):
            diffs.append(f"source {name}: " + ("only in one run" if name not in sa or name not in sb else "file differs"))
    if (a.get("labels") or {}).get("sha256") != (b.get("labels") or {}).get("sha256"):
        diffs.append("labels differ")
    for name in sorted(set(a.get("other_inputs", {})) | set(b.get("other_inputs", {}))):
        if a["other_inputs"].get(name, {}).get("sha256") != b["other_inputs"].get(name, {}).get("sha256"):
            diffs.append(f"{name} differs")
    oa, ob = a.get("outputs", {}), b.get("outputs", {})
    for name in sorted(set(oa) | set(ob)):
        if oa.get(name) != ob.get(name):
            diffs.append(f"output {name}: " + ("only in one run" if name not in oa or name not in ob else "differs"))
    return diffs


def check(out_dir):
    """Recompute an output folder's hashes against its manifest: its outputs, and the input files
    it names that are still where the run read them. A list of what changed since the run; inputs
    no longer at their paths are listed as not checked."""
    m = read(out_dir)
    out, problems = Path(out_dir), []
    for name, sha in m["outputs"].items():
        p = out / name
        if not p.exists():
            problems.append(f"output {name} is missing")
        elif _output_sha256(p) != sha:
            problems.append(f"output {name} changed since the run")
    base = Path(m["config"]["path"]).parent if m.get("config") else Path(".")
    inputs = [(f"source {s['name']}", (base / s["path"]).resolve(), s["sha256"]) for s in m.get("sources", [])]
    if m.get("labels"):
        inputs.append(("labels", Path(m["labels"]["path"]), m["labels"]["sha256"]))
    for name, p, sha in inputs:
        if not p.exists():
            problems.append(f"{name}: {p} is not there any more (not checked)")
        elif file_sha256(p) != sha:
            problems.append(f"{name}: {p} changed since the run")
    return problems
