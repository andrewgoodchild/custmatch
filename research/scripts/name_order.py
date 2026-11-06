"""Names in either order: how far each dataset's names come in either order, and at what share of
swapped names the name-words feature (core.name_words_feature, found on BPID) starts to pay.

  name_order.py stats                    the either-order share (schema.either_order_share) of every
                                         dataset here: Splink's four, the synthetic customers, the
                                         Australian file, NCVR Leipzig's first two parties, and BPID
  name_order.py swap [DATASET ...]       the synthetic customers and historical_50k with 0%, 10%,
                                         25% and 50% of the records' names swapped: clustered F1
                                         with name_words off and on, 2 label draws (seeds 1 and 2)
                                         of random and active labels each (as beat_splink.py), and
                                         the either-order share of each swapped file
Writes research/results/name_order.json: "stats" and "swap".
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_splink as BS  # noqa: E402
import bpid_benchmark as BP  # noqa: E402

from custmatch import io, schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/name_order.json"
SHARES = (0.0, 0.1, 0.25, 0.5)
SWAP_DATASETS = ["synthetic", "historical_50k"]
SEEDS = (1, 2)
# the current defaults, with name_words pinned off and on (the gate would decide it otherwise)
PINNED = {"rich_features": True, "dob_key": True, "date_parts": False}


def load_dataset(name, tmp):
    """(DataFrame as io.load gives it, config path or None) for every dataset the stats cover."""
    if name in BS.DATASETS:
        config, _ = BS.setup(name, tmp)
        return io.load(config)[0]
    if name == "au":
        cfg = json.loads((ROOT / "examples/au/config.json").read_text()) | {"country": "AU"}
        for s in cfg["sources"]:
            s["path"] = str((ROOT / "examples/au" / s["path"]).resolve())
        (tmp / "au.json").write_text(json.dumps(cfg))
        return io.load(tmp / "au.json")[0]
    if name == "ncvr":
        d = ROOT / "data/ncvr_leipzig/5Party-ocp20"
        frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in sorted(d.glob("*.csv"))[:2]]
        raw = pd.concat(frames, ignore_index=True)
        return pd.DataFrame({"first_name": raw.givenname.str.lower().str.strip(),
                             "last_name": raw.surname.str.lower().str.strip()})
    if name == "bpid":
        csv, _, _ = BP.build(tmp)
        (tmp / "bpid.json").write_text(json.dumps(BP.config(csv, {"country": "US"}, True)))
        return io.load(tmp / "bpid.json")[0]
    raise SystemExit(f"unknown dataset {name}")


def stats(which):
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in which:
            df = load_dataset(name, Path(tmp))
            f, l = df.first_name.to_numpy(), df.last_name.to_numpy()
            out[name] = {"records": len(df), "either_order_share": round(100 * schema.either_order_share(df), 2),
                         "one_name_blank": round(100 * float(((f == "") ^ (l == "")).mean()), 2)}
            print(name, out[name], flush=True)
    return out


def swapped_config(config, share, tmp, tag, seed=0):
    """A copy of the config whose sources have `share` of their records' first and last names
    swapped (the same records for every variant: seed 0, not the label seed)."""
    cfg = json.loads(Path(config).read_text())
    rng = np.random.default_rng(seed)
    for s in cfg["sources"]:
        cols = s["columns"]
        if "first_name" not in cols or "last_name" not in cols:
            continue
        path = Path(s["path"])
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype=str, keep_default_na=False)
        swap = rng.random(len(frame)) < share
        f, l = frame[cols["first_name"]].copy(), frame[cols["last_name"]].copy()
        frame.loc[swap, cols["first_name"]], frame.loc[swap, cols["last_name"]] = l[swap].to_numpy(), f[swap].to_numpy()
        new = tmp / f"{tag}_{s['name']}.csv"
        frame.to_csv(new, index=False)
        s["path"] = str(new)
    out = tmp / f"{tag}.json"
    out.write_text(json.dumps(cfg))
    return out


def swap(which):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in which:
            config, truth = BS.setup(name, tmp)
            config = BS.with_overrides(config, PINNED, tmp, f"{name}_pinned")   # absolute source paths
            for share in SHARES:
                key = f"{int(100 * share)}pct"
                tag = f"{name}_{key}"
                sw = swapped_config(config, share, tmp, tag)
                df, _ = io.load(sw)
                entry = {"either_order_share": round(100 * schema.either_order_share(df), 2), "off": {}, "on": {}}
                for on in (False, True):
                    cfg = BS.with_overrides(sw, PINNED | {"name_words": on}, tmp, f"{tag}_{'on' if on else 'off'}")
                    for seed in SEEDS:
                        r = BS.evaluate(cfg, truth, tmp, f"{tag}_{'on' if on else 'off'}_s{seed}", seed=seed)
                        entry["on" if on else "off"][f"seed{seed}"] = {k: r[k] for k in ("random_1000", "active_300_700")}
                mean = lambda d: round(float(np.mean([v for s in d.values() for v in s.values()])), 2)  # noqa: E731
                entry["mean_off"], entry["mean_on"] = mean(entry["off"]), mean(entry["on"])
                res.setdefault("swap", {}).setdefault(name, {})[key] = entry
                print(f"{name:15s} {key:6s} either-order {entry['either_order_share']:5.2f}%  "
                      f"off {entry['mean_off']:6.2f}  on {entry['mean_on']:6.2f}", flush=True)
                OUT.write_text(json.dumps(res, indent=1))
    return res


def main():
    mode, which = sys.argv[1], sys.argv[2:]
    if mode == "stats":
        res = json.loads(OUT.read_text()) if OUT.exists() else {}
        res["stats"] = stats(which or ["febrl3", "febrl4", "historical_50k", "fake_1000", "synthetic", "au", "ncvr", "bpid"])
        OUT.write_text(json.dumps(res, indent=1))
    elif mode == "swap":
        swap(which or SWAP_DATASETS)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
