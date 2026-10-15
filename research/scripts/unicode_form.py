"""What reading text in one Unicode form (io.unicode_form) changes on the benchmark datasets.

  1. how many values change beyond lower-casing, per dataset, with an example
  2. custmatch on historical_50k and the synthetic file - the datasets where values change -
     before (text only lower-cased, as until 0.1.1) and after, with beat_splink.py's "rich"
     settings and labels: 1,000 random labels, and 300 random + 700 active

    python research/scripts/unicode_form.py

Writes research/results/unicode_form.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_splink as BS  # noqa: E402
from custmatch import io  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUN_ON = ["historical_50k", "synthetic"]


def changed_values(config):
    cfg = json.loads(Path(config).read_text())
    n, example = 0, None
    for src in cfg["sources"]:
        p = (Path(config).parent / src["path"]).resolve()
        raw = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype=str, keep_default_na=False)
        for col in src["columns"].values():
            v = raw[col].dropna().astype(str)
            v = v[v.str.contains(r"[^\x00-\x7f]", regex=True)]
            for value, k in v.value_counts().items():
                if io.unicode_form(value) != value.lower():
                    n += int(k)
                    example = example or [value, io.unicode_form(value)]
    return n, example


def main():
    out = {"changed_values": {}, "runs": {}}
    form = io.unicode_form
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in BS.DATASETS:
            config, truth = BS.setup(name, tmp)
            n, example = changed_values(config)
            out["changed_values"][name] = {"values": n, "example": example}
            print(name, out["changed_values"][name], flush=True)
            if name not in RUN_ON:
                continue
            config = BS.with_overrides(config, BS.VARIANTS["rich"], tmp, f"{name}_rich")
            for tag, fn in (("before", str.lower), ("after", form)):
                io.unicode_form = fn
                try:
                    r = BS.evaluate(config, truth, tmp, f"{name}_{tag}")
                finally:
                    io.unicode_form = form
                out["runs"].setdefault(name, {})[tag] = {k: r[k] for k in ("random_1000", "active_300_700")}
                print(name, tag, out["runs"][name][tag], flush=True)
    (ROOT / "research/results/unicode_form.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
