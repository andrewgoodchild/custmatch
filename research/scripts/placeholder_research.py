"""Can fake names and addresses be spotted, and does blanking them help matching?

pseudopeople does not generate placeholder names or addresses, so they are injected: a copy of
the synthetic customer file where 3% of records get a placeholder name ("test test", "john doe",
"unknown", ...) and 3% a placeholder address ("123 main st", "1 test st", ...), identities kept.

  1. false positives: detector run on clean data - the synthetic file and Splink's febrl3,
     historical_50k and fake_1000 (real-looking names, including initials)
  2. detection precision / recall on the injected copy
  3. matching: custmatch on the injected copy with and without "placeholders": true
  4. the same on the clean synthetic file: what blanking costs when there is nothing to find

Writes research/results/placeholder_research.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_splink as BS  # noqa: E402
import custmatch_validate as V  # noqa: E402
from custmatch import io, labels, pipeline, placeholders  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FAKE_NAMES = [("test", "test"), ("john", "doe"), ("jane", "doe"), ("unknown", "unknown"), ("n/a", "n/a"),
              ("asdf", "asdf"), ("customer", "customer"), ("mickey", "mouse"), ("first", "last"), ("xxx", "xxx")]
FAKE_ADDRS = [("123", "main st"), ("1", "test st"), ("0", "unknown"), ("123", "fake st"), ("", "none"),
              ("9999", "no fixed address"), ("12345", "any street")]


def injected(tmp, rate=0.03, seed=0):
    rng = np.random.default_rng(seed)
    cfg = json.loads(V.CONFIG.read_text())
    truth = {}
    for s in cfg["sources"]:
        d = pd.read_parquet((V.CONFIG.parent / s["path"]).resolve()).astype(str).replace({"nan": "", "None": ""})
        n = len(d)
        name_rows = rng.random(n) < rate
        addr_rows = rng.random(n) < rate
        picks = rng.integers(0, len(FAKE_NAMES), n)
        d.loc[name_rows, "first_name"] = [FAKE_NAMES[k][0] for k in picks[name_rows]]
        d.loc[name_rows, "last_name"] = [FAKE_NAMES[k][1] for k in picks[name_rows]]
        d.loc[name_rows, "middle_initial"] = ""
        apicks = rng.integers(0, len(FAKE_ADDRS), n)
        d.loc[addr_rows, "addr_number"] = [FAKE_ADDRS[k][0] for k in apicks[addr_rows]]
        d.loc[addr_rows, "addr_street"] = [FAKE_ADDRS[k][1] for k in apicks[addr_rows]]
        d.loc[addr_rows, "addr_unit"] = ""
        for rid, fn_, fa in zip(d.record_id, name_rows, addr_rows):
            truth[f"{s['name']}:{rid}"] = (bool(fn_), bool(fa))
        path = tmp / f"inj_{s['name']}.parquet"
        d.to_parquet(path, index=False)
        s["path"] = str(path)
    (tmp / "inj.json").write_text(json.dumps(cfg))
    return tmp / "inj.json", truth


def pr(pred, true):
    tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
    return {"precision": round(100 * tp / (tp + fp), 2) if tp + fp else 0.0,
            "recall": round(100 * tp / (tp + fn), 2) if tp + fn else 0.0, "flagged": int(pred.sum())}


def main():
    out = {"false_positives_clean": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("synthetic", "febrl3", "historical_50k", "fake_1000"):
            config, _ = BS.setup(name, tmp)
            df, _ = io.load(config)
            fn_, fa = placeholders.detect(df)
            out["false_positives_clean"][name] = {"records": len(df), "names_flagged": int(fn_.sum()),
                                                  "addresses_flagged": int(fa.sum()),
                                                  "examples": sorted(set(
                                                      (df.first_name + " " + df.last_name)[fn_]))[:5] +
                                                  sorted(set((df.addr_number + " " + df.addr_street)[fa]))[:5]}
            print(name, out["false_positives_clean"][name], flush=True)

        config, fakes = injected(tmp)
        df, _ = io.load(config)
        keys = io.key(df)
        true_name = np.array([fakes[k][0] for k in keys])
        true_addr = np.array([fakes[k][1] for k in keys])
        fn_, fa = placeholders.detect(df)
        out["detection_injected"] = {"names": pr(fn_, true_name), "addresses": pr(fa, true_addr)}
        print("detection:", out["detection_injected"], flush=True)

        gold = V.truth()
        clean = json.loads(V.CONFIG.read_text())
        for s in clean["sources"]:
            s["path"] = str((V.CONFIG.parent / s["path"]).resolve())
        (tmp / "clean.json").write_text(json.dumps(clean))
        for prefix, base in (("", config), ("clean_", tmp / "clean.json")):
            lp = tmp / f"{prefix}l.csv"
            labels.sample(base, 1000, lp, seed=1)
            lab = V.fill(lp, gold)
            run_both(base, lp, lab, gold, tmp, prefix, out)
    (ROOT / "research/results/placeholder_research.json").write_text(json.dumps(out, indent=1, default=float))


def run_both(config, lp, lab, gold, tmp, prefix, out):
    for tag, on in ((f"{prefix}placeholders_off", False), (f"{prefix}placeholders_on", True)):
        c = json.loads(config.read_text()) | {"placeholders": on}
        (tmp / f"{tag}.json").write_text(json.dumps(c))
        pipeline.run(tmp / f"{tag}.json", lp, tmp / tag, log=lambda *_: None)
        out[tag] = V.score(tmp / tag, gold, lab)
        r = out[tag]
        print(f"{tag}: clustered F1 {r['all_records']['f1']:.2f}, households {r['households']['f1']:.2f}", flush=True)


if __name__ == "__main__":
    main()
