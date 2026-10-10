"""Which profanity-list words are real names, and how often the suspect flag fires on real data
(custmatch/suspect.py).

  real names  every single-word entry of the LDNOOBW English list that is a whole first name or
              surname of any of the 5 million North Carolina voters in the Leipzig NCVR files
              (data/ncvr_leipzig/, gitignored). Some are typo corruptions ("cum"), some real
              surnames seen once (Negro, Bastardo, Escort); a flag about a real person must not be
              wrong this way, so all are excluded - suspect.REAL_NAMES, which this checks
  flags       suspect.flag on each NCVR file, and held out: historical_50k and febrl3 (Splink's
              datasets: British and Australian names), fake_1000, and the generated customer files
  time        suspect.flag on the synthetic customer file copied to about a million records
              (custmatch_scale.py), the best of three

    python research/scripts/suspect_research.py
Writes research/results/suspect_research.json.
"""
import glob
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matcher_bakeoff as MB  # noqa: E402

from custmatch import io, suspect  # noqa: E402
from custmatch.country import data_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/suspect_research.json"
NCVR = sorted(glob.glob(str(ROOT / "data/ncvr_leipzig/5Party-ocp20/*.csv")))


def main():
    listed = {w.strip().lower() for w in data_file(suspect.PROFANITY_FILE).read_text().splitlines() if w.strip()}
    single = {w for w in listed if " " not in w}
    names, voters = Counter(), []
    for f in NCVR:
        d = pd.read_csv(f, dtype=str, keep_default_na=False).rename(columns={"givenname": "first_name",
                                                                              "surname": "last_name"})
        d["email"] = ""
        voters.append((Path(f).name, d))
        for c in ("first_name", "last_name"):
            names.update(v for v in d[c] if v in single)
    real = sorted(names)
    res = {"listed_words": len(listed), "single_words": len(single),
           "real_names": {w: names[w] for w in real},
           "real_names_match_the_module": sorted(suspect.REAL_NAMES) == real}
    words = suspect.profanity_words()
    res["ncvr"] = {}
    for name, d in voters:
        fl = suspect.flag(d, words)
        res["ncvr"][name] = {"records": len(d), "flagged": len(fl),
                             "examples": [f"{d.first_name.iat[i]} {d.last_name.iat[i]}" for i in list(fl)[:10]]}
        print(name, len(fl), res["ncvr"][name]["examples"], flush=True)
    res["held_out"] = {}
    for ds in ("historical_50k", "febrl3", "fake_1000", "synthetic", "au"):
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = MB.setup(ds, Path(tmp))
            df, _ = io.load(config)
        fl = suspect.flag(df, words)
        res["held_out"][ds] = {"records": len(df), "flagged": len(fl),
                               "reasons": dict(Counter(r for v in fl.values() for r in v)),
                               "examples": [f"{df.first_name.iat[i]} {df.last_name.iat[i]} <{df.email.iat[i]}>"
                                            for i in list(fl)[:10]]}
        print(ds, len(fl), res["held_out"][ds]["reasons"], res["held_out"][ds]["examples"][:5], flush=True)
    import time

    import custmatch_scale as CS
    with tempfile.TemporaryDirectory() as tmp:
        df, _ = io.load(CS.scaled(50, Path(tmp)))
    times = []
    for _ in range(3):
        t0 = time.time()
        suspect.flag(df, words)
        times.append(time.time() - t0)
    res["million"] = {"records": len(df), "seconds": round(min(times), 1)}
    print("time", res["million"], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
