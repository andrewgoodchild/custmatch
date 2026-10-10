"""Feed the custmatch CLI the inputs real users produce and check it never crashes.

Each case copies the demo (examples/demo), breaks one thing - a missing file, a misspelt column,
an empty source, duplicate IDs, labels that point nowhere, another encoding, one giant block -
and runs `custmatch run` (and `sample` where it matters). A case passes when the command either
succeeds or stops with a clear message (exit code 2, no Python traceback); cases in MUST_STOP must
stop, and cases in MUST_WARN must warn - a silent run there hides a problem from the user.

    python scripts/robustness.py [CASE ...]
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/demo"


def _cfg(d):
    return json.loads((d / "config.json").read_text())


def _write_cfg(d, cfg):
    (d / "config.json").write_text(json.dumps(cfg))


def _csv(d, name):
    return pd.read_csv(d / name, dtype=str, keep_default_na=False)


# each case: (description, function(dir) that breaks the copy, command: "run" | "sample")
def missing_config(d):
    (d / "config.json").unlink()


def invalid_json(d):
    (d / "config.json").write_text("{ not json")


def missing_source_file(d):
    (d / "app.csv").unlink()


def misspelt_column(d):
    cfg = _cfg(d)
    cfg["sources"][0]["columns"]["first_name"] = "firstname"
    _write_cfg(d, cfg)


def unknown_canonical_column(d):
    cfg = _cfg(d)
    cfg["sources"][0]["columns"]["favourite_colour"] = "first_name"
    _write_cfg(d, cfg)


def empty_source(d):
    w = _csv(d, "web.csv")
    w.head(0).to_csv(d / "web.csv", index=False)


def header_only_everywhere(d):
    for n in ("web.csv", "app.csv", "store.csv"):
        _csv(d, n).head(0).to_csv(d / n, index=False)


def all_names_blank(d):
    for n in ("web.csv", "app.csv", "store.csv"):
        x = _csv(d, n)
        x["first_name"] = ""
        x["last_name"] = ""
        x.to_csv(d / n, index=False)


def duplicate_record_ids(d):
    w = _csv(d, "web.csv")
    pd.concat([w, w.head(20)]).to_csv(d / "web.csv", index=False)


def blank_record_ids(d):
    w = _csv(d, "web.csv")
    w.loc[:10, "record_id"] = ""
    w.to_csv(d / "web.csv", index=False)


def one_record(d):
    for n in ("web.csv", "app.csv", "store.csv"):
        _csv(d, n).head(1 if n == "web.csv" else 0).to_csv(d / n, index=False)


def latin1_encoding(d):
    w = _csv(d, "web.csv")
    w.loc[0, "first_name"] = "josé"
    w.loc[1, "last_name"] = "müller"
    w.to_csv(d / "web.csv", index=False, encoding="latin-1")


def utf8_bom(d):
    w = _csv(d, "web.csv")
    w.to_csv(d / "web.csv", index=False, encoding="utf-8-sig")


def numeric_looking_values(d):
    w = _csv(d, "web.csv")
    w["record_id"] = [str(i) for i in range(len(w))]
    w["mobile"] = w.mobile.str.replace(r"\D", "", regex=True)          # 61412345678, read as numbers
    w["postcode"] = w.postcode.str.lstrip("0")
    w.to_csv(d / "web.csv", index=False)
    lab = _csv(d, "labels.csv")
    lab = lab[~lab.a.str.startswith("web:") & ~lab.b.str.startswith("web:")]
    lab.to_csv(d / "labels.csv", index=False)


def odd_dates(d):
    s = _csv(d, "store.csv")
    s.loc[:50, "date_of_birth"] = ["31/02/1990", "1990-13-45", "yesterday", "0000-00-00", "19900101", "1/1/90"] * 8 + ["", "", ""]
    s.to_csv(d / "store.csv", index=False)


def labels_missing_column(d):
    _csv(d, "labels.csv").drop(columns=["label"]).to_csv(d / "labels.csv", index=False)


def labels_unknown_records(d):
    lab = _csv(d, "labels.csv")
    lab.loc[:5, "a"] = "web:no_such_record"
    lab.to_csv(d / "labels.csv", index=False)


def labels_bad_values(d):
    lab = _csv(d, "labels.csv")
    lab.loc[:5, "label"] = "maybe"
    lab.to_csv(d / "labels.csv", index=False)


def labels_all_zero(d):
    lab = _csv(d, "labels.csv")
    lab["label"] = "0"
    lab.to_csv(d / "labels.csv", index=False)


def labels_empty(d):
    _csv(d, "labels.csv").head(0).to_csv(d / "labels.csv", index=False)


def labels_few(d):
    _csv(d, "labels.csv").head(12).to_csv(d / "labels.csv", index=False)


def unknown_country(d):
    cfg = _cfg(d)
    cfg["country"] = "XX"
    _write_cfg(d, cfg)


def extra_field_missing(d):
    cfg = _cfg(d)
    cfg["extra_fields"] = cfg.get("extra_fields", []) + [{"name": "loyalty_tier", "compare": "exact"}]
    _write_cfg(d, cfg)


def bad_compare_type(d):
    cfg = _cfg(d)
    cfg["extra_fields"] = [{"name": "postcode", "compare": "fuzzy"}]
    _write_cfg(d, cfg)


def removed_options(d):
    cfg = _cfg(d)
    cfg |= {"splink_feature": True, "clustering": "correlation", "missing_as_unknown": True}
    _write_cfg(d, cfg)


def removed_model(d):
    cfg = _cfg(d)
    cfg["matcher_model"] = "tabpfn"
    _write_cfg(d, cfg)


def one_giant_block(d):
    for n in ("web.csv", "app.csv", "store.csv"):
        x = _csv(d, n)
        x["first_name"] = "john"
        x["last_name"] = "smith"
        x.to_csv(d / n, index=False)


def junk_everywhere(d):
    for n in ("web.csv", "app.csv", "store.csv"):
        x = _csv(d, n)
        x["email"] = "test@test.com"
        x["mobile"] = "0400 000 000"
        x.to_csv(d / n, index=False)


CASES = {f.__name__ + ("" if cmd == "run" else "/sample"): (f, cmd) for f, cmd in [
    (missing_config, "run"), (invalid_json, "run"), (missing_source_file, "run"), (misspelt_column, "run"),
    (unknown_canonical_column, "run"), (empty_source, "run"), (header_only_everywhere, "run"),
    (all_names_blank, "run"), (duplicate_record_ids, "run"), (blank_record_ids, "run"), (one_record, "run"),
    (latin1_encoding, "run"), (utf8_bom, "run"), (numeric_looking_values, "run"), (odd_dates, "run"),
    (labels_missing_column, "run"), (labels_unknown_records, "run"), (labels_bad_values, "run"),
    (labels_all_zero, "run"), (labels_empty, "run"), (labels_few, "run"), (unknown_country, "run"),
    (extra_field_missing, "run"), (bad_compare_type, "run"), (removed_options, "run"), (removed_model, "run"),
    (one_giant_block, "run"), (junk_everywhere, "run"),
    (empty_source, "sample"), (header_only_everywhere, "sample"), (one_record, "sample"), (one_giant_block, "sample"),
]}


MUST_STOP = {"invalid_json", "header_only_everywhere", "header_only_everywhere/sample", "latin1_encoding",
             "unknown_country", "extra_field_missing", "bad_compare_type", "blank_record_ids", "removed_model"}
MUST_WARN = {"one_giant_block", "removed_options", "unknown_canonical_column"}


def run_case(name, fn, cmd):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "demo"
        shutil.copytree(DEMO, d)
        fn(d)
        args = (["run", "config.json", "--labels", "labels.csv", "--out", "out"] if cmd == "run"
                else ["sample", "config.json", "--n", "50", "--out", "s.csv"])
        p = subprocess.run([sys.executable, "-m", "custmatch", *args], cwd=d, capture_output=True, text=True,
                           timeout=600)
        err = p.stderr.strip()
        crashed = "Traceback" in err
        warnings = []
        if p.returncode == 0 and cmd == "run":
            warnings = json.loads((d / "out" / "summary.json").read_text()).get("warnings", [])
        msg = (err.splitlines()[-1] if err else "") if p.returncode else "; ".join(w[:70] for w in warnings)
        silent = (name in MUST_STOP and p.returncode == 0) or \
                 (name in MUST_WARN and p.returncode == 0 and not any("random" not in w for w in warnings))
        return {"case": name, "exit": p.returncode, "crashed": crashed or silent, "message": msg[:200],
                "warnings": len(warnings)}


def main():
    names = sys.argv[1:] or list(CASES)
    bad = []
    for name in names:
        fn, cmd = CASES[name]
        r = run_case(name, fn, cmd)
        status = "CRASH" if r["crashed"] else ("ok   " if r["exit"] == 0 else "error")
        print(f"{status}  {r['case']:40s} exit {r['exit']}  warnings {r['warnings']}  {r['message']}", flush=True)
        if r["crashed"]:
            bad.append(r["case"])
    print(f"\n{len(bad)} crashed: {bad}")


if __name__ == "__main__":
    main()
