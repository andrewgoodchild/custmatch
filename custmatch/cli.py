"""custmatch - match customer records across sources.

  custmatch sample CONFIG --n 500 --out to_label.csv [--active labels_so_far.csv]
      random candidate pairs, side by side, for you to label (fill `label` with 1 or 0);
      --active picks the pairs the forest is least sure of instead

  custmatch run CONFIG --labels labels.csv --out results/ [--previous DIR] [--model DIR]
      clusters.csv, golden_records.csv, review.csv, history.csv, summary.json;
      --previous keeps that run's customer IDs; --model reuses its trained model (no labels)

  custmatch blocking CONFIG
      blocking diagnostics before you label: pairs per rule, largest blocks, splits and drops

  custmatch add NEW_CONFIG --state results/ --out results2/ [--delete deleted.csv]
      match new or changed records against a previous run's customers without rerunning it,
      and remove deleted ones (--delete: a CSV of source, record_id); writes changes.csv,
      merges.csv and history.csv

  custmatch forget RECORDS.csv --state results/ --out results2/
      an erasure request: remove these records (source, record_id) and rebuild the affected
      customers without them; delete the old output folders yourself

  custmatch explain --state results/ RECORD_A RECORD_B
      why two records were or were not matched: blocking, evidence, score, overrides

  custmatch monitor results/2026-09 results/2026-10 ... [--out trend.csv] [--fail-on-drift]
      match rates, field fill and score drift across outputs, oldest first; exits 3 on drift
      with --fail-on-drift

CONFIG maps your files and columns onto the pipeline's (see custmatch/io.py and docs/guide.md).
"""
import argparse
import json
import sys

from . import io, labels, pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="custmatch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw candidate pairs for hand-labelling")
    s.add_argument("config")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--active", metavar="LABELS",
                   help="instead of random pairs, the ones the forest trained on LABELS is least sure of")
    s.add_argument("--strategy", choices=["uncertainty", "disagreement"], default="uncertainty",
                   help="with --active: disagreement also offers pairs blocking missed that the forest calls matches")
    s.add_argument("--mix", type=float, default=0.0,
                   help="with --active: share of each batch drawn at random instead (e.g. 0.5)")
    r = sub.add_parser("run", help="match, cluster, households, golden records")
    r.add_argument("config")
    r.add_argument("--labels", help="labelled pairs; optional with --model")
    r.add_argument("--previous", metavar="DIR", help="a previous run's output: keep its customer IDs")
    r.add_argument("--model", metavar="DIR", help="reuse that run's trained model and threshold")
    r.add_argument("--out", required=True)
    bl = sub.add_parser("blocking", help="blocking diagnostics: pairs per rule, largest blocks, splits")
    bl.add_argument("config")
    ad = sub.add_parser("add", help="match new records against a previous run's customers")
    ad.add_argument("config", help="config for the NEW records only")
    ad.add_argument("--state", required=True, help="output folder of a previous run or add")
    ad.add_argument("--out", required=True)
    ad.add_argument("--delete", metavar="CSV", help="records to remove: columns source, record_id")
    fg = sub.add_parser("forget", help="erase records: remove them and rebuild their customers")
    fg.add_argument("records", help="CSV of source, record_id")
    fg.add_argument("--state", required=True)
    fg.add_argument("--out", required=True)
    ex = sub.add_parser("explain", help="why two records were or were not matched")
    ex.add_argument("a", help="first record, source:record_id")
    ex.add_argument("b", help="second record, source:record_id")
    ex.add_argument("--state", required=True, help="output folder of a run or add")
    mo = sub.add_parser("monitor", help="match rates and drift across outputs, oldest first")
    mo.add_argument("outputs", nargs="+", help="output folders of runs or adds, in time order")
    mo.add_argument("--out", help="also write the table to this CSV")
    mo.add_argument("--fail-on-drift", action="store_true", help="exit 3 when any drift alert fires")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "sample":
            k = (labels.sample_active(a.config, a.active, a.n, a.out, a.seed, a.mix, a.strategy) if a.active
                 else labels.sample(a.config, a.n, a.out, a.seed))
            print(f"wrote {k} pairs to {a.out}: fill the label column with 1 (same person) or 0")
        elif a.cmd == "blocking":
            from .blocking import candidates_for
            from .core import profile_contacts
            df, cfg = io.load(a.config)
            # the blocker `run` will use: default, "fixed" or "learned" (which, before labels
            # exist, falls back to fixed rules)
            _, _, rep, _ = candidates_for(df, {f: profile_contacts(df, f) for f in ("email", "phone")}, cfg)
            rep.pop("_skipped", None)
            print(json.dumps(rep, indent=1, default=str))
        elif a.cmd == "add":
            from . import incremental
            print(json.dumps(incremental.add(a.state, a.config, a.out, delete=a.delete), indent=1))
        elif a.cmd == "forget":
            from . import incremental
            print(json.dumps(incremental.add(a.state, None, a.out, delete=a.records), indent=1))
        elif a.cmd == "monitor":
            from . import monitor
            table, alerts = monitor.compare(a.outputs)
            print(table.to_string(index=False))
            print("\n".join(["", "drift alerts:"] + alerts) if alerts else "\nno drift alerts")
            if a.out:
                table.to_csv(a.out, index=False)
            if alerts and a.fail_on_drift:
                return 3
        elif a.cmd == "explain":
            from . import explain
            print(json.dumps(explain.why(a.state, a.a, a.b), indent=1))
        else:
            if a.labels is None and a.model is None:
                print("custmatch: run needs --labels (or --model to reuse a trained model)", file=sys.stderr)
                return 2
            print(json.dumps(pipeline.run(a.config, a.labels, a.out, previous_dir=a.previous, model_dir=a.model), indent=1))
    except (io.ConfigError, FileNotFoundError) as e:
        print(f"custmatch: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
