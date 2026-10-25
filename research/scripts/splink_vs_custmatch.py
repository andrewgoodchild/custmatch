"""Head-to-head: Splink vs custmatch on the synthetic customer file, scored identically.

Splink (MIT, Fellegi-Sunter, unsupervised) is the most used open-source alternative. Both run
on the same canonical records (`custmatch.io.load`), with Splink's blocking mirroring
custmatch's keys and Splink's standard comparisons - term-frequency adjustment on email and
phone is its own answer to junk values like test@test.com. Both are clustered by connected
components and scored as clustered F1 against the true identities on all records.

  splink_no_labels_at_<p>      Splink with no labels, at conventional match-probability cuts
  splink_threshold_from_labels Splink, threshold chosen after clustering on the same 1,000
                               labelled pairs custmatch used (Splink never trains on them)
  custmatch_1000_labels        from research/results/custmatch_validation.json (same labels)
  custmatch_seeded_by_splink   gap 4, no hand labels: Splink's most confident matches and
                               non-matches become custmatch's training labels

Writes research/results/splink_vs_custmatch.json.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custmatch_validate as V  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import components, partition_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def splink_scores(df, null_junk=False, labelled=None):
    """Splink as its docs recommend: separate name comparisons with term-frequency adjustment,
    EM rounds blocked so every comparison is trained. null_junk blanks contact values that
    custmatch's unsupervised profiler flags as junk, isolating what that profiling adds.
    labelled = (i, j, y): estimate m from those labelled pairs (Splink's supervised option)
    instead of EM."""
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    from custmatch.core import profile_contacts
    d = df.copy()
    if null_junk:
        for f in ("email", "phone"):
            prof = profile_contacts(df, f)
            d.loc[d[f].map(lambda v: prof.get(v, ("personal",))[0] == "junk"), f] = ""
    d["unique_id"] = np.arange(len(d))
    for c in io.CANONICAL[1:]:
        d[c] = d[c].replace("", None)
    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("last_name", "substr(first_name, 1, 3)"), block_on("last_name", "addr_street"),
            block_on("addr_number", "addr_street"), block_on("first_name", "addr_street"),
            block_on("email"), block_on("phone")],
        comparisons=[
            cl.NameComparison("first_name").configure(term_frequency_adjustments=True),
            cl.NameComparison("last_name").configure(term_frequency_adjustments=True),
            cl.ExactMatch("middle_initial"),
            cl.DateOfBirthComparison("date_of_birth", input_is_string=True,
                                     datetime_format="%m/%d/%Y", invalid_dates_as_null=True),
            cl.ExactMatch("addr_number"),
            cl.JaroWinklerAtThresholds("addr_street", [0.9]),
            cl.EmailComparison("email").configure(term_frequency_adjustments=True),
            cl.ExactMatch("phone").configure(term_frequency_adjustments=True),
            cl.ExactMatch("ssn"),
        ],
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker(d, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("first_name", "last_name", "addr_number")], recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
    if labelled is None:
        # each round leaves its blocking columns untrained, so the rounds cover each other
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("addr_number", "addr_street"))
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("first_name", "last_name"))
    else:
        i, j, y = labelled
        lab = pd.DataFrame({"unique_id_l": i, "source_dataset_l": "d", "unique_id_r": j,
                            "source_dataset_r": "d", "clerical_match_score": np.asarray(y, float)})
        linker.table_management.register_table(lab, "labels", overwrite=True)
        linker.training.estimate_m_from_pairwise_labels("labels")
    p = linker.inference.predict(threshold_match_probability=0.001).as_pandas_dataframe()
    pairs = np.c_[np.minimum(p.unique_id_l, p.unique_id_r), np.maximum(p.unique_id_l, p.unique_id_r)]
    # match_weight (log2 odds) rather than probability: probabilities saturate at 1.0, so the
    # useful thresholds lie far beyond 0.999
    return pairs.astype(np.int32), p.match_weight.to_numpy()


def score_labels(df, cc, gold):
    sid = np.array([gold[k][0] for k in io.key(df)])
    _, c = np.unique(sid, return_counts=True)
    return partition_scores(cc, sid, float(np.sum(c * (c - 1) / 2)))


WEIGHTS = np.arange(-5, 61, 1.0)          # match-weight grid (log2 Bayes factor + prior)
CONVENTIONAL = {"p>0.9": np.log2(0.9 / 0.1), "p>0.99": np.log2(0.99 / 0.01)}


def main():
    gold = V.truth()
    df, _ = io.load(V.CONFIG)
    n = len(df)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        lp = Path(tmp) / "labels_1000.csv"      # the same 1,000 pairs custmatch_validate used
        labels.sample(V.CONFIG, 1000, lp, seed=1, stratified=False)
        V.fill(lp, gold)
        li, lj, ly = pipeline.read_labels(lp, df)
    lab = np.c_[np.minimum(li, lj), np.maximum(li, lj)]
    half = np.random.default_rng(0).permutation(len(ly)) % 2 == 0

    for name, kw, tune in (
            ("splink_unsupervised", {}, np.ones(len(ly), bool)),
            # m estimated from half the labels; the threshold from the other half
            ("splink_m_from_500_labels", {"labelled": (lab[half, 0], lab[half, 1], ly[half])}, ~half)):
        t0 = time.time()
        pairs, w = splink_scores(df, **kw)
        r = {"seconds": round(time.time() - t0, 1), "pairs": int(len(pairs))}
        if name == "splink_unsupervised":
            for label, t in CONVENTIONAL.items():
                r[f"no_labels_{label}"] = score_labels(df, components(n, pairs, w > t), gold)
            best = max(((t, score_labels(df, components(n, pairs, w > t), gold)) for t in WEIGHTS),
                       key=lambda z: z[1]["f1"])
            r["ceiling_threshold_picked_on_truth"] = best[1] | {"match_weight": float(best[0])}
            unsup = (pairs, w)
        th, _ = pipeline.choose_threshold(n, pairs, lab, ly, [(w, tune)], WEIGHTS)
        r["threshold_from_labels"] = score_labels(df, components(n, pairs, w > th), gold) | {"match_weight": th}
        out[name] = r
        print(name, json.dumps({k: (v["f1"] if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)

    # gap 4, no hand labels: Splink's most confident calls become custmatch's training labels
    pairs, w = unsup
    keys, cand = io.key(df), labels.candidates(df)
    wmap = dict(zip(map(tuple, pairs.tolist()), w))
    cw = np.array([wmap.get((i, j), -99.0) for i, j in cand.tolist()])
    top, bottom = np.argsort(-cw)[:1000], np.argsort(cw)[:1000]
    seeded = pd.DataFrame([{"a": keys[i], "b": keys[j], "label": 1} for i, j in cand[top]] +
                          [{"a": keys[i], "b": keys[j], "label": 0} for i, j in cand[bottom]])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        seeded.to_csv(tmp / "seeded.csv", index=False)
        acc = float(np.mean([int(gold[a][0] == gold[b][0]) == y for a, b, y in
                             zip(seeded.a, seeded.b, seeded.label)]))
        pipeline.run(V.CONFIG, tmp / "seeded.csv", tmp / "out", log=lambda *_: None)
        s = V.score(tmp / "out", gold, seeded)
    out["custmatch_seeded_by_splink"] = s | {"seed_label_accuracy": round(100 * acc, 2)}
    print(f"custmatch seeded by Splink's 2,000 most confident calls ({100 * acc:.2f}% correct): "
          f"clustered F1 {s['all_records']['f1']:.2f}", flush=True)

    v = json.load(open(ROOT / "research/results/custmatch_validation.json"))
    out["custmatch_1000_labels"] = v["hand_1000"]["all_records"]
    print(f"custmatch, 1,000 hand labels: clustered F1 {out['custmatch_1000_labels']['f1']:.2f}")
    (ROOT / "research/results/splink_vs_custmatch.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
