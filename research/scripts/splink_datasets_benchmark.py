"""custmatch vs Splink on Splink's own bundled datasets, with Splink's own settings.

Splink runs the settings and training of its official v4.0.17 example notebooks, verbatim
(sources in each function). custmatch gets the same fields through its config extras, and its
extra blocking copies Splink's blocking rules for that dataset (the ones expressible as column
equality), so neither side wins on candidate generation alone. Both are clustered by connected
components and scored as clustered F1 against the dataset's ground truth, on all records.

  splink_documented      Splink at the clustering threshold its example uses, where it has one
  splink_from_labels     Splink, match-weight threshold chosen on custmatch's 1,000 labels
  splink_ceiling         Splink at its best match-weight threshold, picked on the truth
  custmatch_1000         custmatch, 1,000 random labelled pairs (labels filled from the truth)
  custmatch_active       custmatch, 300 random + 700 active-learning labelled pairs

Writes research/results/splink_datasets_benchmark.json.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import io, labels, pipeline
from custmatch.core import components, partition_scores

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = np.arange(-10, 61, 0.5)


# ------------------------------------------------------------------ Splink, official settings
def splink_fake_1000(df):
    # docs/demos/tutorials/04_Estimating_model_parameters.ipynb @ v4.0.17
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[cl.NameComparison("first_name"), cl.NameComparison("surname"),
                     cl.LevenshteinAtThresholds("dob", 1),
                     cl.ExactMatch("city").configure(term_frequency_adjustments=True),
                     cl.EmailComparison("email")],
        blocking_rules_to_generate_predictions=[block_on("first_name", "city"), block_on("surname")],
        retain_intermediate_calculation_columns=True)
    linker = Linker(df, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("first_name", "dob"),
         "l.first_name = r.first_name and levenshtein(r.surname, l.surname) <= 2",
         block_on("email")], recall=0.7)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("first_name", "surname"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
    return linker, 0.5                       # tutorial 05 clusters at 0.5


def splink_febrl3(df):
    # docs/demos/examples/duckdb/febrl3.ipynb @ v4.0.17 (the example does not cluster)
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    settings = SettingsCreator(
        unique_id_column_name="rec_id", link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("soc_sec_id"), block_on("given_name"), block_on("surname"),
            block_on("date_of_birth"), block_on("postcode")],
        comparisons=[cl.NameComparison("given_name"), cl.NameComparison("surname"),
                     cl.DateOfBirthComparison("date_of_birth", input_is_string=True, datetime_format="%Y%m%d"),
                     cl.DamerauLevenshteinAtThresholds("soc_sec_id", [2]),
                     cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
                     cl.ExactMatch("postcode").configure(term_frequency_adjustments=True)],
        retain_intermediate_calculation_columns=True)
    linker = Linker(df, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("soc_sec_id"), block_on("given_name", "surname", "date_of_birth"),
         "l.given_name = r.surname and l.surname = r.given_name and l.date_of_birth = r.date_of_birth"],
        recall=0.9)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("date_of_birth"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("postcode"))
    return linker, None


def splink_febrl4(dfs):
    # docs/demos/examples/duckdb/febrl4.ipynb @ v4.0.17, the "detailed" model
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    settings = SettingsCreator(
        unique_id_column_name="rec_id", link_type="link_only",
        blocking_rules_to_generate_predictions=[
            block_on("given_name", "surname"), "l.given_name = r.surname and l.surname = r.given_name",
            block_on("date_of_birth"), block_on("soc_sec_id"), block_on("state", "address_1"),
            block_on("street_number", "address_1"), block_on("postcode")],
        comparisons=[cl.NameComparison("given_name").configure(term_frequency_adjustments=True),
                     cl.NameComparison("surname").configure(term_frequency_adjustments=True),
                     cl.DateOfBirthComparison("date_of_birth", input_is_string=True,
                                              datetime_format="%Y%m%d", invalid_dates_as_null=True),
                     cl.DamerauLevenshteinAtThresholds("soc_sec_id", [1, 2]),
                     cl.ExactMatch("street_number").configure(term_frequency_adjustments=True),
                     cl.DamerauLevenshteinAtThresholds("postcode", [1, 2]).configure(
                         term_frequency_adjustments=True)],
        retain_intermediate_calculation_columns=True)
    linker = Linker(dfs, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("soc_sec_id"), block_on("given_name", "surname", "date_of_birth")], recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("date_of_birth"), estimate_without_term_frequencies=True)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("postcode"), estimate_without_term_frequencies=True)
    return linker, 0.99


def splink_historical(df):
    # docs/demos/examples/duckdb/deduplicate_50k_synthetic.ipynb @ v4.0.17
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("substr(first_name,1,3)", "substr(surname,1,4)"), block_on("surname", "dob"),
            block_on("first_name", "dob"), block_on("postcode_fake", "first_name"),
            block_on("postcode_fake", "surname"), block_on("dob", "birth_place"),
            block_on("substr(postcode_fake,1,3)", "dob"), block_on("substr(postcode_fake,1,3)", "first_name"),
            block_on("substr(postcode_fake,1,3)", "surname"),
            block_on("substr(first_name,1,2)", "substr(surname,1,2)", "substr(dob,1,4)")],
        comparisons=[cl.ForenameSurnameComparison("first_name", "surname",
                                                  forename_surname_concat_col_name="first_name_surname_concat"),
                     cl.DateOfBirthComparison("dob", input_is_string=True),
                     cl.PostcodeComparison("postcode_fake"),
                     cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
                     cl.ExactMatch("occupation").configure(term_frequency_adjustments=True)],
        retain_intermediate_calculation_columns=True)
    df = df.copy()
    df["first_name_surname_concat"] = df["first_name"] + " " + df["surname"]
    linker = Linker(df, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("first_name", "surname", "dob"),
         block_on("substr(first_name,1,2)", "surname", "substr(postcode_fake,1,2)"),
         block_on("dob", "postcode_fake")], recall=0.6)
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name", "surname"), estimate_without_term_frequencies=True)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("dob"), estimate_without_term_frequencies=True)
    return linker, 0.95


# ------------------------------------------------------------------ the four datasets
def febrl_prep(d):
    d = d.rename(columns=lambda x: x.strip())
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip().replace({"nan": ""})
    d["cluster"] = d["rec_id"].apply(lambda x: "-".join(x.split("-")[:2]))
    return d


def datasets():
    from splink import splink_datasets as _sd

    class _Frames:          # Splink 5 returns Arrow tables; Splink 4 pandas frames
        def __getattr__(self, name):
            t = getattr(_sd, name)
            return t.to_pandas() if hasattr(t, "to_pandas") else t
    sd = _Frames()
    fake = sd.fake_1000.copy()
    fake = fake.astype({"unique_id": str})
    for c in ("first_name", "surname", "dob", "city", "email"):
        fake[c] = fake[c].where(fake[c].notna(), None)
    hist = sd.historical_50k.copy()
    hist["unique_id"] = hist["unique_id"].astype(str)
    common_febrl = {"record_id": "rec_id", "first_name": "given_name", "last_name": "surname",
                    "addr_number": "street_number", "addr_street": "address_1", "addr_unit": "address_2",
                    "date_of_birth": "date_of_birth", "ssn": "soc_sec_id",
                    "postcode": "postcode", "suburb": "suburb", "state": "state"}
    febrl_extra = [{"name": "postcode", "compare": "exact"}, {"name": "suburb", "compare": "exact"},
                   {"name": "state", "compare": "exact"}]
    return {
        "fake_1000": {
            "frames": {"fake": fake}, "truth_col": "cluster", "id_col": "unique_id", "splink": splink_fake_1000,
            "columns": {"record_id": "unique_id", "first_name": "first_name", "last_name": "surname",
                        "date_of_birth": "dob", "email": "email", "city": "city"},
            "extra_fields": [{"name": "city", "compare": "exact"}],
            "extra_blocking": [["first_name", "city"], ["last_name"]]},
        "febrl3": {
            "frames": {"febrl3": febrl_prep(sd.febrl3)}, "truth_col": "cluster", "id_col": "rec_id",
            "splink": splink_febrl3, "columns": common_febrl, "extra_fields": febrl_extra,
            "extra_blocking": [["ssn"], ["first_name"], ["last_name"], ["date_of_birth"], ["postcode"]]},
        "febrl4": {
            "frames": {"a": febrl_prep(sd.febrl4a), "b": febrl_prep(sd.febrl4b)}, "truth_col": "cluster",
            "id_col": "rec_id", "splink": splink_febrl4, "columns": common_febrl, "extra_fields": febrl_extra,
            "extra_blocking": [["first_name", "last_name"], ["date_of_birth"], ["ssn"], ["state", "addr_street"],
                               ["addr_number", "addr_street"], ["postcode"]]},
        "historical_50k": {
            "frames": {"hist": hist}, "truth_col": "cluster", "id_col": "unique_id", "splink": splink_historical,
            "columns": {"record_id": "unique_id", "first_name": "first_name", "last_name": "surname",
                        "date_of_birth": "dob", "birth_place": "birth_place", "postcode": "postcode_fake",
                        "occupation": "occupation", "gender": "gender"},
            "extra_fields": [{"name": "birth_place", "compare": "exact"}, {"name": "postcode", "compare": "similarity"},
                             {"name": "occupation", "compare": "exact"}, {"name": "gender", "compare": "exact"}],
            "extra_blocking": [["last_name", "date_of_birth"], ["first_name", "date_of_birth"],
                               ["postcode", "first_name"], ["postcode", "last_name"],
                               ["date_of_birth", "birth_place"]]},
    }


def clustered_f1(labels_pred, truth):
    _, c = np.unique(truth, return_counts=True)
    return partition_scores(labels_pred, truth, float(np.sum(c * (c - 1) / 2)))


def run_dataset(name, spec, tmp):
    out = {}
    # ---- custmatch config over the same frames
    # custmatch as it was before its matcher options: rich features were off by default then
    cfg = {"sources": [], "extra_fields": spec["extra_fields"], "extra_blocking": spec["extra_blocking"],
           "rich_features": False}
    for src, frame in spec["frames"].items():
        path = tmp / f"{name}_{src}.csv"
        frame.fillna("").astype(str).to_csv(path, index=False)
        cfg["sources"].append({"name": src, "path": path.name, "columns": spec["columns"]})
    (tmp / f"{name}.json").write_text(json.dumps(cfg))
    config = tmp / f"{name}.json"
    df, _ = io.load(config)
    truth_of = {}
    for src, frame in spec["frames"].items():
        truth_of.update(dict(zip(src + ":" + frame[spec["id_col"]].astype(str), frame[spec["truth_col"]].astype(str))))
    keys = io.key(df)
    truth = np.array([truth_of[k] for k in keys])

    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(truth_of[a] == truth_of[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab

    def score_run(out_dir):
        c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
        return clustered_f1(c.customer_id.to_numpy(), np.array([truth_of[f"{s}:{r}"] for s, r in zip(c.source, c.record_id)]))

    n_cand = len(labels.candidates(df, cfg))
    budget = min(1000, n_cand)
    lp = tmp / f"{name}_labels.csv"
    labels.sample(config, budget, lp, seed=1)
    lab = fill(lp)
    t0 = time.time()
    s = pipeline.run(config, lp, tmp / f"{name}_out", log=lambda *_: None)
    out["custmatch_1000"] = score_run(tmp / f"{name}_out") | {"labels": int(len(lab)), "seconds": round(time.time() - t0, 1),
                                                          "candidate_pairs": s["candidate_pairs"]}
    if budget >= 1000:
        ap = tmp / f"{name}_active.csv"
        labels.sample(config, 300, ap, seed=1)
        al = fill(ap)
        while len(al) < 1000:
            nxt = tmp / "next.csv"
            labels.sample_active(config, ap, 100, nxt)
            al = pd.concat([al, fill(nxt)], ignore_index=True)
            al.to_csv(ap, index=False)
        pipeline.run(config, ap, tmp / f"{name}_out_a", log=lambda *_: None)
        out["custmatch_active"] = score_run(tmp / f"{name}_out_a")

    # ---- Splink, official settings, on the original frames
    frames = list(spec["frames"].values())
    t0 = time.time()
    linker, documented = spec["splink"](frames if len(frames) > 1 else frames[0])
    pred = linker.inference.predict(threshold_match_probability=0.001).as_pandas_dataframe()
    secs = round(time.time() - t0, 1)
    idx = {k: i for i, k in enumerate(keys)}
    srcs = list(spec["frames"])
    idc = "unique_id" if "unique_id_l" in pred.columns else "rec_id"
    if len(srcs) == 1:
        li = [idx[f"{srcs[0]}:{v}"] for v in pred[f"{idc}_l"].astype(str)]
        rj = [idx[f"{srcs[0]}:{v}"] for v in pred[f"{idc}_r"].astype(str)]
    else:   # link_only: Splink names the input tables; map each back to its source by record id
        where = {}
        for src, frame in spec["frames"].items():
            for v in frame[spec["id_col"]].astype(str):
                where.setdefault(v, src)
        li = [idx[f"{where[v]}:{v}"] for v in pred[f"{idc}_l"].astype(str)]
        rj = [idx[f"{where[v]}:{v}"] for v in pred[f"{idc}_r"].astype(str)]
    pairs = np.c_[np.minimum(li, rj), np.maximum(li, rj)].astype(np.int32)
    w, p = pred.match_weight.to_numpy(), pred.match_probability.to_numpy()
    n = len(df)
    if documented is not None:
        out["splink_documented"] = clustered_f1(components(n, pairs, p > documented), truth) | {
            "threshold_probability": documented, "seconds": secs}
    li_, lj_, ly = pipeline.read_labels(lp, df)
    lab_pairs = np.c_[np.minimum(li_, lj_), np.maximum(li_, lj_)]
    th, _ = pipeline.choose_threshold(n, pairs, lab_pairs, ly, [(w, np.ones(len(ly), bool))], WEIGHTS)
    out["splink_from_labels"] = clustered_f1(components(n, pairs, w > th), truth) | {"match_weight": th, "seconds": secs}
    best = max(((t, clustered_f1(components(n, pairs, w > t), truth)) for t in WEIGHTS), key=lambda z: z[1]["f1"])
    out["splink_ceiling"] = best[1] | {"match_weight": float(best[0])}
    if name == "fake_1000":
        # tutorial 07: "The data quality is very poor in this dataset, so we need looser blocking
        # rules to achieve decent recall" - reload the trained model with its looser rules
        from splink import DuckDBAPI, Linker, block_on
        model = linker.misc.save_model_to_json()
        model["blocking_rules_to_generate_predictions"] = [
            r.get_blocking_rule("duckdb").as_dict() for r in
            (block_on("first_name"), block_on("city"), block_on("email"), block_on("dob"))]
        loose = Linker(frames[0], model, db_api=DuckDBAPI())
        pr = loose.inference.predict(threshold_match_probability=0.001).as_pandas_dataframe()
        li2 = [idx[f"{srcs[0]}:{v}"] for v in pr["unique_id_l"].astype(str)]
        rj2 = [idx[f"{srcs[0]}:{v}"] for v in pr["unique_id_r"].astype(str)]
        pairs2 = np.c_[np.minimum(li2, rj2), np.maximum(li2, rj2)].astype(np.int32)
        w2, p2 = pr.match_weight.to_numpy(), pr.match_probability.to_numpy()
        out["splink_documented_loose_blocking"] = clustered_f1(components(n, pairs2, p2 > documented), truth)
        th2, _ = pipeline.choose_threshold(n, pairs2, lab_pairs, ly, [(w2, np.ones(len(ly), bool))], WEIGHTS)
        out["splink_loose_from_labels"] = clustered_f1(components(n, pairs2, w2 > th2), truth) | {"match_weight": th2}
        best2 = max(((t, clustered_f1(components(n, pairs2, w2 > t), truth)) for t in WEIGHTS), key=lambda z: z[1]["f1"])
        out["splink_loose_ceiling"] = best2[1] | {"match_weight": float(best2[0])}
    if name == "fake_1000":
        # custmatch configured for this dataset the way Splink's examples are: (1) the same
        # looser blocking tutorial 07 gives Splink; (2) also max_block raised so the 173-record
        # London block is not skipped, and fuzzy DOB / email - 58% of true pairs differ in DOB
        variants = {
            "loose_blocking": dict(cfg, extra_blocking=[["first_name"], ["city"], ["email"], ["date_of_birth"]]),
            "tuned": dict(cfg, extra_blocking=[["first_name"], ["city"], ["email"], ["date_of_birth"]],
                          max_block=200,
                          extra_fields=cfg["extra_fields"] + [{"name": "date_of_birth", "compare": "edit"},
                                                              {"name": "email", "compare": "similarity"}]),
        }
        for tag, vcfg in variants.items():
            (tmp / f"{name}_{tag}.json").write_text(json.dumps(vcfg))
            vpath = tmp / f"{name}_{tag}.json"
            lp2 = tmp / f"{name}_{tag}_labels.csv"
            labels.sample(vpath, min(1000, len(labels.candidates(df, vcfg))), lp2, seed=1)
            fill(lp2)
            pipeline.run(vpath, lp2, tmp / f"{name}_out_{tag}", log=lambda *_: None)
            out[f"custmatch_1000_{tag}"] = score_run(tmp / f"{name}_out_{tag}")
            ap2 = tmp / f"{name}_{tag}_active.csv"
            labels.sample(vpath, 300, ap2, seed=1)
            al2 = fill(ap2)
            while len(al2) < 1000:
                labels.sample_active(vpath, ap2, 100, tmp / "next.csv")
                al2 = pd.concat([al2, fill(tmp / "next.csv")], ignore_index=True)
                al2.to_csv(ap2, index=False)
            pipeline.run(vpath, ap2, tmp / f"{name}_out_{tag}_a", log=lambda *_: None)
            out[f"custmatch_active_{tag}"] = score_run(tmp / f"{name}_out_{tag}_a")
    return out


def main():
    which = sys.argv[1:] or ["fake_1000", "febrl3", "febrl4", "historical_50k"]
    path = ROOT / "research/results/splink_datasets_benchmark.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    specs = datasets()
    with tempfile.TemporaryDirectory() as tmp:
        for name in which:
            res[name] = run_dataset(name, specs[name], Path(tmp))
            print(name, json.dumps({k: v["f1"] for k, v in res[name].items()}), flush=True)
            path.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
