"""The custmatch tool: config handling, the end-to-end run, labelling, and everything after.

The data is generated here - people in two
sources with typos, households sharing an address and surname, and a junk email and phone
carried by unrelated people - so the run's output can be checked against known identities.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from custmatch import explain, incremental, io, labels, pipeline
from custmatch.core import partition_scores

FIRST = ["anna", "ben", "carla", "david", "erin", "frank", "grace", "henry", "iris", "jack",
         "kate", "liam", "maya", "noah", "olga", "paul", "quinn", "rosa", "sam", "tara"]
LAST = ["kofron", "alonso", "teague", "hixon", "burtness", "sutphin", "johnston", "thompson",
        "mclaren", "clausen", "gurney", "wyatt", "harrison", "goldsmith", "henderson"]
STREETS = ["145th st", "interlake ave n", "portulaca dr", "wisteria dr", "via rosa", "mallory ln"]


def typo(s, rng):
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = random.Random(0)
    tmp = tmp_path_factory.mktemp("cm")
    people, crm, billing = [], [], []
    for h in range(60):                       # 60 households, 1-3 people each
        last, street, num = rng.choice(LAST), rng.choice(STREETS), str(100 + h)
        for m in range(rng.randint(1, 3)):
            pid = f"p{len(people)}"
            first = FIRST[(h * 3 + m) % len(FIRST)]
            people.append((pid, f"h{h}"))
            dob = f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}/{rng.randint(1940,2005)}"
            email = f"{first}.{last}{h}{m}@mail.com"
            phone = f"206{rng.randint(1000000, 9999999)}"
            crm.append({"id": f"c{len(crm)}", "fname": first, "lname": last, "dob": dob,
                        "address": f"{num} {street}", "email": email, "phone": phone, "pid": pid})
            b_first = typo(first, rng) if rng.random() < 0.3 else first
            b_email, b_phone = email, phone
            if rng.random() < 0.15:           # junk contact values shared by unrelated people
                b_email, b_phone = "test@test.com", "1111111111"
            billing.append({"acct": f"b{len(billing)}", "first": b_first, "last": last,
                            "street_address": f"{num} {street}", "mail": b_email,
                            "tel": f"({b_phone[:3]}) {b_phone[3:6]}-{b_phone[6:]}", "pid": pid})
    pd.DataFrame(crm).to_csv(tmp / "crm.csv", index=False)
    pd.DataFrame(billing).to_csv(tmp / "billing.csv", index=False)
    cfg = {"sources": [
        {"name": "crm", "path": "crm.csv", "columns": {"record_id": "id", "first_name": "fname",
         "last_name": "lname", "date_of_birth": "dob", "address": "address", "email": "email", "phone": "phone"}},
        {"name": "billing", "path": "billing.csv", "columns": {"record_id": "acct", "first_name": "first",
         "last_name": "last", "address": "street_address", "email": "mail", "phone": "tel"}}],
        "source_priority": ["crm", "billing"]}
    (tmp / "config.json").write_text(json.dumps(cfg))
    truth = {**{f"crm:{r['id']}": r["pid"] for r in crm}, **{f"billing:{r['acct']}": r["pid"] for r in billing}}
    household = dict(people)
    return tmp, truth, household


def all_candidate_labels(tmp, truth, path):
    df, _ = io.load(tmp / "config.json")
    keys = io.key(df)
    pairs = labels.candidates(df)
    pd.DataFrame([{"a": keys[i], "b": keys[j], "label": int(truth[keys[i]] == truth[keys[j]])}
                  for i, j in pairs]).to_csv(path, index=False)
    return len(pairs)


# ------------------------------------------------------------------------------ config / io
def test_split_address():
    assert io.split_address("5112 145th St Apt 3") == ("5112", "145th st", "apt 3")
    assert io.split_address("Via Rosa") == ("", "via rosa", "")
    assert io.split_address("12  Main  Street") == ("12", "main street", "")


def test_load_normalises_and_maps(world):
    tmp, *_ = world
    df, cfg = io.load(tmp / "config.json")
    b = df[df.source == "billing"].iloc[0]
    assert b.phone.isdigit() and len(b.phone) == 10, "phone punctuation should be stripped"
    assert b.addr_number and b.addr_street, "a single address column should be split"
    assert (df[df.source == "billing"].date_of_birth == "").all(), "unmapped columns are missing"


@pytest.mark.parametrize("bad, message", [
    ({"sources": [{"name": "x", "path": "crm.csv", "columns": {"record_id": "id", "last_name": "lname"}}]},
     "required"),
    ({"sources": [{"name": "x", "path": "crm.csv", "columns": {"record_id": "id", "first_name": "fname",
                                                                "last_name": "nope"}}]}, "not in"),
    ({"sources": []}, "no 'sources'"),
])
def test_config_errors_are_explained(world, bad, message):
    tmp, *_ = world
    (tmp / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(io.ConfigError, match=message):
        io.load(tmp / "bad.json")


# ------------------------------------------------------------------------------ the run
def test_run_end_to_end(world):
    tmp, truth, household = world
    n_labels = all_candidate_labels(tmp, truth, tmp / "labels.csv")
    assert n_labels >= pipeline.MIN_LABELS
    s = pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "out")
    c = pd.read_csv(tmp / "out" / "clusters.csv", dtype=str)
    pid = np.array([truth[f"{a}:{b}"] for a, b in zip(c.source, c.record_id)])
    _, sizes = np.unique(pid, return_counts=True)
    score = partition_scores(c.customer_id.to_numpy(), pid, float(np.sum(sizes * (sizes - 1) / 2)))
    assert score["f1"] > 95, score
    assert s["largest_customer"] <= 3, "junk contacts must not chain unrelated people together"
    contacts = pd.read_csv(tmp / "out" / "shared_and_junk_contacts.csv", dtype=str)
    junk = set(contacts[contacts["class"] == "junk"].value)
    assert "test@test.com" in junk and "1111111111" in junk
    g = pd.read_csv(tmp / "out" / "golden_records.csv", dtype=str)
    assert len(g) == s["customers"] and g.customer_id.is_unique


def test_forest_catboost_runs_end_to_end_and_reuses_its_saved_model(world):
    pytest.importorskip("catboost")
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "labels.csv")
    cfg = json.loads((tmp / "config.json").read_text()) | {"matcher_model": "forest_catboost"}
    (tmp / "ensemble.json").write_text(json.dumps(cfg))
    s = pipeline.run(tmp / "ensemble.json", tmp / "labels.csv", tmp / "out_e", log=lambda *_: None)
    assert s["matcher_model"] == "forest_catboost"
    c = pd.read_csv(tmp / "out_e" / "clusters.csv", dtype=str)
    pid = np.array([truth[f"{a}:{b}"] for a, b in zip(c.source, c.record_id)])
    _, sizes = np.unique(pid, return_counts=True)
    assert partition_scores(c.customer_id.to_numpy(), pid, float(np.sum(sizes * (sizes - 1) / 2)))["f1"] > 95
    again = pipeline.run(tmp / "ensemble.json", None, tmp / "out_e2", log=lambda *_: None, model_dir=tmp / "out_e")
    assert again["threshold"] == s["threshold"] and again["customers"] == s["customers"]
    a, b = (f"{x}:{y}" for x, y in zip(c.source[:2], c.record_id[:2]))
    assert 0 <= explain.why(tmp / "out_e", a, b)["score"] <= 1
    added = incremental.add(tmp / "out_e", tmp / "ensemble.json", tmp / "out_e3")   # every record again, unchanged
    after = pd.read_csv(tmp / "out_e3" / "clusters.csv", dtype=str)
    assert added["new_records"] == 0 and len(after) == len(c)


def test_forest_probabilities_are_the_same_every_time():
    """sklearn's predict_proba sums the trees from threads in finishing order, so its last bit
    varied between runs and active learning asked different pairs. p_match must not."""
    from custmatch import models
    rng = np.random.default_rng(0)
    X = rng.random((2000, 6)).astype(np.float32)
    m = models.factory({})().fit(X, (X[:, 0] + rng.random(2000) / 2 > 0.8).astype(int))
    Z = rng.random((3 * models.PREDICT_CHUNK_THREAD + 7, 6)).astype(np.float32)
    first = models.p_match(m, Z)
    assert all((models.p_match(m, Z) == first).all() for _ in range(3))
    assert np.abs(first - m.predict_proba(Z)[:, 1]).max() < 1e-12
    assert (models.p_match(m, Z[:500]) == first[:500]).all()        # the small-batch path agrees


def test_forest_catboost_single_class_fold_predicts_that_class():
    pytest.importorskip("catboost")
    from custmatch.models import factory, p_match
    X = np.random.default_rng(0).random((20, 4))
    m = factory({"matcher_model": "forest_catboost"})().fit(X, np.ones(20, int))
    assert p_match(m, X).tolist() == [1.0] * 20 and m.predict_proba(X)[:, 1].tolist() == [1.0] * 20
    assert not hasattr(m, "feature_importances_")          # nothing was fitted


def test_a_saved_catboost_model_without_catboost_says_how_to_install(world, monkeypatch):
    pytest.importorskip("catboost")
    import pickle
    import sys

    from custmatch.models import factory, load_model
    X = np.random.default_rng(0).random((40, 4)); y = (X[:, 0] > .5).astype(int)
    state = world[0] / "cb_state"
    state.mkdir()
    (state / "model.pkl").write_bytes(pickle.dumps({"forest": factory({"matcher_model": "forest_catboost"})().fit(X, y)}))
    for name in [m for m in sys.modules if m == "catboost" or m.startswith("catboost.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "catboost", None)
    with pytest.raises(io.ConfigError, match=r"custmatch\[catboost\]"):
        load_model(state)


def test_forest_catboost_without_catboost_says_how_to_install(monkeypatch):
    import sys

    from custmatch.models import factory
    monkeypatch.setitem(sys.modules, "catboost", None)
    with pytest.raises(ValueError, match=r"custmatch\[catboost\]"):
        factory({"matcher_model": "forest_catboost"})


def test_threshold_is_chosen_after_clustering(world):
    """The chosen threshold must be the best closure-aware one, ties to the stricter."""
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "labels.csv")
    s = pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "out2")
    assert s["threshold"] in pipeline.THRESHOLDS
    assert s["cross_fitted_closure_aware"]["folds"] == pipeline.FOLDS


def test_too_few_labels_is_refused(world):
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "labels.csv")
    pd.read_csv(tmp / "labels.csv").head(50).to_csv(tmp / "few.csv", index=False)
    with pytest.raises(io.ConfigError, match="at least"):
        pipeline.run(tmp / "config.json", tmp / "few.csv", tmp / "out3")


# ------------------------------------------------------------------------------ active learning
def test_active_sampling_picks_new_uncertain_pairs(world):
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "all.csv")
    first = pd.read_csv(tmp / "all.csv").sample(300, random_state=0)
    first.to_csv(tmp / "first.csv", index=False)
    k = labels.sample_active(tmp / "config.json", tmp / "first.csv", 40, tmp / "active.csv")
    got = pd.read_csv(tmp / "active.csv", dtype=str)
    assert k == len(got) <= 40 and (got.by == "active").all()
    done = set(zip(first.a, first.b)) | set(zip(first.b, first.a))
    assert not any((a, b) in done for a, b in zip(got.a, got.b)), "must not re-ask labelled pairs"


def test_active_labels_train_but_never_choose_the_threshold(world, monkeypatch):
    """choose_threshold must only ever be judged on random (non-active) pairs."""
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "all.csv")
    lab = pd.read_csv(tmp / "all.csv")
    lab["by"] = np.where(np.arange(len(lab)) % 3 == 0, "active", "random")
    lab.to_csv(tmp / "mixed.csv", index=False)
    active = (lab.by == "active").to_numpy()
    real = pipeline.choose_threshold
    judged = []

    def spy(n, pairs, lab_pairs, ly, fold_scores, grid, **kw):
        judged.append(np.any([held for _, held in fold_scores], axis=0))
        return real(n, pairs, lab_pairs, ly, fold_scores, grid, **kw)
    monkeypatch.setattr(pipeline, "choose_threshold", spy)
    s = pipeline.run(tmp / "config.json", tmp / "mixed.csv", tmp / "out_m", log=lambda *_: None)
    assert s["labelled_pairs"]["active"] == int(active.sum())
    assert not (judged[0] & active).any() and judged[0][~active].all()


# ------------------------------------------------------------------------------ explanations
def test_links_explain_merges_weakest_first(world):
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "labels.csv")
    s = pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "out_l", log=lambda *_: None)
    L = pd.read_csv(tmp / "out_l" / "links.csv")
    assert len(L) == s["links"] > 0 and L.score.is_monotonic_increasing
    assert L.evidence.str.startswith("names ").all()
    c = pd.read_csv(tmp / "out_l" / "clusters.csv", dtype=str)
    cust = dict(zip(c.source + ":" + c.record_id, c.customer_id.astype(int)))
    assert all(cust[a] == cust[b] == k for a, b, k in zip(L.a, L.b, L.customer_id)), \
        "every link must sit inside the customer it reports"


# ------------------------------------------------------------------------------ incremental
def _split(tmp, frac=0.2):
    cfg = json.loads((tmp / "config.json").read_text())
    old, new = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
    for so, sn in zip(old["sources"], new["sources"]):
        d = pd.read_csv(tmp / so["path"], dtype=str, keep_default_na=False)
        is_new = np.random.default_rng(0).random(len(d)) < frac
        d[~is_new].to_csv(tmp / f"old_{so['path']}", index=False)
        d[is_new].to_csv(tmp / f"new_{so['path']}", index=False)
        so["path"], sn["path"] = f"old_{so['path']}", f"new_{sn['path']}"
    (tmp / "old.json").write_text(json.dumps(old))
    (tmp / "new.json").write_text(json.dumps(new))


def test_incremental_add_keeps_ids_and_matches_new_records(world):
    from custmatch import incremental
    tmp, truth, _ = world
    _split(tmp)
    all_candidate_labels(tmp, truth, tmp / "all.csv")
    old_keys = set(io.key(io.load(tmp / "old.json")[0]))
    lab = pd.read_csv(tmp / "all.csv")
    lab[lab.a.isin(old_keys) & lab.b.isin(old_keys)].to_csv(tmp / "old_labels.csv", index=False)
    pipeline.run(tmp / "old.json", tmp / "old_labels.csv", tmp / "inc0", log=lambda *_: None)
    before = pd.read_csv(tmp / "inc0" / "clusters.csv", dtype=str)
    s = incremental.add(tmp / "inc0", tmp / "new.json", tmp / "inc1")
    after = pd.read_csv(tmp / "inc1" / "clusters.csv", dtype=str)
    assert s["new_records"] == len(after) - len(before) > 0
    kept = after.merge(before, on=["source", "record_id"], suffixes=("", "_before"))
    merged_away = set()
    for row in pd.read_csv(tmp / "inc1" / "merges.csv", dtype=str).itertuples():
        merged_away |= set(row.merged_customer_ids.split(";")) - {row.into_customer_id}
    moved = kept[kept.customer_id != kept.customer_id_before]
    assert moved.customer_id_before.isin(merged_away).all(), "existing ids change only by logged merges"
    pid = np.array([truth[f"{a}:{b}"] for a, b in zip(after.source, after.record_id)])
    _, sizes = np.unique(pid, return_counts=True)
    got = partition_scores(after.customer_id.to_numpy(), pid, float(np.sum(sizes * (sizes - 1) / 2)))
    assert got["f1"] > 95, got


def test_incremental_add_treats_existing_records_as_updates_and_refuses_ditto_state(world):
    from custmatch import incremental
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "labels.csv")
    pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "incd", log=lambda *_: None)
    before = pd.read_csv(tmp / "incd" / "clusters.csv", dtype=str)
    s = incremental.add(tmp / "incd", tmp / "config.json", tmp / "incd2")   # every record again, unchanged
    after = pd.read_csv(tmp / "incd2" / "clusters.csv", dtype=str)
    assert s["updated_records"] == len(before) and s["new_records"] == 0 and len(after) == len(before)
    both = before.merge(after, on=["source", "record_id"], suffixes=("", "_after"))
    assert (both.customer_id == both.customer_id_after).mean() > 0.97, "unchanged records moved customers"


# ------------------------------------------------------------------------------ schema extras
def test_extra_fields_add_one_feature_each_and_can_fuzz_standard_columns(world):
    from custmatch import schema
    from custmatch.core import block, features, profile_contacts
    tmp, *_ = world
    cfg = json.loads((tmp / "config.json").read_text())
    cfg["extra_fields"] = [{"name": "date_of_birth", "compare": "edit"},
                           {"name": "email", "compare": "similarity"}]
    (tmp / "extra.json").write_text(json.dumps(cfg))
    df, cfg = io.load(tmp / "extra.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs, _ = block(df, prof)
    X = features(df, pairs, prof, extra_fields=schema.extra_fields(cfg))
    assert X.shape[1] == 15
    both_dob = (df.date_of_birth.to_numpy()[pairs[:, 0]] != "") & (df.date_of_birth.to_numpy()[pairs[:, 1]] != "")
    assert ((X[both_dob, 13] >= 0) & (X[both_dob, 13] <= 1)).all() and (X[~both_dob, 13] == -1).all()
    assert (df.date_of_birth != "").any(), "naming a standard column as an extra must not blank it"


def test_extra_blocking_and_max_block(world):
    from custmatch.core import block, profile_contacts
    tmp, *_ = world
    df, _ = io.load(tmp / "config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    base, _ = block(df, prof)
    wider, _ = block(df, prof, extra_keys=(("last_name",),))
    assert len(wider) > len(base) and set(map(tuple, base.tolist())) <= set(map(tuple, wider.tolist()))
    big = df.last_name.value_counts().max()
    capped, hubs_capped = block(df, prof, extra_keys=(("last_name",),), max_block=big - 1)
    kept, hubs_kept = block(df, prof, extra_keys=(("last_name",),), max_block=big)
    assert hubs_capped > hubs_kept and len(kept) >= len(capped)
    cfg = json.loads((tmp / "config.json").read_text())
    (tmp / "badkey.json").write_text(json.dumps(cfg | {"extra_blocking": [["nope"]]}))
    with pytest.raises(io.ConfigError, match="unknown field"):
        io.load(tmp / "badkey.json")


def test_cluster_coherence_sees_chaining_that_labelled_pairs_miss():
    """Two people chained by one false link: the chained cluster's own implied pairs are mostly
    different people, and the estimate must say so."""
    # records 0-2 are person A, 3-5 person B; candidate links within each and one false 2-3
    pairs = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [2, 3]], dtype=np.int32)
    score = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    person = np.array([0, 0, 0, 1, 1, 1])
    predict = lambda p: (person[p[:, 0]] == person[p[:, 1]]).astype(float)
    chained = pipeline.cluster_coherence(6, pairs, score, 0.5, predict, k=4000)
    assert 0.3 < chained < 0.5, chained          # 6 of 15 implied pairs are true: 0.4
    split = pipeline.cluster_coherence(6, pairs, np.array([0.9, 0.9, 0.9, 0.9, 0.1]), 0.5, predict)
    assert split == 1.0


# ------------------------------------------------------------------------------ beating Splink
def test_rich_features_add_nine_columns_and_parse_every_date_format(world):
    from custmatch.core import block, features, parse_dates, profile_contacts
    tmp, *_ = world
    df, _ = io.load(tmp / "config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs, _ = block(df, prof)
    X, R = features(df, pairs, prof), features(df, pairs, prof, rich=True)
    assert R.shape == (len(pairs), X.shape[1] + 9) and np.allclose(R[:, :13], X)
    got = parse_dates(["1971-06-24", "19560409", "05/06/1994", "1630-08-01", "", "n/a"])
    assert got.tolist() == [[1971, 6, 24], [1956, 4, 9], [1994, 5, 6], [1630, 8, 1], [-1, -1, -1], [-1, -1, -1]]


def test_active_mix_splits_a_batch_between_uncertain_and_random(world):
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "all.csv")
    pd.read_csv(tmp / "all.csv").sample(300, random_state=0).to_csv(tmp / "first.csv", index=False)
    labels.sample_active(tmp / "config.json", tmp / "first.csv", 40, tmp / "mixed_batch.csv", mix=0.5)
    got = pd.read_csv(tmp / "mixed_batch.csv", dtype=str)
    assert (got.by == "random").sum() == 20 and (got.by == "active").sum() == 20
    assert not got.duplicated(["a", "b"]).any()


def test_match_weight_ranks_matches_without_labels_and_reloads(world):
    """The Fellegi-Sunter weight is fitted with no labels; it must rank true pairs above false
    ones, and a scorer rebuilt from the saved model gives the same weights in any pair order."""
    from custmatch import matchweight
    from custmatch.core import profile_contacts
    tmp, truth, _ = world
    df, cfg = io.load(tmp / "config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs = labels.candidates(df, cfg)
    (w,), scorer = matchweight.weights(df, prof, [pairs], cfg, return_scorer=True)
    keys = io.key(df)
    y = np.array([truth[keys[i]] == truth[keys[j]] for i, j in pairs])
    assert np.isfinite(w).all() and w[y].mean() > w[~y].mean() + 5
    again = matchweight.load_scorer(df, prof, json.loads(json.dumps(scorer.model_json))).weights(pairs[::-1])[::-1]
    assert np.allclose(w, again)


def test_match_weight_feature_run_add_and_explain(world):
    """With "match_weight_feature" the saved model scores new pairs in `add` and in `explain`."""
    from custmatch import explain, incremental
    tmp, truth, _ = world
    _split(tmp)
    all_candidate_labels(tmp, truth, tmp / "all.csv")
    old = set(io.key(io.load(tmp / "old.json")[0]))
    lab = pd.read_csv(tmp / "all.csv")
    lab[lab.a.isin(old) & lab.b.isin(old)].to_csv(tmp / "old_lab_mw.csv", index=False)
    for name in ("old", "new"):
        c = json.loads((tmp / f"{name}.json").read_text()) | {"match_weight_feature": True}
        (tmp / f"{name}_mw.json").write_text(json.dumps(c))
    pipeline.run(tmp / "old_mw.json", tmp / "old_lab_mw.csv", tmp / "out_mw", log=lambda *_: None)
    s = incremental.add(tmp / "out_mw", tmp / "new_mw.json", tmp / "out_mw2")
    after = pd.read_csv(tmp / "out_mw2" / "clusters.csv", dtype=str)
    assert s["new_records"] > 0
    pid = np.array([truth[f"{a}:{b}"] for a, b in zip(after.source, after.record_id)])
    _, sizes = np.unique(pid, return_counts=True)
    assert partition_scores(after.customer_id.to_numpy(), pid, float(np.sum(sizes * (sizes - 1) / 2)))["f1"] > 95
    links = pd.read_csv(tmp / "out_mw2" / "links.csv")
    assert explain.why(tmp / "out_mw2", links.a[0], links.b[0])["score"] > 0



# ------------------------------------------------------------------------------ blocking v2
def test_oversized_blocks_are_split_not_dropped(world):
    from custmatch import blocking
    from custmatch.core import profile_contacts
    tmp, *_ = world
    df, cfg = io.load(tmp / "config.json")
    keys = blocking.Keys(df, {f: profile_contacts(df, f) for f in ("email", "phone")})
    rule = [("last_name:exact",)]
    big = int(df.last_name.value_counts().max())
    pairs, rep = blocking.generate(df, keys, rule, max_block=big - 1)
    r = rep["rules"][0]
    assert r["oversized_blocks_split"] >= 1
    # every pair produced still agrees on the rule: splitting only refines
    ln = df.last_name.to_numpy()
    assert (ln[pairs[:, 0]] == ln[pairs[:, 1]]).all()
    unlimited, _ = blocking.generate(df, keys, rule, max_block=10**9)
    assert set(map(tuple, pairs.tolist())) <= set(map(tuple, unlimited.tolist()))
    assert rep["candidate_pairs"] == len(pairs) and r["new_pairs"] == len(pairs)


def test_learned_rules_reach_their_coverage_target(world):
    from custmatch import blocking
    from custmatch.core import profile_contacts
    tmp, truth, _ = world
    df, cfg = io.load(tmp / "config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    keys = blocking.Keys(df, prof)
    k = io.key(df)
    person = np.array([truth[x] for x in k])
    pos = np.array([(i, j) for i in range(len(df)) for j in range(i + 1, len(df)) if person[i] == person[j]])
    rules, rep = blocking.learn_rules(df, keys, pos, cfg, target=0.95)
    assert rules and rep["covered"] >= 0.95 * rep["coverable_by_library"]
    pairs, _ = blocking.generate(df, keys, rules)
    found = set(map(tuple, pairs.tolist()))
    assert sum(tuple(p) in found for p in pos.tolist()) >= 0.9 * len(pos)


def test_learned_blocking_run_and_incremental_reuse_rules(world):
    from custmatch import incremental
    tmp, truth, _ = world
    _split(tmp)
    all_candidate_labels(tmp, truth, tmp / "all.csv")
    old = set(io.key(io.load(tmp / "old.json")[0]))
    lab = pd.read_csv(tmp / "all.csv")
    lab[lab.a.isin(old) & lab.b.isin(old)].to_csv(tmp / "old_lab.csv", index=False)
    for name in ("old", "new"):
        c = json.loads((tmp / f"{name}.json").read_text()) | {"blocking": "learned"}
        (tmp / f"{name}_learned.json").write_text(json.dumps(c))
    s = pipeline.run(tmp / "old_learned.json", tmp / "old_lab.csv", tmp / "lb0", log=lambda *_: None)
    rep = json.loads((tmp / "lb0" / "blocking_report.json").read_text())
    assert rep["mode"] == "learned" and rep["learned"]["rules"] and s["candidate_pairs"] == rep["candidate_pairs"]
    added = incremental.add(tmp / "lb0", tmp / "new_learned.json", tmp / "lb1")
    assert added["new_records"] > 0


def test_disagreement_sampling_offers_pairs_blocking_missed(world):
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "all.csv")
    pd.read_csv(tmp / "all.csv").sample(300, random_state=0).to_csv(tmp / "first.csv", index=False)
    labels.sample_active(tmp / "config.json", tmp / "first.csv", 40, tmp / "dis.csv", strategy="disagreement")
    got = pd.read_csv(tmp / "dis.csv", dtype=str)
    done = set(zip(pd.read_csv(tmp / "first.csv").a, pd.read_csv(tmp / "first.csv").b))
    assert len(got) <= 40 and not any((a, b) in done for a, b in zip(got.a, got.b))
    cand = set(map(tuple, pd.read_csv(tmp / "all.csv")[["a", "b"]].values.tolist()))
    assert (got.by == "active").all()
    # whatever came from outside blocking really was outside it
    outside = [(a, b) for a, b in zip(got.a, got.b) if (a, b) not in cand and (b, a) not in cand]
    assert len(outside) <= 20


def test_recut_breaks_a_chained_customer_and_confidence_is_bounded():
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    score = np.array([0.99, 0.99, 0.6, 0.99, 0.99])
    customers = np.zeros(6, dtype=np.int64)
    labels_, recut = pipeline.recut_large(6, pairs, score, 0.5, customers, 3, pipeline.THRESHOLDS)
    assert recut == 1 and np.bincount(np.unique(labels_, return_inverse=True)[1]).max() <= 3
    conf = pipeline.confidence(6, pairs, score, labels_)
    assert ((conf >= 0) & (conf <= 1)).all()


def _people(rows):
    cols = ["first_name", "last_name", "addr_number", "addr_street", "addr_unit", "email", "phone",
            "date_of_birth", "middle_initial"]
    return pd.DataFrame([dict(zip(cols, r)) | {"middle_initial": ""} for r in rows], columns=cols)


def test_contacts_are_profiled_over_customers_not_first_names():
    from custmatch import households
    from custmatch.core import profile_contacts
    # one person, a nickname in one source: records say "different first names", customers do not
    df = _people([("robert", "hale", "12", "elm st", "", "rh@mail.com", "", "", ""),
                  ("bob", "hale", "12", "elm st", "", "rh@mail.com", "", "", ""),
                  ("ann", "cole", "9", "oak rd", "", "cole@mail.com", "", "", ""),
                  ("tom", "cole", "9", "oak rd", "", "cole@mail.com", "", "", "")])
    assert profile_contacts(df, "email")["rh@mail.com"][0] == "household"
    prof = households.profile_by_customer(df, "email", [0, 0, 1, 2])
    assert prof["rh@mail.com"][0] == "personal" and prof["cole@mail.com"][0] == "household"


def test_address_keys_standardise_fuzz_the_street_and_fill_a_missing_unit():
    from custmatch import households
    df = _people([("a", "x", "12", "Elm Street", "Apt 4", "", "", "", ""),
                  ("b", "x", "012", "elm st.", "#4", "", "", "", ""),
                  ("c", "x", "12", "elmm st", "", "", "", "", ""),       # typo, no unit
                  ("d", "y", "13", "elm st", "4", "", "", "", ""),       # another house
                  ("e", "z", "", "", "", "", "", "", "")])
    k = households.address_keys(df)
    assert k[0] == k[1] == k[2] != k[3] and k[4] == ""


def test_households_merge_on_a_shared_contact_and_surname_up_to_the_cap():
    from custmatch import households
    # eight Lees at eight addresses sharing a household phone, plus a Kim sharing it too
    rows = [(f"p{i}", "lee", str(i + 1), "main rd", "", "", "2065550199" if i < 8 else "", "", "")
            for i in range(9)] + [("kim", "park", "99", "hill rd", "", "", "2065550199", "", "")]
    df = _people(rows)
    profiles = {"email": {}, "phone": {"2065550199": ("household", 0.35)}}
    hh, _ = households.assign(df, np.arange(len(df)), profiles)
    lees = np.bincount(hh[:8])
    assert lees.max() == households.MERGE_CAP          # merged, but never above the cap
    assert hh[8] not in hh[:8]                        # no shared contact: own address
    assert hh[9] not in hh[:9]                        # shared contact, different surname


def test_placeholder_names_and_addresses_are_blanked_part_by_part():
    from custmatch import placeholders
    assert placeholders.name_flags("john", "doe") == (True, True)
    assert placeholders.name_flags("maria", "unknown") == (False, True)
    for first, last in (("i", "wyatt"), ("sir", "john"), ("john", "iii"), ("william", "tester"),
                        ("4rthur", "lee"), ("aa", "bo")):
        assert placeholders.name_flags(first, last) == (False, False), (first, last)
    assert placeholders.address_pattern("123", "main st") and placeholders.address_pattern("0", "elm st")
    assert not placeholders.address_pattern("1234", "main st")   # a real address
    assert not placeholders.address_pattern("12", "maine st")
    df = _people([("maria", "unknown", "123", "fake st", "2", "", "", "1990-01-01", ""),
                  ("maria", "lopez", "5", "oak rd", "", "", "", "1990-01-01", "")])
    out, fake_name, fake_addr = placeholders.blank(df)
    assert out.first_name.tolist() == ["maria", "maria"] and out.last_name.tolist() == ["", "lopez"]
    assert out.addr_street.tolist() == ["", "oak rd"] and out.addr_unit[0] == ""
    assert fake_name.tolist() == [True, False] and fake_addr.tolist() == [True, False]


def test_a_full_name_on_many_unrelated_records_is_a_placeholder():
    from custmatch import placeholders
    shared = [("pat", "smithers", str(i), f"street {i}", "", "", "", f"19{50 + i}-01-01", "") for i in range(6)]
    family = [("jo", "banks", "7", "elm st", "", "", "", f"19{60 + i}-02-02", "") for i in range(6)]
    fake_name, _ = placeholders.detect(_people(shared + family))
    assert fake_name[:6].all() and not fake_name[6:].any()


def test_reserved_fictional_phone_numbers_are_blanked_only_with_placeholders_on():
    from custmatch import placeholders
    from custmatch.core import syntactic_junk
    assert not syntactic_junk("2065550142") and syntactic_junk("5551234567")
    df = _people([("ann", "lee", "5", "oak rd", "", "", p, "", "") for p in ("2065550142", "2065550242")])
    out, _, _ = placeholders.blank(df)
    assert out.phone.tolist() == ["", "2065550242"]


def test_country_normalisation_makes_australian_formats_agree():
    from custmatch import country
    df = _people([("mick", "", "", "", "", "Mia.Lee+shop@GMAIL.com".lower(), p, d, "")
                  for p, d in (("0412 345 670".replace(" ", ""), "11/09/2005"), ("61412345670", "2005-09-11"),
                               ("412345670", ""))])
    out = country.normalise(df, {"country": "AU"})
    assert out.phone.nunique() == 1 and out.phone[0] == "61412345670"
    assert out.date_of_birth.tolist()[:2] == ["2005-09-11", "2005-09-11"]
    assert out.email[0] == "mialee@gmail.com" and out.email_user[0] == "mialee"
    assert "michael" in out.first_roots[0].split("|")
    assert country.normalise(df, {}).equals(df)                       # off without a country
    us = country.normalise(df, {"country": "US", "normalise": {"phone": False}})
    assert us.date_of_birth[0] == "2005-11-09"                         # month first in the US


def test_domain_typos_are_fixed_but_real_domains_are_not():
    from custmatch import country
    for typo, real in (("gmial.com", "gmail.com"), ("gmai.com", "gmail.com"), ("hotmail.con", "hotmail.com"),
                       ("yaho.com.au", "yahoo.com.au"), ("iclod.com", "icloud.com")):
        assert country.fix_domain(typo) == real, typo
    for real in ("ge.com", "ms.com", "cloud.com", "aon.com", "mail.com", "email.com", "gmx.com", "example.com"):
        assert country.fix_domain(real) == real, real          # real domains a frequency rule rewrote
    assert country.email("jo.bloggs+shop@gmial.com") == "jobloggs@gmail.com"


def test_fake_numbers_on_sight_and_patterned_numbers():
    """Reserved and impossible numbers are junk on sight; numbers that only look made up - which
    people also buy as "gold" numbers - are patterned, and need strangers to share them."""
    from custmatch.core import PATTERNED_NUMBERS, syntactic_junk, syntactic_junk_mask
    for fake in ("0000000000", "5551234567", "61491570156", "61255501234"):
        assert syntactic_junk(fake), fake
    patterned = ["61400000000", "0400000000", "61412345678", "0411111111", "0412121212", "61412121212",
                 "0123123123", "3128675309", "0423456789", "0498765432"]
    assert syntactic_junk_mask(patterned, PATTERNED_NUMBERS).all()
    assert not any(syntactic_junk(v) for v in patterned)
    real = ["61412345670", "61293745521", "61400514096", "0412312345", "2065550142"]
    assert not syntactic_junk_mask(real, PATTERNED_NUMBERS).any() and not any(syntactic_junk(v) for v in real)


def test_a_bought_gold_number_stays_personal_until_strangers_share_it():
    from custmatch import households
    from custmatch.core import profile_contacts
    own = [("ann", "lee", "4", "elm st", "", f"ann{k}@mail.com", "0411111111", "", "") for k in range(5)]
    strangers = [(f, s, "", "", "", f"{f}@mail.com", "0412121212", "", "")
                 for f, s in (("bob", "wu"), ("cal", "diaz"), ("dee", "ross"))]
    nameless = [("", "", "", "", "", f"u{k}@mail.com", "0423456789", "", "") for k in range(3)]
    df = _people(own + strangers + nameless)
    prof = profile_contacts(df, "phone")
    assert prof["0411111111"][0] != "junk"          # one person's gold number, on all her records
    assert prof["0412121212"][0] == "junk"          # the same pattern typed by three strangers
    assert prof["0423456789"][0] == "junk"          # three nameless sign-ups with different emails
    cust = households.profile_by_customer(df, "phone", np.arange(len(df)))
    assert cust["0411111111"][0] != "junk" and cust["0412121212"][0] == "junk"


def test_blank_addresses_do_not_make_a_junk_email_look_like_a_household():
    from custmatch import households
    from custmatch.core import profile_contacts
    firsts = ["ann", "ben", "cal", "dee", "eve", "fay", "gus", "hal", "ivy", "jo"]
    rows = [(f, "", "", "", "", "na@na.com", "", "", "") for f in firsts]            # web sign-ups
    rows += [("kim", "wu", "4", "elm st", "", "wu@mail.com", "", "", ""),
             ("lee", "wu", "4", "elm st", "", "wu@mail.com", "", "", "")]             # a real family email
    df = _people(rows)
    prof = profile_contacts(df, "email")
    assert prof["na@na.com"][0] == "junk" and prof["wu@mail.com"][0] == "household"
    cust = households.profile_by_customer(df, "email", np.arange(len(df)))
    assert cust["na@na.com"][0] == "junk" and cust["wu@mail.com"][0] == "household"


def test_a_made_up_mobile_shared_by_nameless_sign_ups_is_junk():
    """People reuse the same fake mobile, and it collects their real emails. With no names to
    disagree, only the many different emails show it is not one person's number."""
    from custmatch import households
    from custmatch.core import profile_contacts
    rows = [("", "", "", "", "", f"user{k}@mail.com", "3128675309", "", "") for k in range(10)]
    rows += [("ann", "lee", "4", "elm st", "", f"lee{k}@mail.com", "2065550199", "", "") for k in range(9)]  # one family
    rows += [("bob", "wu", "", "", "", "bob.wu@mail.com", "4155550123", "", "")]
    df = _people(rows)
    prof = profile_contacts(df, "phone")
    assert prof["3128675309"][0] == "junk"
    assert prof["2065550199"][0] != "junk" and prof["4155550123"][0] == "personal"
    cust = households.profile_by_customer(df, "phone", np.arange(len(df)))
    assert cust["3128675309"][0] == "junk" and cust["2065550199"][0] != "junk"


def test_suspect_records_are_flagged_without_flagging_real_names():
    from custmatch import suspect
    words = frozenset({"fuck", "shit", "hooker", "two girls one cup"}) - suspect.REAL_NAMES
    df = pd.DataFrame({
        "first_name": ["fuck", "john", "mary", "bob", "ann", "joe", "lee", "two", "na'shota"],
        "last_name": ["you", "hooker", "dickens", "scunthorpe", "smith", "", "wu", "girls one cup", "lytle"],
        "email": ["a@mail.com", "jhooker@mail.com", "mary.dickens@mail.com", "bob@mail.com", "ann@mailinator.com",
                  "xk7fq2zr9p@gmail.com", "shit.happens@mail.com", "b@mail.com", "leewu1984@mail.com"]})
    got = suspect.flag(df, words)
    assert got == {0: ["profanity in name"], 7: ["profanity in name"], 6: ["profanity in email"],
                   4: ["disposable email"], 5: ["random-looking email"]}
    assert suspect.flag(df.assign(first_name="xkfq"), None).get(5) is None   # built from the name: xk7fq...
    assert not suspect.flag(df, None).get(8)            # leewu1984: built from a name, not random


def test_suspect_records_are_written_and_never_change_matching(world, monkeypatch):
    from custmatch import suspect
    tmp, truth, _ = world
    all_candidate_labels(tmp, truth, tmp / "labels.csv")
    crm = pd.read_csv(tmp / "crm.csv", dtype=str)
    crm.loc[0, "email"] = "xk7fq2zr9p@mailinator.com"
    crm.to_csv(tmp / "crm_s.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    cfg["sources"][0]["path"] = "crm_s.csv"
    (tmp / "suspect.json").write_text(json.dumps(cfg))
    (tmp / "suspect_off.json").write_text(json.dumps(cfg | {"suspect_records": False}))
    monkeypatch.setattr(suspect, "profanity_words", lambda: frozenset({"fuck"}))
    on = pipeline.run(tmp / "suspect.json", tmp / "labels.csv", tmp / "out_s", log=lambda *_: None)
    off = pipeline.run(tmp / "suspect_off.json", tmp / "labels.csv", tmp / "out_s_off", log=lambda *_: None)
    t = pd.read_csv(tmp / "out_s" / "suspect_records.csv", dtype=str)
    assert on["suspect_records"] == 1 and t.record_id.tolist() == [crm.id[0]] and "disposable" in t.reasons[0]
    assert not (tmp / "out_s_off" / "suspect_records.csv").exists() and off["suspect_records"] == 0
    a = pd.read_csv(tmp / "out_s" / "clusters.csv", dtype=str)
    b = pd.read_csv(tmp / "out_s_off" / "clusters.csv", dtype=str)
    assert a.equals(b)                                   # a flag for review, never a change
    monkeypatch.setattr(suspect, "profanity_words", lambda: None)   # the list could not be downloaded
    s = pipeline.run(tmp / "suspect.json", tmp / "labels.csv", tmp / "out_s2", log=lambda *_: None)
    assert s["suspect_records"] == 1 and any("profanity list could not be downloaded" in w for w in s["warnings"])


def test_suspect_records_are_found_before_names_and_emails_are_hashed(world, monkeypatch):
    from custmatch import suspect
    tmp, _, _ = world
    crm = pd.read_csv(tmp / "crm.csv", dtype=str)
    crm.loc[0, "fname"], crm.loc[1, "email"] = "fuck", "ann@mailinator.com"
    crm.to_csv(tmp / "crm_h.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    cfg["sources"][0]["path"] = "crm_h.csv"
    cfg["hash_fields"] = ["first_name", "last_name", "email"]
    monkeypatch.setenv("CUSTMATCH_SALT", "test-salt")
    monkeypatch.setattr(suspect, "profanity_words", lambda: frozenset({"fuck"}))
    df, _ = io.load_config(cfg, tmp)
    assert df.first_name.iloc[0] != "fuck"             # hashed
    assert df.attrs["suspect"] == {0: ["profanity in name"], 1: ["disposable email"]}


def test_large_blocks_are_kept_only_when_their_pairs_match():
    """An oversized block whose sampled pairs the model calls matches is one big entity; one whose
    pairs it rejects (junk, or many people) stays skipped; the pair budget caps what is added."""
    from custmatch.blocking import check_large_blocks
    entity, junk = list(range(0, 100)), list(range(100, 250))
    same = np.zeros(250, int)
    same[100:] = np.arange(150) + 1                           # 0-99 one entity; 100-249 all different
    predict = lambda p: (same[p[:, 0]] == same[p[:, 1]]).astype(float)   # noqa: E731
    pairs, rows = check_large_blocks([(("x", "acme"), entity), (("email", "a@a.com"), junk)], predict, 0.5,
                                     budget=10_000)
    assert len(pairs) == 100 * 99 // 2 and set(np.unique(pairs)) == set(entity)
    assert [r["kept"] for r in sorted(rows, key=lambda r: r["records"])] == [True, False]
    none, rows = check_large_blocks([(("x", "acme"), entity)], predict, 0.5, budget=100)
    assert len(none) == 0 and not rows[0]["kept"]             # over the budget: not added


def test_a_large_organisation_is_one_customer_in_run_explain_and_add(world, tmp_path):
    """70 records of one organisation share every blocking key, so every block is over max_block
    and skipped as a hub; the large-block check keeps them, and `explain` and `add` agree."""
    import shutil
    tmp, truth, _ = world
    for f in ("crm.csv", "billing.csv", "config.json"):
        shutil.copy(tmp / f, tmp_path / f)
    crm = pd.read_csv(tmp_path / "crm.csv", dtype=str)
    org = pd.DataFrame([{"id": f"acme{k}", "fname": "acme", "lname": "industries", "dob": "",
                         "address": "1 acme plaza", "email": f"buyer{k}@acme-industries.com",
                         "phone": "2065550000", "pid": "acme"} for k in range(70)])
    pd.concat([crm, org]).to_csv(tmp_path / "crm.csv", index=False)
    truth = truth | {f"crm:acme{k}": "acme" for k in range(70)}
    all_candidate_labels(tmp_path, truth, tmp_path / "labels.csv")
    s = pipeline.run(tmp_path / "config.json", tmp_path / "labels.csv", tmp_path / "out", log=lambda *_: None)
    rep = json.loads((tmp_path / "out" / "blocking_report.json").read_text())
    assert any(r["kept"] for r in rep["large_blocks_checked"]) and rep["large_block_pairs_added"] > 0
    c = pd.read_csv(tmp_path / "out" / "clusters.csv", dtype=str)
    acme = c[c.record_id.str.startswith("acme")]
    assert acme.customer_id.nunique() == 1 and len(acme) == 70
    assert explain.why(tmp_path / "out", "crm:acme1", "crm:acme2")["reason"].startswith("matched directly")
    incremental.add(tmp_path / "out", tmp_path / "config.json", tmp_path / "out2")   # every record again
    c2 = pd.read_csv(tmp_path / "out2" / "clusters.csv", dtype=str)
    assert c2[c2.record_id.str.startswith("acme")].customer_id.nunique() == 1
    off = json.loads((tmp_path / "config.json").read_text()) | {"check_large_blocks": False}
    (tmp_path / "off.json").write_text(json.dumps(off))
    pipeline.run(tmp_path / "off.json", tmp_path / "labels.csv", tmp_path / "out_off", log=lambda *_: None)
    c3 = pd.read_csv(tmp_path / "out_off" / "clusters.csv", dtype=str)
    assert c3[c3.record_id.str.startswith("acme")].customer_id.nunique() == 70   # never compared
    assert s["customers"] < pd.read_csv(tmp_path / "out_off" / "clusters.csv").customer_id.nunique()


@pytest.mark.parametrize("name", ["forest", "forest_catboost"])
def test_every_matcher_model_gives_probabilities(name):
    from custmatch.models import factory
    if name == "forest_catboost":
        pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)); y = (X[:, 0] > 0).astype(int)
    p = factory({"matcher_model": name})().fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (200,) and ((p >= 0) & (p <= 1)).all() and np.corrcoef(p, y)[0, 1] > 0.5
    with pytest.raises(ValueError, match="matcher_model"):
        factory({"matcher_model": "svm"})


def test_names_in_chinese_and_arabic_script_are_transliterated():
    from custmatch import country
    df = pd.DataFrame({"first_name": ["محمد", "mohammed", "王伟", "伟", "wei"],
                       "last_name": ["حداد", "haddad", "", "王", "wang"]})
    out = country.transliterate(df)
    assert out.first_name.tolist()[2:] == ["wei", "wei", "wei"] and out.last_name.tolist()[2:] == ["wang"] * 3
    assert out.name_skeleton[0] == out.name_skeleton[1] == "mhmd|hd"
    assert out.name_script.tolist() == [True, False, True, True, False]
    latin = pd.DataFrame({"first_name": ["mia"], "last_name": ["lee"]})
    assert "name_skeleton" not in country.transliterate(latin).columns     # nothing changes without them
    for roman, arabic in (("fatima", "ftmh"), ("chahine", "shhyn"), ("youssef", "ywsf"), ("abdallah", "`bdllh")):
        assert country.skeleton(roman) == country.skeleton(arabic), roman


def test_postcode_is_read_off_an_australian_address():
    from custmatch import country
    df = _people([("a", "b", "3/12", "high st, richmond vic 3121", "", "", "", "", ""),
                  ("c", "d", "4", "elm st", "", "", "", "", "")]).assign(source="s", record_id=["1", "2"], ssn="")
    out = country.normalise(df, {"country": "AU", "normalise": {"phone": False}})
    assert out.postcode.tolist() == ["3121", ""]


def test_households_ignore_blank_surnames_and_postcode_join_is_opt_in():
    from custmatch import households
    rows = [("ann", "", "", "", "", "", "0411222333", "", ""), ("bob", "", "", "", "", "", "0411222333", "", ""),
            ("cal", "lee", "5", "oak rd", "", "", "", "", ""), ("dee", "lee", "", "", "", "", "", "", "")]
    df = _people(rows).assign(postcode=["", "", "3121", "3121"])
    prof = {"email": {}, "phone": {"0411222333": ("household", 0.35)}}
    hh, _ = households.assign(df, np.arange(4), prof)
    assert hh[0] == hh[1]                 # the shared household contact is their home
    assert hh[3] != hh[2]                 # no address, no contact: on their own by default
    hh, _ = households.assign(df, np.arange(4), prof, postcode_join=True)
    assert hh[3] == hh[2]                 # same surname, same postcode, one household there


# ------------------------------------------------------------ regressions from the adversarial review
def _world_copy(world, tmp_path, extra_cfg=None, extra_rows=()):
    """The world's files in tmp_path, config overrides, extra crm rows (with their own truth)."""
    import shutil
    src, truth, _ = world
    for f in ("crm.csv", "billing.csv"):
        shutil.copy(src / f, tmp_path / f)
    truth = dict(truth)
    if extra_rows:
        crm = pd.read_csv(tmp_path / "crm.csv", dtype=str, keep_default_na=False)
        crm = pd.concat([crm, pd.DataFrame(list(extra_rows))], ignore_index=True)
        crm.to_csv(tmp_path / "crm.csv", index=False)
        truth.update({f"crm:{r['id']}": r["pid"] for r in extra_rows})
    cfg = json.loads((src / "config.json").read_text()) | (extra_cfg or {})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path, truth


def _labels_from_truth(cfg_path, truth, out):
    df, cfg = io.load(cfg_path)
    keys = io.key(df)
    pairs = labels.candidates(df, cfg)
    pd.DataFrame([{"a": keys[i], "b": keys[j], "label": int(truth[keys[i]] == truth[keys[j]])}
                  for i, j in pairs]).to_csv(out, index=False)


def test_labels_as_spreadsheets_write_them_are_read_and_unknown_ones_refused(world, tmp_path):
    tmp, truth, _ = world
    df, _ = io.load(tmp / "config.json")
    keys = io.key(df)
    rows = [{"a": keys[0], "b": keys[1], "label": v} for v in ("1.0", "True", "yes", "0.0", "FALSE", "n")]
    pd.DataFrame(rows).to_csv(tmp_path / "l.csv", index=False)
    _, _, y = pipeline.read_labels(tmp_path / "l.csv", df)
    assert y.tolist() == [True, True, True, False, False, False]
    pd.DataFrame([{"a": keys[0], "b": keys[1], "label": "maybe"}]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(io.ConfigError, match="unrecognised label"):
        pipeline.read_labels(tmp_path / "bad.csv", df)


def _run_then_add(world, tmp_path, extra_cfg, new_rows):
    """Run on the world with extra_cfg, then `add` a batch of new crm rows."""
    from custmatch import incremental
    tmp, truth = _world_copy(world, tmp_path, extra_cfg)
    _labels_from_truth(tmp / "config.json", truth, tmp / "labels.csv")
    pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "run", log=lambda *_: None)
    pd.DataFrame(new_rows).to_csv(tmp / "new_crm.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    new = {"sources": [dict(cfg["sources"][0], path="new_crm.csv")]}          # no country etc.
    (tmp / "new.json").write_text(json.dumps(new))
    return incremental.add(tmp / "run", tmp / "new.json", tmp / "add"), tmp


def test_add_reads_new_records_with_the_saved_runs_settings(world, tmp_path):
    rows = [{"id": "n1", "fname": "伟", "lname": "王", "dob": "01/02/1980", "address": "9 elm st",
             "email": "wei.wang@mail.com", "phone": "+61 412 345 670", "pid": "x1"},
            {"id": "n2", "fname": "mia", "lname": "o’brien", "dob": "", "address": "", "email": "",
             "phone": "0412 345 670", "pid": "x1"}]
    s, tmp = _run_then_add(world, tmp_path, {"country": "AU"}, rows)
    assert s["new_records"] == 2
    rec = pd.read_parquet(tmp / "add" / "state" / "records.parquet")
    new = rec[rec.record_id.isin(["n1", "n2"])]
    assert new.first_name.tolist() == ["wei", "mia"]                 # transliterated with the saved settings
    assert set(new.phone) == {"61412345670"}                         # normalised although new.json has no country
    assert rec.first_roots.notna().all()


def test_add_blanks_placeholders_when_the_run_did(world, tmp_path):
    rows = [{"id": f"n{k}", "fname": "john", "lname": "doe", "dob": f"0{k}/01/19{50 + k}", "address": f"{k} x st",
             "email": "", "phone": "", "pid": f"x{k}"} for k in (1, 2)]
    s, tmp = _run_then_add(world, tmp_path, {"placeholders": True}, rows)
    rec = pd.read_parquet(tmp / "add" / "state" / "records.parquet")
    assert (rec[rec.record_id.isin(["n1", "n2"])][["first_name", "last_name"]] == "").all().all()
    c = pd.read_csv(tmp / "add" / "clusters.csv", dtype=str)
    got = c[c.record_id.isin(["n1", "n2"])].customer_id
    assert got.nunique() == 2, "two unrelated 'john doe' records were matched on a placeholder"


def test_add_respects_the_saved_threshold_and_max_cluster(world, tmp_path):
    import pickle

    from custmatch import incremental
    tmp, truth = _world_copy(world, tmp_path, {"max_cluster": 2})
    _labels_from_truth(tmp / "config.json", truth, tmp / "labels.csv")
    pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "run", log=lambda *_: None)
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False).head(5)
    crm["id"] = "dup" + crm["id"]                                     # copies of existing people
    crm.to_csv(tmp / "new_crm.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    (tmp / "new.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="new_crm.csv")]}))
    s = incremental.add(tmp / "run", tmp / "new.json", tmp / "add")
    assert s["links_accepted"] + s["links_refused_by_max_cluster"] > 0
    before = pd.read_csv(tmp / "run" / "clusters.csv", dtype=str).customer_id.value_counts()
    after = pd.read_csv(tmp / "add" / "clusters.csv", dtype=str)
    grew = after.customer_id.value_counts()
    for cid, size in grew.items():
        assert size <= max(2, before.get(cid, 0)), "add grew a customer past max_cluster"
    with open(tmp / "run" / "state" / "model.pkl", "rb") as f:
        model = pickle.load(f)
    model["threshold"] = 1.0             # exact copies score 1.0; nothing is above 1.0
    with open(tmp / "run" / "state" / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    assert incremental.add(tmp / "run", tmp / "new.json", tmp / "add2")["links_accepted"] == 0


def test_add_merges_take_the_smallest_id(world):
    tmp = world[0]
    merges = tmp / "inc1" / "merges.csv"
    if not merges.exists():
        pytest.skip("runs after test_incremental_add_keeps_ids_and_matches_new_records")
    after = pd.read_csv(tmp / "inc1" / "clusters.csv", dtype=str)
    for row in pd.read_csv(merges, dtype=str).itertuples():
        ids = row.merged_customer_ids.split(";")
        assert row.into_customer_id == str(min(map(int, ids)))
        assert not after.customer_id.isin(set(ids) - {row.into_customer_id}).any()


def test_splitting_a_big_block_keeps_records_without_the_refining_key():
    from custmatch.blocking import Keys, generate
    rows = [(f, "smith", str(k), "elm st", "", "", "", "", "") for k, f in enumerate(["al"] * 40 + ["alan"] * 40)]
    df = _people(rows).assign(source="s", record_id=[str(k) for k in range(80)], ssn="")
    pairs, _ = generate(df, Keys(df), [("last_name:exact",)], max_block=60)
    al = set(range(40))
    assert any(a in al and b in al for a, b in pairs), "the 'al' records lost every pair"


def test_fixed_and_learned_blocking_keep_the_country_keys():
    from custmatch import country
    from custmatch.blocking import candidates_for
    from custmatch.core import profile_contacts
    df = _people([("bill", "jones", "1", "elm st", "", "", "", "1970-01-01", ""),
                  ("william", "jones", "9", "oak rd", "", "", "", "1970-01-01", "")]).assign(
        source="s", record_id=["1", "2"], ssn="")
    df = country.normalise(df, {"country": "AU"})
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    for mode in (None, "fixed", "learned"):
        cfg = {"country": "AU"} | ({"blocking": mode} if mode else {})
        pairs = candidates_for(df, prof, cfg)[0]
        assert [0, 1] in pairs.tolist(), mode


def test_household_streets_differing_in_direction_are_not_merged():
    from custmatch import households
    df = _people([("a", "x", "12", "n main st", "", "", "", "", ""), ("b", "y", "12", "s main st", "", "", "", "", ""),
                  ("c", "x", "12", "park rd", "", "", "", "", ""), ("d", "y", "12", "park dr", "", "", "", "", ""),
                  ("e", "x", "12", "wisteria dr", "", "", "", "", ""), ("f", "y", "12", "wistera dr", "", "", "", "", "")])
    k = households.address_keys(df)
    assert k[0] != k[1] and k[4] == k[5]
    assert k[2] == k[3]        # street types are not compared (measured: guarding them cost recall)


def test_households_do_not_merge_two_blank_surnames(world):
    from custmatch import households
    df = _people([("ann", "", "1", "elm st", "", "", "0411222333", "", ""),
                  ("bob", "", "2", "oak rd", "", "", "0411222333", "", "")])
    prof = {"email": {}, "phone": {"0411222333": ("household", 0.35)}}
    hh, _ = households.assign(df, np.arange(2), prof)
    assert hh[0] != hh[1]


def test_typed_parquet_columns_are_read_as_text(tmp_path):
    pd.DataFrame({"id": ["1", "2"], "fname": ["ann", "bob"], "phone": [412345678.0, np.nan],
                  "dob": pd.to_datetime(["1990-05-01", None])}).to_parquet(tmp_path / "s.parquet")
    (tmp_path / "c.json").write_text(json.dumps({"sources": [{"name": "s", "path": "s.parquet", "columns": {
        "record_id": "id", "first_name": "fname", "phone": "phone", "date_of_birth": "dob"}}]}))
    df, _ = io.load(tmp_path / "c.json")
    assert df.phone.tolist() == ["412345678", ""] and df.date_of_birth.tolist() == ["1990-05-01", ""]


def test_australian_address_forms_are_split():
    assert io.split_address("12 main st #4") == ("12", "main st", "#4")
    assert io.split_address("unit 4, 12 main st") == ("12", "main st", "4")
    assert io.split_address("u4 12 main st") == ("12", "main st", "4")
    assert io.split_address("4/12 main st") == ("12", "main st", "4")


def test_active_sampling_with_one_label_class_falls_back_to_random(world, tmp_path):
    tmp, truth, _ = world
    df, _ = io.load(tmp / "config.json")
    keys = io.key(df)
    pairs = labels.candidates(df)[:20]
    pd.DataFrame([{"a": keys[i], "b": keys[j], "label": 0} for i, j in pairs]).to_csv(tmp_path / "l.csv", index=False)
    k = labels.sample_active(tmp / "config.json", tmp_path / "l.csv", 10, tmp_path / "next.csv")
    got = pd.read_csv(tmp_path / "next.csv")
    assert k == 10 and (got.by == "random").all()


def test_no_candidate_pairs_is_explained(tmp_path):
    rows = [{"id": str(k), "fname": f"f{k}", "lname": f"l{k}"} for k in range(300)]
    pd.DataFrame(rows).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "c.json").write_text(json.dumps({"sources": [{"name": "s", "path": "s.csv", "columns": {
        "record_id": "id", "first_name": "fname", "last_name": "lname"}}]}))
    pd.DataFrame([{"a": f"s:{k}", "b": f"s:{k + 1}", "label": int(k % 5 == 0)} for k in range(0, 250)]).to_csv(
        tmp_path / "l.csv", index=False)
    with pytest.raises(io.ConfigError, match="no candidate pairs"):
        pipeline.run(tmp_path / "c.json", tmp_path / "l.csv", tmp_path / "out", log=lambda *_: None)


def test_junk_contacts_are_never_blocking_keys():
    from custmatch import country
    from custmatch.blocking import FIXED_RULES, Keys, generate
    from custmatch.core import block, profile_contacts
    rows = [(f"p{k}", f"surname{k}", str(k), f"street {k}", "", "noreply@shop.com", "", "", "") for k in range(12)]
    df = _people(rows).assign(source="s", record_id=[str(k) for k in range(12)], ssn="")
    df = country.normalise(df, {"country": "AU", "normalise": {"phone": False}})
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    assert prof["email"]["noreply@shop.com"][0] == "junk"
    assert len(block(df, prof)[0]) == 0
    assert len(generate(df, Keys(df, prof), FIXED_RULES, max_block=60)[0]) == 0
    assert not [k for k, _ in country.blocking_keys(df, prof) if k[0] == "eu"]


def test_choose_threshold_takes_the_middle_of_thresholds_the_labels_cannot_tell_apart():
    n, pairs = 4, np.array([[0, 1], [2, 3]], dtype=np.int32)
    ly = np.array([True, False])
    folds = [(np.array([0.9, 0.1]), np.ones(2, bool))]
    # every threshold from 0.1 to 0.85 is perfect on these labels
    th, _ = pipeline.choose_threshold(n, pairs, pairs, ly, folds, pipeline.THRESHOLDS)
    # one held-out match: anything within 50 F1 points of the best is indistinguishable - here 0.05
    # (F1 66.7) to 0.85 - and the middle of that is taken, not the strictest edge
    tied = [t for t in pipeline.THRESHOLDS if t <= 0.85]
    assert th == tied[len(tied) // 2]
    th, _ = pipeline.choose_threshold(n, pairs, pairs, ly, folds, pipeline.THRESHOLDS, rule="strictest")
    assert th == 0.85                                  # the old rule is still available


def test_choose_threshold_lets_the_in_cluster_check_overrule_labelled_pairs():
    n, pairs = 6, np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    ly = np.array([True, True, False])
    folds = [(np.array([0.4, 0.9, 0.1]), np.ones(3, bool))]
    loose, _ = pipeline.choose_threshold(n, pairs, pairs, ly, folds, pipeline.THRESHOLDS)
    th, metrics = pipeline.choose_threshold(n, pairs, pairs, ly, folds, pipeline.THRESHOLDS,
                                            coherence=lambda t: 0.3 if t < 0.5 else 1.0)
    assert loose < 0.4 <= th and "labelled_pair_precision" in metrics


def test_placeholders_on_in_run_keeps_two_john_does_apart(world, tmp_path):
    extra = [{"id": f"jd{k}", "fname": "john", "lname": "doe", "dob": f"0{k}/0{k}/19{60 + k}",
              "address": "123 main st", "email": "", "phone": "", "pid": f"jd{k}"} for k in (1, 2)]
    tmp, truth = _world_copy(world, tmp_path, {"placeholders": True}, extra)
    _labels_from_truth(tmp / "config.json", truth, tmp / "labels.csv")
    s = pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "ph", log=lambda *_: None)
    c = pd.read_csv(tmp / "ph" / "clusters.csv", dtype=str)
    assert c[c.record_id.isin(["jd1", "jd2"])].customer_id.nunique() == 2
    assert s["placeholder_names"] >= 2 and (tmp / "ph" / "placeholders.csv").exists()


def test_blanked_placeholder_names_leave_no_name_features(world):
    from custmatch import country, placeholders
    df = _people([("test", "smith", "1", "elm st", "", "", "", "", "")]).assign(source="s", record_id=["1"], ssn="")
    df = country.normalise(df, {"country": "AU"})
    assert df.first_roots[0] == "test"
    out, _, _ = placeholders.blank(df)
    assert out.first_name[0] == "" and out.first_roots[0] == ""


def test_cli_commands(world, tmp_path):
    from custmatch import cli
    tmp, truth = _world_copy(world, tmp_path)
    cfg = str(tmp / "config.json")
    assert cli.main(["sample", cfg, "--n", "30", "--out", str(tmp / "s.csv")]) == 0
    assert len(pd.read_csv(tmp / "s.csv")) == 30
    assert cli.main(["blocking", cfg]) == 0
    _labels_from_truth(tmp / "config.json", truth, tmp / "labels.csv")
    assert cli.main(["run", cfg, "--labels", str(tmp / "labels.csv"), "--out", str(tmp / "out")]) == 0
    assert cli.main(["run", cfg, "--labels", str(tmp / "nope.csv"), "--out", str(tmp / "o2")]) == 2


def test_confidence_is_higher_for_strongly_linked_records():
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    conf = pipeline.confidence(4, pairs, np.array([0.99, 0.55]), np.array([0, 0, 1, 1]))
    assert conf[0] > conf[2]


def test_the_bundled_demo_runs(tmp_path):
    """README's first command works from a fresh clone: the demo ships its data and labels."""
    root = Path(__file__).resolve().parents[1] / "examples" / "demo"
    s = pipeline.run(root / "config.json", root / "labels.csv", tmp_path / "out", log=lambda *_: None)
    c = pd.read_csv(tmp_path / "out" / "clusters.csv", dtype=str)
    frames = [pd.read_csv(root / f, dtype=str, keep_default_na=False).assign(source=f[:-4])
              for f in ("web.csv", "app.csv", "store.csv")]
    truth = pd.concat(frames)
    pid = dict(zip(truth.source + ":" + truth.record_id, truth.person_id))
    t = np.array([pid[f"{a}:{b}"] for a, b in zip(c.source, c.record_id)])
    _, n = np.unique(t, return_counts=True)
    assert s["records"] == len(truth) and partition_scores(c.customer_id.to_numpy(), t, float(np.sum(n * (n - 1) / 2)))["f1"] > 70


# ------------------------------------------------------------------------------ country profiles
def _country_df(rows, cols=("first_name", "last_name", "addr_street", "date_of_birth", "phone", "ssn", "country")):
    base = {c: "" for c in ("record_id", "first_name", "middle_initial", "last_name", "date_of_birth", "addr_number",
                             "addr_street", "addr_unit", "email", "phone", "ssn", "country")}
    return pd.DataFrame([base | {"record_id": str(k), "source": "s"} | dict(zip(cols, r)) for k, r in enumerate(rows)])


def test_us_ssns_that_are_never_issued_are_not_evidence():
    from custmatch.countries import ssn
    assert ssn("123-45-6789") == "" and ssn("078-05-1120") == "" and ssn("111111111") == ""
    assert ssn("000-12-3456") == "" and ssn("666-12-3456") == "" and ssn("123-00-4567") == "" and ssn("123-45-0000") == ""
    assert ssn("536-22-1234") == "536221234" and ssn("912-78-1234") == "912781234"     # an ITIN is real
    assert ssn("53622123") == ""                                                      # too short


def test_each_country_reads_its_own_formats():
    from custmatch import country
    rows = [("ann", "lee", "400 pine st, seattle wa 98101-1234", "05/06/1994", "(206) 555-7812", "536-22-1234", "US"),
            ("bec", "lee", "3/12 high st, richmond vic 3121", "05/06/1994", "0412 345 670", "", "Australia"),
            ("tom", "ng", "10 downing st, london sw1a 2aa", "", "", "", "uk"),
            ("guy", "roy", "24 sussex dr, ottawa on k1m 1m4", "", "", "", "CA")]
    df = country.normalise(_country_df(rows), {"country": ["AU", "US", "GB", "CA"]})
    assert df.postcode.tolist() == ["98101", "3121", "sw1a2aa", "k1m1m4"]
    assert df.date_of_birth.tolist()[:2] == ["1994-05-06", "1994-06-05"]      # month first only in the US
    assert df.phone.tolist()[:2] == ["12065557812", "61412345670"]
    assert df.ssn[0] == "536221234"
    assert "rebecca" in df.first_roots[1].split("|")                           # the Australian list


def test_the_australian_nickname_list_only_where_it_applies():
    from custmatch import country
    for c, expect in (("AU", True), ("NZ", True), ("GB", True), ("US", False)):
        df = country.normalise(_country_df([("shazza", "lee", "", "", "", "", "")]), {"country": c})
        assert ("sharon" in df.first_roots[0].split("|")) == expect, c
    df = country.normalise(_country_df([("con", "georgiou", "", "", "", "", "")]), {"country": "AU"})
    assert "konstantinos" in df.first_roots[0].split("|")


def test_date_order_can_be_set_for_a_country_that_uses_both():
    from custmatch import country
    df = _country_df([("a", "b", "", "05/06/1994", "", "", "")])
    assert country.normalise(df, {"country": "CA"}).date_of_birth[0] == "1994-06-05"
    assert country.normalise(df, {"country": "CA", "date_order": "mdy"}).date_of_birth[0] == "1994-05-06"


def test_a_country_column_is_read_from_the_source(tmp_path):
    pd.DataFrame({"id": ["1", "2"], "fname": ["ann", "bob"], "dob": ["05/06/1994", "05/06/1994"],
                  "ctry": ["us", "au"]}).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "c.json").write_text(json.dumps({"country": ["AU", "US"], "sources": [{"name": "s", "path": "s.csv",
        "columns": {"record_id": "id", "first_name": "fname", "date_of_birth": "dob", "country": "ctry"}}]}))
    df, _ = io.load(tmp_path / "c.json")
    assert df.date_of_birth.tolist() == ["1994-05-06", "1994-06-05"]


# ------------------------------------------------------------------------------ operating over time
def _ran(world, tmp_path, extra=None, out="run"):
    tmp, truth = _world_copy(world, tmp_path, extra)
    _labels_from_truth(tmp / "config.json", truth, tmp / "labels.csv")
    s = pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / out, log=lambda *_: None)
    c = pd.read_csv(tmp / out / "clusters.csv", dtype=str)
    return tmp, truth, s, dict(zip(c.source + ":" + c.record_id, c.customer_id))


def test_overrides_join_and_unmerge_and_add_respects_them(world, tmp_path):
    from custmatch import incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    by = pd.Series(cust).groupby(pd.Series(cust)).groups
    pair = next(sorted(v) for v in by.values() if len(v) >= 2)          # two records of one customer
    other = next(k for k in cust if cust[k] != cust[pair[0]])          # a record of another customer
    pd.DataFrame([{"a": pair[0], "b": pair[1], "decision": "different"},
                  {"a": pair[0], "b": other, "decision": "same"}]).to_csv(tmp / "overrides.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text()) | {"overrides": "overrides.csv"}
    (tmp / "o.json").write_text(json.dumps(cfg))
    s = pipeline.run(tmp / "o.json", tmp / "labels.csv", tmp / "ov", log=lambda *_: None)
    c = pd.read_csv(tmp / "ov" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert got[pair[0]] != got[pair[1]], "cannot-link did not unmerge"
    assert got[pair[0]] == got[other], "must-link did not join"
    assert s["overrides"]["cannot_link"] == 1 and s["overrides"]["must_link"] == 1
    # `add` keeps them apart too: re-adding pair[1]'s record cannot rejoin pair[0]'s customer
    src, rid = pair[1].split(":")
    raw = pd.read_csv(tmp / f"{src}.csv", dtype=str, keep_default_na=False)
    raw[raw.iloc[:, 0] == rid].to_csv(tmp / "again.csv", index=False)
    new = {"sources": [dict(s_, path="again.csv") for s_ in cfg["sources"] if s_["name"] == src]}
    (tmp / "again.json").write_text(json.dumps(new))
    incremental.add(tmp / "ov", tmp / "again.json", tmp / "ov2")
    c2 = pd.read_csv(tmp / "ov2" / "clusters.csv", dtype=str)
    got2 = dict(zip(c2.source + ":" + c2.record_id, c2.customer_id))
    assert got2[pair[0]] != got2[pair[1]]


def test_trusted_ids_link_records_and_ignore_placeholder_values():
    from custmatch import stewardship
    df = pd.DataFrame({"loyalty": ["L1", "L1", "L2", "", "X", "X", "X", "X", "X", "X"]})
    pairs = stewardship.trusted_pairs(df, {"trusted_ids": ["loyalty"]})
    assert pairs.tolist() == [[0, 1]]            # "X" on six records is a placeholder, not an ID


def test_review_band_and_history_and_ids_kept_across_reruns(world, tmp_path):
    tmp, truth, s, first = _ran(world, tmp_path)
    review = pd.read_csv(tmp / "run" / "review.csv")
    assert s["review_pairs"] == len(review) and (abs(review.score - s["threshold"]) < 0.1).all()
    s2 = pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "run2", previous_dir=tmp / "run",
                      log=lambda *_: None)
    c = pd.read_csv(tmp / "run2" / "clusters.csv", dtype=str)
    second = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert first == second, "a rerun on the same data changed customer IDs"
    assert (tmp / "run2" / "history.csv").exists()


def test_carry_ids_logs_merges_and_splits():
    from custmatch import stewardship
    keys = np.array(["a", "b", "c", "d", "e"])
    previous = {"a": 7, "b": 7, "c": 9, "d": 9, "e": 9}
    ids, events = stewardship.carry_ids(keys, np.zeros(5, int), previous)      # 7 and 9 joined
    assert set(ids) == {9} and [e[1] for e in events] == ["merged"]           # 9 held more records
    ids, events = stewardship.carry_ids(keys, np.array([0, 0, 1, 1, 2]), previous)  # 9 divided
    assert ids[0] == 7 and ids[2] == 9 and ids[4] not in (7, 9)
    assert {e[1] for e in events} == {"new", "split"}


def test_model_reuse_scores_without_labels(world, tmp_path):
    tmp, truth, s, first = _ran(world, tmp_path)
    s2 = pipeline.run(tmp / "config.json", None, tmp / "reuse", model_dir=tmp / "run", log=lambda *_: None)
    c = pd.read_csv(tmp / "reuse" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    same = pd.Series(first).groupby(pd.Series(first)).apply(lambda g: len({got[k] for k in g.index}) == 1)
    assert same.mean() > 0.99 and s2["threshold"] == s["threshold"] and s2["model_reused_from"]


def test_add_deletes_and_updates_and_forget(world, tmp_path):
    from custmatch import cli, incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    by = pd.Series(cust).groupby(pd.Series(cust)).groups
    two = [sorted(v) for v in by.values() if len(v) >= 2]
    gone, big = two[0], two[1]
    pd.DataFrame([{"source": gone[0].split(":")[0], "record_id": gone[0].split(":")[1]}]).to_csv(tmp / "del.csv", index=False)
    assert cli.main(["forget", str(tmp / "del.csv"), "--state", str(tmp / "run"), "--out", str(tmp / "fg")]) == 0
    c = pd.read_csv(tmp / "fg" / "clusters.csv", dtype=str)
    assert gone[0] not in set(c.source + ":" + c.record_id)
    rec = pd.read_parquet(tmp / "fg" / "state" / "records.parquet")
    assert gone[0] not in set(rec.source + ":" + rec.record_id), "erased record still in the saved state"
    h = pd.read_csv(tmp / "fg" / "history.csv", dtype=str)
    assert (h.event == "record_deleted").any()
    # an update: the same key with a different person's details moves it out of its customer
    src, rid = big[0].split(":")
    raw = pd.read_csv(tmp / f"{src}.csv", dtype=str, keep_default_na=False)
    row = raw[raw.iloc[:, 0] == rid].copy()
    for col in row.columns[1:]:
        if col != "pid":
            row[col] = "zz" + row[col].astype(str) if row[col].iloc[0] else row[col]
    row.to_csv(tmp / "upd.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    (tmp / "upd.json").write_text(json.dumps({"sources": [dict(s_, path="upd.csv") for s_ in cfg["sources"] if s_["name"] == src]}))
    s = incremental.add(tmp / "fg", tmp / "upd.json", tmp / "up")
    assert s["updated_records"] == 1
    c2 = pd.read_csv(tmp / "up" / "clusters.csv", dtype=str)
    got = dict(zip(c2.source + ":" + c2.record_id, c2.customer_id))
    assert got[big[0]] != got[big[1]], "an updated record kept its old customer despite new details"


def test_explain_says_why_and_why_not(world, tmp_path):
    from custmatch import explain
    tmp, truth, s, cust = _ran(world, tmp_path)
    links = pd.read_csv(tmp / "run" / "links.csv", dtype=str)
    a, b = links.a.iloc[-1], links.b.iloc[-1]
    w = explain.why(tmp / "run", a, b)
    assert w["same_customer"] and w["compared_by_blocking"] and "above the threshold" in w["reason"]
    far = [k for k in cust if cust[k] != cust[a]][0]
    w2 = explain.why(tmp / "run", a, far)
    assert not w2["same_customer"] and ("never compared" in w2["reason"] or "below the threshold" in w2["reason"])


def test_survivorship_by_field():
    from custmatch.golden import build_golden
    df = pd.DataFrame({"source": ["crm", "web", "app"], "email": ["old@x.com", "new@x.com", ""],
                       "phone": ["1", "2", "3"], "addr_number": ["1", "9", ""], "addr_street": ["elm st", "oak rd", ""],
                       "addr_unit": ["", "", ""], "updated_at": ["2020-01-01T00:00:00", "2026-01-01T00:00:00", "2025-01-01T00:00:00"]})
    rules = {"email": "most_recent", "phone": {"prefer": ["app", "crm"]}, "address": "most_recent"}
    g = build_golden(df, np.zeros(3, int), rules, ["email", "phone", "addr_number", "addr_street"])[0]
    assert g["email"] == "new@x.com" and g["phone"] == "3"
    assert (g["addr_number"], g["addr_street"]) == ("9", "oak rd")          # one record's address, whole


def test_hashed_matching_keeps_accuracy_and_no_plain_contacts(world, tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTMATCH_SALT", "test-secret")
    tmp, truth, s, cust = _ran(world, tmp_path, {"hash_fields": ["email", "phone"], "minimal_outputs": True}, out="h")
    rec = pd.read_parquet(tmp / "h" / "state" / "records.parquet")
    assert not rec.email.str.contains("@").any() and not rec.phone.str.match(r"^\d+$").any()
    assert not (tmp / "h" / "golden_records.csv").exists()
    pid = np.array([truth[k] for k in cust])
    _, n = np.unique(pid, return_counts=True)
    assert partition_scores(np.array(list(cust.values())), pid, float(np.sum(n * (n - 1) / 2)))["f1"] > 95
    monkeypatch.delenv("CUSTMATCH_SALT")
    with pytest.raises(io.ConfigError, match="CUSTMATCH_SALT"):
        io.load(tmp / "config.json") if False else io.load_config(
            json.loads((tmp / "config.json").read_text()) | {"hash_fields": ["email"]}, tmp)


# ------------------------------------------------------------------------------ order dependence
def test_hysteresis_keeps_a_borderline_decision():
    from custmatch import stewardship
    keys = np.array(["a", "b", "c", "d"])
    pairs = np.array([[0, 1], [2, 3]])
    prev_links = pd.DataFrame({"a": ["a"], "b": ["b"]})                    # a-b linked last time, c-d not
    s = stewardship.hysteresis(pairs, np.array([0.48, 0.52]), keys, prev_links, ["a", "b", "c", "d"], 0.05)
    assert s[0] > 0.5 and s[1] < 0.5                                         # both keep their last decision
    s = stewardship.hysteresis(pairs, np.array([0.30, 0.70]), keys, prev_links, ["a", "b", "c", "d"], 0.05)
    assert s[0] < 0.5 and s[1] > 0.5                                         # clear evidence still wins


# ------------------------------------------------------------------------------ code-review fixes
def test_a_state_saved_before_links_survives_two_adds(world, tmp_path):
    from custmatch import incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    (tmp / "run" / "state" / "links.parquet").unlink()                   # an older state
    cfg = json.loads((tmp / "config.json").read_text())
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    for k in (1, 2):
        crm.head(2).assign(id=[f"new{k}a", f"new{k}b"]).to_csv(tmp / f"b{k}.csv", index=False)
        (tmp / f"b{k}.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path=f"b{k}.csv")]}))
    incremental.add(tmp / "run", tmp / "b1.json", tmp / "a1")
    incremental.add(tmp / "a1", tmp / "b2.json", tmp / "a2")
    c = pd.read_csv(tmp / "a2" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    groups = pd.Series(cust).groupby(pd.Series(cust)).groups
    kept = [len({got[k] for k in ks}) == 1 for ks in groups.values() if len(ks) > 1]
    assert np.mean(kept) > 0.99, "the second add broke the old customers apart"


def test_a_retired_customer_id_is_never_given_out_again(world, tmp_path):
    from custmatch import cli, incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    top = max(cust.values(), key=int)
    gone = [k for k, c in cust.items() if c == top]
    pd.DataFrame([{"source": k.split(":")[0], "record_id": k.split(":")[1]} for k in gone]).to_csv(tmp / "erase.csv", index=False)
    assert cli.main(["forget", str(tmp / "erase.csv"), "--state", str(tmp / "run"), "--out", str(tmp / "f")]) == 0
    cfg = json.loads((tmp / "config.json").read_text())
    pd.DataFrame([{"id": "fresh1", "fname": "zed", "lname": "quux", "dob": "01/01/1901", "address": "1 nowhere rd",
                   "email": "zed@quux.example", "phone": "2069990000", "pid": "x"}]).to_csv(tmp / "fresh.csv", index=False)
    (tmp / "fresh.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="fresh.csv")]}))
    incremental.add(tmp / "f", tmp / "fresh.json", tmp / "f2")
    c = pd.read_csv(tmp / "f2" / "clusters.csv", dtype=str)
    assert c[c.record_id == "fresh1"].customer_id.iloc[0] != top


def test_summary_counts_customers_when_ids_are_carried(world, tmp_path):
    tmp, truth, s, _ = _ran(world, tmp_path)
    s2 = pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "r2", previous_dir=tmp / "run", log=lambda *_: None)
    c = pd.read_csv(tmp / "r2" / "clusters.csv", dtype=str)
    assert s2["customers"] == c.customer_id.nunique()


def test_a_must_link_survives_the_steps_that_split_customers(world, tmp_path):
    tmp, truth, _, cust = _ran(world, tmp_path)
    a = next(iter(cust))
    b = next(k for k in cust if cust[k] != cust[a])
    pd.DataFrame([{"a": a, "b": b, "decision": "same"}]).to_csv(tmp / "ov.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text()) | {"overrides": "ov.csv", "twin_guard": True,
                                                           "max_cluster": 1}
    (tmp / "m.json").write_text(json.dumps(cfg))
    pipeline.run(tmp / "m.json", tmp / "labels.csv", tmp / "m", log=lambda *_: None)
    c = pd.read_csv(tmp / "m" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert got[a] == got[b]


def test_add_applies_a_must_link_between_existing_records(world, tmp_path):
    from custmatch import incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    a = next(iter(cust))
    b = next(k for k in cust if cust[k] != cust[a])
    pd.DataFrame([{"a": a, "b": b, "decision": "same"}]).to_csv(tmp / "ov.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    crm.head(1).assign(id="later1").to_csv(tmp / "later.csv", index=False)
    (tmp / "later.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="later.csv")],
                                                "overrides": "ov.csv"}))
    incremental.add(tmp / "run", tmp / "later.json", tmp / "l")
    c = pd.read_csv(tmp / "l" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert got[a] == got[b]
    rec = pd.read_parquet(tmp / "l" / "state" / "records.parquet")
    assert not rec.updated_at.isna().any() and not (rec.updated_at == "nan").any()


def test_an_updated_records_old_links_are_not_saved(world, tmp_path):
    from custmatch import incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    links = pd.read_parquet(tmp / "run" / "state" / "links.parquet")
    a = links.a.iloc[0]
    src, rid = a.split(":")
    raw = pd.read_csv(tmp / f"{src}.csv", dtype=str, keep_default_na=False)
    row = raw[raw.iloc[:, 0] == rid].copy()
    for col in row.columns[1:]:
        if col != "pid" and row[col].iloc[0]:
            row[col] = "qq" + row[col]
    row.to_csv(tmp / "u.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    (tmp / "u.json").write_text(json.dumps({"sources": [dict(s_, path="u.csv") for s_ in cfg["sources"] if s_["name"] == src]}))
    incremental.add(tmp / "run", tmp / "u.json", tmp / "u")
    after = pd.read_parquet(tmp / "u" / "state" / "links.parquet")
    old = set(zip(links.a, links.b)) & set(zip(after.a, after.b))
    assert not [p for p in old if a in p], "a link from the record's old version was kept"


# ------------------------------------------------------------------------------ monitoring
def test_monitor_lines_runs_up_and_flags_drift(world, tmp_path):
    from custmatch import cli, monitor
    tmp, truth, _, _ = _ran(world, tmp_path)
    m = json.loads((tmp / "run" / "monitor.json").read_text())
    assert 0 < m["match_rate"] <= 100 and m["customers"] > 0 and sum(m["score_histogram"]) == m["pairs_scored"]
    # the same data again: no alerts
    pipeline.run(tmp / "config.json", None, tmp / "same", model_dir=tmp / "run", log=lambda *_: None)
    table, alerts = monitor.compare([tmp / "run", tmp / "same"])
    assert alerts == [] and len(table) == 2
    # a feed that stopped sending email: fill-rate and match-rate alerts, and exit 3 on request
    cfg = json.loads((tmp / "config.json").read_text())
    for s in cfg["sources"]:
        raw = pd.read_csv(tmp / s["path"], dtype=str, keep_default_na=False)
        raw[s["columns"]["email"]] = ""
        raw.to_csv(tmp / f"noemail_{s['path']}", index=False)
        s["path"] = f"noemail_{s['path']}"
    (tmp / "noemail.json").write_text(json.dumps(cfg))
    pipeline.run(tmp / "noemail.json", None, tmp / "broken", model_dir=tmp / "run", log=lambda *_: None)
    _, alerts = monitor.compare([tmp / "run", tmp / "broken"])
    assert any("email filled" in a for a in alerts), alerts
    assert cli.main(["monitor", str(tmp / "run"), str(tmp / "broken"), "--fail-on-drift",
                     "--out", str(tmp / "m.csv")]) == 3
    assert (tmp / "m.csv").exists()


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    from custmatch import monitor
    assert monitor.psi([10, 20, 70], [10, 20, 70]) == 0
    assert monitor.psi([80, 10, 10], [10, 10, 80]) > monitor.PSI_ALERT


def test_survivorship_is_checked_up_front_and_prefer_has_a_chosen_fallback():
    from custmatch import golden
    assert golden.check({"default": "most_common", "email": {"prefer": ["app"], "then": "most_recent"},
                         "address": "most_recent"}, ["email", "addr_street"]) == []
    assert golden.check("newest") and golden.check({"email": "newest"}, ["email"])
    assert golden.check({"emial": "most_recent"}, ["email"])                      # misspelt field
    assert golden.check({"email": {"prefer": "app"}}, ["email"])                  # not a list
    # no preferred source has a value: the `then` rule decides
    v = golden.survive(["old@x.com", "new@x.com"], ["web", "store"], {"prefer": ["app"], "then": "most_recent"},
                       dates=["2020-01-01", "2024-01-01"])
    assert v == "new@x.com"


def test_a_bad_survivorship_rule_is_a_config_error(world, tmp_path):
    tmp, _ = _world_copy(world, tmp_path, {"survivorship": {"email": "newest"}})
    with pytest.raises(io.ConfigError, match="unknown rule 'newest'"):
        io.load(tmp / "config.json")


def test_record_dates_add_a_years_apart_feature(world, tmp_path):
    from custmatch import schema
    from custmatch.core import block, features, profile_contacts
    assert ("updated_at", "years_apart") in schema.extra_fields({"record_dates": True})
    assert schema.extra_fields({}) == ()
    tmp, _ = _world_copy(world, tmp_path)
    df, cfg = io.load(tmp / "config.json")
    df["updated_at"] = ["2020-01-01" if i % 2 else "2023-01-01" for i in range(len(df))]
    df.loc[0, "updated_at"] = ""
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs, _ = block(df, prof)
    gap = features(df, pairs, prof, extra_fields=(("updated_at", "years_apart"),))[:, -1]
    has0 = (pairs == 0).any(axis=1)
    assert (gap[has0] == -1).all()                                           # missing date
    rest = gap[~has0]
    assert set(np.round(rest, 1)) <= {0.0, 3.0} and (rest >= 0).all()


def test_explain_gives_each_kind_of_reason(world, tmp_path):
    from custmatch import explain
    tmp, truth, _, cust = _ran(world, tmp_path)
    links = pd.read_csv(tmp / "run" / "links.csv", dtype=str)
    a, b = links.a.iloc[0], links.b.iloc[0]
    assert explain.why(tmp / "run", a, b)["reason"].startswith("matched directly")
    with pytest.raises(io.ConfigError, match="not in"):
        explain.why(tmp / "run", a, "crm:nope")
    # two records with nothing in common were never compared
    keys = sorted(cust)
    far = next(k for k in keys[::-1] if cust[k] != cust[a] and not explain.why(tmp / "run", a, k)["compared_by_blocking"])
    assert explain.why(tmp / "run", a, far)["reason"].startswith("never compared")
    # a steward's "different" on a direct link: reported as the override
    pd.DataFrame([{"a": a, "b": b, "decision": "different"}]).to_csv(tmp / "ov.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text()) | {"overrides": "ov.csv"}
    (tmp / "ov.json").write_text(json.dumps(cfg))
    pipeline.run(tmp / "ov.json", tmp / "labels.csv", tmp / "ov", log=lambda *_: None)
    w = explain.why(tmp / "ov", a, b)
    assert "different (override)" in w["reason"] and not w["same_customer"]


def test_add_refuses_to_change_what_the_model_was_trained_with(world, tmp_path):
    from custmatch import incremental
    tmp, *_ = _ran(world, tmp_path)
    cfg = json.loads((tmp / "config.json").read_text())
    (tmp / "c.json").write_text(json.dumps({"sources": cfg["sources"][:1], "country": "AU"}))
    with pytest.raises(io.ConfigError, match="fixed by the trained model"):
        incremental.add(tmp / "run", tmp / "c.json", tmp / "x")


# ------------------------------------------------------------------------------ cookies
def _two_customers(tmp, cust):
    """Two crm records in different customers, with an email and a phone each."""
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    ok = crm[(crm.email != "") & (crm.phone != "")]
    a = ok.iloc[0]
    b = next(r for _, r in ok.iterrows() if cust["crm:" + r["id"]] != cust["crm:" + a["id"]])
    return a, b


def test_a_cookie_seen_with_two_identities_joins_them_but_a_shared_device_does_not(world, tmp_path):
    tmp, truth, _, cust = _ran(world, tmp_path)
    a, b = _two_customers(tmp, cust)
    other = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False).iloc[-1]
    pd.DataFrame([
        {"anon": "c1", "mail": a["email"].upper(), "tel": ""},             # logged in as a ...
        {"anon": "c1", "mail": "", "tel": b["phone"]},                     # ... and gave b's phone
        {"anon": "c1", "mail": "", "tel": ""},                             # anonymous page views
        {"anon": "c2", "mail": a["email"], "tel": ""},                     # a family laptop:
        {"anon": "c2", "mail": other["email"], "tel": ""},                 # two people logged in
        {"anon": "c3", "mail": "nobody@example.org", "tel": ""}]).to_csv(tmp / "events.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text()) | {"identity_events": [
        {"path": "events.csv", "cookie": "anon", "identifiers": {"email": "mail", "phone": "tel"}}],
        "cookie_name_check": False}            # a and b are two people here: test the mechanics
    (tmp / "ck.json").write_text(json.dumps(cfg))
    s = pipeline.run(tmp / "ck.json", tmp / "labels.csv", tmp / "ck", log=lambda *_: None)
    assert s["cookies"]["stitched"] == 1 and s["cookies"]["shared_device"] == 1 and s["cookies"]["no_known_identity"] == 1
    c = pd.read_csv(tmp / "ck" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert got["crm:" + a["id"]] == got["crm:" + b["id"]]
    assert got["crm:" + a["id"]] != got["crm:" + other["id"]] or cust["crm:" + a["id"]] == cust["crm:" + other["id"]]
    ck = pd.read_csv(tmp / "ck" / "cookies.csv", dtype=str, keep_default_na=False).set_index("cookie")
    assert ck.loc["c1", "customer_id"] == got["crm:" + a["id"]] and ck.loc["c2", "status"] == "shared_device"
    # a steward's "different" still wins over the cookie
    pd.DataFrame([{"a": "crm:" + a["id"], "b": "crm:" + b["id"], "decision": "different"}]).to_csv(tmp / "ov.csv", index=False)
    (tmp / "ck2.json").write_text(json.dumps(cfg | {"overrides": "ov.csv", "minimal_outputs": True}))
    pipeline.run(tmp / "ck2.json", tmp / "labels.csv", tmp / "ck2", log=lambda *_: None)
    c = pd.read_csv(tmp / "ck2" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert got["crm:" + a["id"]] != got["crm:" + b["id"]]
    assert (pd.read_csv(tmp / "ck2" / "cookies.csv", dtype=str, keep_default_na=False).identifiers == "").all()


def test_add_stitches_cookies_from_new_events(world, tmp_path):
    from custmatch import incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    a, b = _two_customers(tmp, cust)
    pd.DataFrame([{"anon": "d1", "mail": a["email"], "tel": ""}, {"anon": "d1", "mail": "", "tel": b["phone"]}]
                 ).to_csv(tmp / "ev2.csv", index=False)
    cfg = json.loads((tmp / "config.json").read_text())
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    crm.head(1).assign(id="cknew").to_csv(tmp / "n.csv", index=False)
    (tmp / "n.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="n.csv")], "identity_events": [
        {"path": "ev2.csv", "cookie": "anon", "identifiers": {"email": "mail", "phone": "tel"}}],
        "cookie_name_check": False}))
    s = incremental.add(tmp / "run", tmp / "n.json", tmp / "a")
    assert s["cookies"]["stitched"] == 1
    c = pd.read_csv(tmp / "a" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert got["crm:" + a["id"]] == got["crm:" + b["id"]]
    assert (tmp / "a" / "cookies.csv").exists()


def test_cookie_config_errors(world, tmp_path):
    from custmatch import identity
    tmp, _ = _world_copy(world, tmp_path)
    df, cfg = io.load(tmp / "config.json")
    pd.DataFrame([{"anon": "c", "x": "1"}]).to_csv(tmp / "e.csv", index=False)
    with pytest.raises(io.ConfigError, match="not in"):
        identity.stitch(df, {"identity_events": [{"path": "e.csv", "cookie": "anon", "identifiers": {"email": "mail"}}]}, tmp)
    with pytest.raises(io.ConfigError, match="no field"):
        identity.stitch(df, {"identity_events": [{"path": "e.csv", "cookie": "anon", "identifiers": {"loyalty": "x"}}]}, tmp)
    with pytest.raises(io.ConfigError, match="'cookie'"):
        identity.load_events({"identity_events": [{"path": "e.csv"}]}, tmp)
    assert len(identity.stitch(df, {}, tmp)[0]) == 0


def test_call_centre_cookies_are_excluded_or_caught_by_the_name_check(world, tmp_path):
    tmp, truth, _, cust = _ran(world, tmp_path)
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    ok = crm[(crm.email != "") & (crm.phone != "")]
    a = ok.iloc[0]
    b = next(r for _, r in ok.iterrows() if cust["crm:" + r["id"]] != cust["crm:" + a["id"]]
             and r["fname"][0] != a["fname"][0])
    # an agent: customer a's email, then customer b's phone - one of each, so the per-field
    # limit does not see a shared device; the names do
    pd.DataFrame([{"anon": "agent7", "mail": a["email"], "tel": "", "channel": "web"},
                  {"anon": "agent7", "mail": "", "tel": b["phone"], "channel": "web"},
                  {"anon": "cc1", "mail": a["email"], "tel": "", "channel": "Call_Centre"},
                  {"anon": "cc1", "mail": "", "tel": b["phone"], "channel": "call_centre"}]).to_csv(tmp / "ev.csv", index=False)
    pd.DataFrame({"cookie": ["agent7"]}).to_csv(tmp / "staff.csv", index=False)
    spec = {"path": "ev.csv", "cookie": "anon", "identifiers": {"email": "mail", "phone": "tel"},
            "exclude": {"column": "channel", "values": ["call_centre"]}}
    cfg = json.loads((tmp / "config.json").read_text()) | {"identity_events": [spec]}
    (tmp / "c.json").write_text(json.dumps(cfg))
    s = pipeline.run(tmp / "c.json", tmp / "labels.csv", tmp / "c", log=lambda *_: None)
    assert s["cookies"] == {"cookies": 1, "conflicting_names": 1, "links": 0}         # cc1 excluded, agent7 caught
    (tmp / "c2.json").write_text(json.dumps(cfg | {"identity_events": [spec | {"exclude_cookies": "staff.csv"}]}))
    assert "cookies" not in pipeline.run(tmp / "c2.json", tmp / "labels.csv", tmp / "c2", log=lambda *_: None)
    (tmp / "c3.json").write_text(json.dumps(cfg | {"cookie_name_check": False}))
    s = pipeline.run(tmp / "c3.json", tmp / "labels.csv", tmp / "c3", log=lambda *_: None)
    assert s["cookies"]["stitched"] == 1                                              # what the check prevents


def test_extra_blocking_keys_can_use_transforms(world, tmp_path):
    from custmatch.core import block, profile_contacts
    tmp, _ = _world_copy(world, tmp_path, {"extra_blocking": [["date_of_birth:year", "last_name:first1"]]})
    df, cfg = io.load(tmp / "config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    base, _ = block(df, prof)
    more, _ = block(df, prof, extra_keys=[("date_of_birth:year", "last_name:first1")])
    assert len(more) >= len(base)
    for bad, msg in (([["nope:first1"]], "unknown field"), ([["last_name:reverse"]], "unknown transform")):
        d = tmp_path / bad[0][0].replace(":", "_")
        d.mkdir()
        tmp2, _ = _world_copy(world, d, {"extra_blocking": bad})
        with pytest.raises(io.ConfigError, match=msg):
            io.load(tmp2 / "config.json")


def test_hashed_first_names_leave_no_plain_text_in_derived_columns(world, tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTMATCH_SALT", "k")
    tmp, _ = _world_copy(world, tmp_path, {"hash_fields": ["first_name", "last_name", "email"]})
    df, _ = io.load(tmp / "config.json")
    raw = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    names = {n.lower() for n in raw.fname if len(n) > 3}
    cols = [c for c in df.columns if not c.startswith("addr_")]          # streets are not hashed here
    text = " ".join(df[cols].astype(str).to_numpy().ravel())
    assert not [n for n in names if n in text.split() or f"|{n}" in text or f"{n}|" in text]
    # each root is a hash, so roots still meet when two names share one
    assert "first_roots" not in df.columns or all(len(t) == 32 for v in df.first_roots if v for t in v.split("|"))


def test_unique_sources_keep_one_record_per_source_in_run_and_add(world, tmp_path):
    from custmatch import incremental
    tmp, truth, _, cust = _ran(world, tmp_path, {"unique_sources": ["crm"]})
    c = pd.read_csv(tmp / "run" / "clusters.csv", dtype=str)
    assert not c[c.source == "crm"].duplicated("customer_id").any()
    # a new crm record copying an existing one must not join that customer through add
    cfg = json.loads((tmp / "config.json").read_text())
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    crm.head(1).assign(id="copy1").to_csv(tmp / "cp.csv", index=False)
    (tmp / "cp.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="cp.csv")]}))
    incremental.add(tmp / "run", tmp / "cp.json", tmp / "a")
    c = pd.read_csv(tmp / "a" / "clusters.csv", dtype=str)
    assert not c[c.source == "crm"].duplicated("customer_id").any()


# ------------------------------------------------------------------ safe for marketing
def test_min_precision_picks_a_stricter_threshold_and_says_when_it_cannot(world, tmp_path):
    tmp, truth, base, _ = _ran(world, tmp_path)
    cfg = json.loads((tmp / "config.json").read_text())
    (tmp / "p.json").write_text(json.dumps(cfg | {"min_precision": 0.999}))
    s = pipeline.run(tmp / "p.json", tmp / "labels.csv", tmp / "p", log=lambda *_: None)
    assert s["threshold"] >= base["threshold"]
    assert s["cross_fitted_closure_aware"]["min_precision"] == 0.999
    if not s["cross_fitted_closure_aware"]["min_precision_met"]:
        assert any("min_precision" in w for w in s["warnings"])


def test_contact_safety_takes_contacts_only_through_certain_links(world, tmp_path):
    from custmatch import incremental
    from custmatch.golden import safe_contacts
    # three records of one customer; 0-1 certain, 1-2 weak: record 2's email never reaches 0 or 1
    df = pd.DataFrame({"source": ["a", "a", "a"], "email": ["x@a.com", "", "y@b.com"], "phone": ["", "1", ""]})
    per_customer, per_record = safe_contacts(df, np.array([0, 0, 0]), np.array([[0, 1]]), "most_complete")
    assert per_record["email"].tolist() == ["x@a.com", "x@a.com", "y@b.com"]
    assert per_customer[0]["email"] == "x@a.com"                              # the larger certain group
    # a household's shared email is used only when the group has no personal one
    df2 = pd.DataFrame({"source": ["a"] * 3, "email": ["fam@x.com", "me@x.com", "fam@x.com"], "phone": ["", "", ""]})
    _, pr = safe_contacts(df2, np.array([0, 0, 1]), np.array([[0, 1]]), "most_complete",
                          classes={"email": {"fam@x.com": ("household", 1.0)}})
    assert pr["email"].tolist() == ["me@x.com", "me@x.com", "fam@x.com"] and pr["email_shared"].tolist() == [False, False, True]
    tmp, truth, _, _ = _ran(world, tmp_path, {"contact_confidence": 0.95})
    c = pd.read_csv(tmp / "run" / "contacts.csv", dtype=str, keep_default_na=False)
    assert len(c) == len(pd.read_csv(tmp / "run" / "clusters.csv")) and {"email", "phone"} <= set(c.columns)
    cfg = json.loads((tmp / "config.json").read_text())
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    crm.head(1).assign(id="ck9").to_csv(tmp / "n.csv", index=False)
    (tmp / "n.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="n.csv")]}))
    incremental.add(tmp / "run", tmp / "n.json", tmp / "a")
    assert (tmp / "a" / "contacts.csv").exists()


def test_random_only_labels_get_a_warning(world, tmp_path):
    tmp, truth, s, _ = _ran(world, tmp_path)
    assert any("drawn at random" in w for w in s["warnings"])


# ------------------------------------------------------------------ fixes from the adversarial review
def test_an_add_configs_override_holds_for_later_adds(world, tmp_path):
    from custmatch import incremental
    tmp, truth, _, cust = _ran(world, tmp_path)
    cfg = json.loads((tmp / "config.json").read_text())
    crm = pd.read_csv(tmp / "crm.csv", dtype=str, keep_default_na=False)
    x = crm.iloc[0]
    crm.head(1).assign(id="d1").to_csv(tmp / "b1.csv", index=False)
    crm.head(1).assign(id="d2").to_csv(tmp / "b2.csv", index=False)
    pd.DataFrame([{"a": f"crm:{x['id']}", "b": "crm:d1", "decision": "different"}]).to_csv(tmp / "ov.csv", index=False)
    (tmp / "b1.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="b1.csv")], "overrides": "ov.csv"}))
    (tmp / "b2.json").write_text(json.dumps({"sources": [dict(cfg["sources"][0], path="b2.csv")]}))
    incremental.add(tmp / "run", tmp / "b1.json", tmp / "a1")
    incremental.add(tmp / "a1", tmp / "b2.json", tmp / "a2")
    c = pd.read_csv(tmp / "a2" / "clusters.csv", dtype=str)
    got = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    assert got[f"crm:{x['id']}"] != got["crm:d1"], "the first add's decision was forgotten"


def test_blocking_report_and_skeleton_hold_no_plain_text_when_asked(world, tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTMATCH_SALT", "k")
    tmp, truth, _, _ = _ran(world, tmp_path, {"blocking": "fixed", "minimal_outputs": True})
    rep = (tmp / "run" / "blocking_report.json").read_text()
    assert '"key": "(redacted)"' in rep and "|" not in rep.split('"largest_blocks"')[1][:200]
    df = pd.DataFrame({"source": ["a"], "record_id": ["1"], "first_name": ["x"], "last_name": ["y"],
                       "name_skeleton": ["mhmd|l"], "first_roots": ["x"]})
    out = io.hash_fields(df, {"hash_fields": ["first_name", "last_name"]})
    assert out.name_skeleton[0] != "mhmd|l" and len(out.name_skeleton[0]) == 32


def test_hashed_names_switch_the_twin_guard_off():
    from custmatch import stewardship
    df = pd.DataFrame({"first_name": ["a" * 32, "b" * 32], "last_name": ["c" * 32] * 2, "date_of_birth": ["1980-01-01"] * 2})
    assert not stewardship.twin_pairs(df, np.array([[0, 1]]), hashed=True).any()


def test_lsh_blocking_with_fixed_blocking_is_a_config_error(world, tmp_path):
    from custmatch import blocking
    from custmatch.core import profile_contacts
    tmp, _ = _world_copy(world, tmp_path)
    df, cfg = io.load(tmp / "config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    with pytest.raises(io.ConfigError, match="lsh_blocking"):
        blocking.candidates_for(df, prof, cfg | {"blocking": "fixed", "lsh_blocking": {"fields": ["first_name"]}})


def test_contact_safety_never_uses_junk_and_survivorship_keeps_addresses_whole():
    from custmatch.golden import build_golden, safe_contacts
    df = pd.DataFrame({"source": ["a"], "email": ["test@test.com"], "phone": [""]})
    _, pr = safe_contacts(df, np.array([0]), np.zeros((0, 2)), "most_complete",
                          classes={"email": {"test@test.com": ("junk", 0.0)}})
    assert pr["email"].tolist() == [""]
    d2 = pd.DataFrame({"source": ["a"] * 4, "addr_number": ["1", "1", "2", "3"], "addr_unit": [""] * 4,
                       "addr_street": ["main st", "elm st", "oak ave", "oak ave"]})
    g = build_golden(d2, np.zeros(4, int), "most_complete", ["addr_number", "addr_street", "addr_unit"])[0]
    assert (g["addr_number"], g["addr_street"]) in {("1", "main st"), ("1", "elm st"), ("2", "oak ave"), ("3", "oak ave")}


def test_event_phones_meet_records_of_any_configured_country(tmp_path):
    from custmatch import identity
    df = pd.DataFrame({"first_name": ["kiri"], "phone": ["64211234567"], "email": [""]})
    pd.DataFrame({"anon": ["c1", "c1"], "tel": ["021 123 4567", ""], "mail": ["", "k@x.co.nz"]}).to_csv(tmp_path / "e.csv", index=False)
    df2 = pd.concat([df, pd.DataFrame({"first_name": ["kiri"], "phone": [""], "email": ["k@x.co.nz"]})], ignore_index=True)
    cfg = {"country": ["AU", "NZ"], "identity_events": [{"path": "e.csv", "cookie": "anon",
                                                         "identifiers": {"phone": "tel", "email": "mail"}}]}
    pairs, table, counts = identity.stitch(df2, cfg, tmp_path)
    assert counts.get("stitched") == 1 and pairs.tolist() == [[0, 1]]


def test_unknown_config_keys_are_reported(world, tmp_path):
    tmp, truth, s, _ = _ran(world, tmp_path, {"twin_gaurd": True})
    assert any('did you mean "twin_guard"' in w for w in s["warnings"])


def test_a_removed_matcher_model_is_a_clear_config_error(world, tmp_path):
    tmp, truth = _world_copy(world, tmp_path, {"matcher_model": "tabpfn"})
    _labels_from_truth(tmp / "config.json", truth, tmp / "labels.csv")
    with pytest.raises(io.ConfigError, match="removed in 0.1.0"):
        pipeline.run(tmp / "config.json", tmp / "labels.csv", tmp / "out", log=lambda *_: None)
    assert any("auto_models" in w for w in io.unknown_keys({"auto_models": ["forest"]}))
