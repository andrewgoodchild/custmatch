"""Guard against the write-up drifting from the results files.

Every headline number in the docs (README.md and docs/) is typed by hand. If a rerun changes a
result, these fail until the prose is updated - which is the failure mode that let a claim
about a section that did not exist survive in this repo.
"""
import json
from pathlib import Path
from statistics import fmean

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research/results"
DOCS = ROOT / "docs"
# the report: an introduction and one file per chapter, read together in order
REPORT = "\n".join(p.read_text() for p in sorted((DOCS / "report").glob("*.md"), key=lambda p: (p.name != "README.md", p.name)))
# the research summary: once README.md, now the short README, the report's opening and reproducing.md
README = "\n".join([(ROOT / "README.md").read_text(), REPORT, (DOCS / "reproducing.md").read_text()])
# the guide and the report: the measurements behind the guide's options moved into the report
GUIDE = "\n".join([(DOCS / "guide.md").read_text(), REPORT])


def load(name):
    """A committed results file. Missing is a failure, not a skip: a renamed file would otherwise
    silently switch off every claim test that reads it."""
    path = RESULTS / name
    if not path.exists():
        pytest.fail(f"research/results/{name} is missing - regenerate it or update the test and docs")
    return json.load(open(path))


def quoted(value, decimals=2):
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".") if decimals else str(value)


def test_report_sections_are_numbered_without_gaps():
    import re
    nums = [int(m) for m in re.findall(r"^# (\d+)\.", REPORT, re.M)]
    assert nums == list(range(1, len(nums) + 1)), f"section numbering broken: {nums}"


def test_every_report_section_reference_exists():
    """Catches 'written up as section N' when section N was never written."""
    import re
    referenced = {int(m) for m in re.findall(r"(?:[Ss]ection|[Cc]hapter) (\d+)", REPORT)}
    present = {int(m) for m in re.findall(r"^# (\d+)\.", REPORT, re.M)}
    assert referenced <= present, f"docs/report/ refers to missing chapters {referenced - present}"


def test_no_unfilled_placeholders():
    """An HTML comment marker is invisible when rendered, so a section left as
    '<!--WARMUP-->' reads as finished while the claims pointing at it have no support."""
    import re
    for name, text in (("docs/report/", REPORT), ("README.md + docs", README), ("docs/guide.md", GUIDE)):
        left = re.findall(r"<!--\s*[A-Z_]+\s*-->|\bTODO\b|\bTBD\b", text)
        assert not left, f"{name} still has placeholders: {left}"


def test_magellan_numbers_match_results():
    rows = load("magellan_baseline.json")
    for r in rows:
        assert r["test_f1"] >= r["reported_magellan"] - 0.5, \
            f"{r['task']} no longer matches or beats the published Magellan number"
    abt = next(r for r in rows if r["task"] == "Textual/Abt-Buy")
    assert f"{abt['test_f1']:.1f}" in REPORT


def test_person_head_to_head_numbers_are_in_the_report():
    d = load("person_head_to_head.json")
    for model in ("random_forest", "ditto_roberta"):
        best_pair = max(s["pairwise"]["f1"] for s in d[model]["sweep"])
        best_clu = max(s["clustered"]["f1"] for s in d[model]["sweep"])
        assert f"{best_pair:.1f}" in REPORT or f"{best_pair:.2f}" in REPORT
        assert f"{best_clu:.1f}" in REPORT or f"{best_clu:.2f}" in REPORT


def test_clustering_collapse_claim_still_holds():
    """The report claims the pairwise-optimal threshold produces a giant cluster."""
    d = load("person_head_to_head.json")
    rf = d["random_forest"]
    best_pair = max(rf["sweep"], key=lambda s: s["pairwise"]["f1"])
    assert best_pair["clustered"]["f1"] < 5, "collapse claim no longer true"
    assert best_pair["clustered"]["largest_cluster"] > 1000
    assert str(best_pair["clustered"]["largest_cluster"]) in REPORT.replace(",", "")


def test_customer_pipeline_claims():
    d = load("customer_pipeline.json")
    aware, naive = d["pathology_aware"], d["naive"]
    assert aware["candidate_pairs"] * 5 < naive["candidate_pairs"], \
        "the 22x blocking reduction claim no longer holds"
    for field in ("email_junk", "phone_junk"):
        det = aware["contact_detection"][field]
        assert det["precision"] == 100.0 and det["recall"] == 100.0, \
            "junk detection is no longer exact - update the report"
    assert aware["households"]["f1"] > 85, "household detection regressed"
    assert f"{aware['households']['f1']:.1f} F1" in README, "README household F1 is stale"


def test_readme_headline_table_matches_results():
    d = load("person_head_to_head.json")
    rf_clu = max(s["clustered"]["f1"] for s in d["random_forest"]["sweep"])
    dt_clu = max(s["clustered"]["f1"] for s in d["ditto_roberta"]["sweep"])
    assert f"{rf_clu:.1f}" in README
    assert f"{dt_clu:.1f}" in README


def test_best_threshold_is_inside_the_sweep():
    """A best threshold on the edge of the grid means the grid was too short, not that the
    optimum was found - a sweep capped at 0.95 once reported Ditto's clustered F1 as 88.1
    when the real optimum, at 0.998, is 94.7."""
    d = load("person_head_to_head.json")
    for model in ("random_forest", "ditto_roberta"):
        ths = [s["threshold"] for s in d[model]["sweep"]]
        best = d[model]["best_clustered"]["threshold"]
        assert min(ths) < best < max(ths), f"{model}: best clustered threshold {best} is on the edge"


def test_bridge_claim_matches_fair_comparison():
    """Bridges must be compared with plain thresholding tuned on the same grid; the report
    says that fair comparison makes the bridge rule a small gain, not a large one."""
    c = load("cascade.json")
    by = {r["config"]: r for r in c["bridge_aware"]}
    for name in ("random_forest_only", "ditto_only"):
        gain = by[name]["bridge"]["f1"] - by[name]["plain_f1"]
        assert 0 <= gain < 1.5, f"{name}: bridge gain is now {gain:.2f} - update section 10"
        assert f"{by[name]['bridge']['f1']:.2f}" in REPORT
    d = load("person_head_to_head.json")
    assert by["ditto_only"]["plain_f1"] == d["ditto_roberta"]["best_clustered"]["clustered"]["f1"]


def test_cascade_claims():
    """The forest-as-veto cascade must stay level with Ditto out of sample and beat it on work;
    letting the forest accept must stay worse. Those are the report's two recommendations."""
    c = load("cascade.json")
    for r in c["split_half"]:
        if r["config"] == "forest_rejects_only":
            assert r["f1"] > r["ditto_only_f1"] - 0.5, r
            assert r["routed_frac"] < 0.9, r
        else:
            assert r["f1"] < r["ditto_only_f1"], r
    veto = next(r for r in c["frontier_picked_on_test"] if r["hi"] == 1.0)
    assert f"{veto['f1']:.2f}" in REPORT and f"{100 * (1 - veto['routed_frac']):.0f}%" in README


def test_golden_record_claims():
    """Survivorship must stay below the ceiling, and the ceiling must stay below 100 -
    if either breaks, the 'source data is the bottleneck' conclusion needs revisiting."""
    g = load("golden_record.json")
    ceiling = g["value_available_ceiling"]["oracle"]
    for field in ("first_name", "last_name"):
        best = max(g["oracle"][s]["per_field"][field] for s in g["oracle"])
        raw = g["baseline_raw_records"][field]
        assert raw < best <= ceiling[field], (
            f"{field}: raw {raw}, best rule {best}, ceiling {ceiling[field]}")
        assert ceiling[field] < 100, "some errors must be unrecoverable"
    # perfect linkage should help, but only a little
    gain = (g["oracle"]["first_seen"]["all_fields_correct"]
            - g["predicted"]["first_seen"]["all_fields_correct"])
    assert 0 < gain < 3, f"linkage contribution moved to {gain:.2f} points"
    assert f"{g['baseline_raw_records']['first_name']:.2f}" in REPORT


PUBLISHED_GPT = {  # domain-complex-force F1, authors' results.xlsx
    "gpt-4o-mini": {"abt-buy-sampled-gs": 90.95, "amazon-google-sampled-gs": 70.98,
                    "dblp-scholar-sampled-gs": 86.11, "walmart-amazon-sampled-gs": 86.36,
                    "wdcproducts-80cc-seen-sampled-250-gs-2": 80.84},
    "gpt-4o-2024-08-06": {"abt-buy-sampled-gs": 90.47, "amazon-google-sampled-gs": 73.56,
                          "dblp-scholar-sampled-gs": 89.76, "walmart-amazon-sampled-gs": 84.04,
                          "wdcproducts-80cc-seen-sampled-250-gs-2": 87.64},
}


def test_gpt_reruns_reproduce_and_are_reported():
    for model, published in PUBLISHED_GPT.items():
        for ds, f1 in published.items():
            r = load(f"llm/result_{ds}_domain-complex-force_{model}.json")
            assert r["n"] > 1000, f"{model} {ds}: partial run ({r['n']} pairs)"
            assert abs(r["f1"] - f1) <= 1.5, f"{model} {ds}: {r['f1']} vs published {f1}"
            assert f"{r['f1']:.2f}" in REPORT


def test_zero_shot_llms_lose_to_the_forest_on_person_pairs():
    """The customer-matching conclusion of section 6: even GPT-4o is below the forest."""
    for model in PUBLISHED_GPT:
        r = load(f"llm/result_historical50k-person-hard_domain-complex-force_{model}.json")
        assert r["n"] == 660 and r["f1"] < 76.5, f"{model} now scores {r['f1']} on person pairs"
        assert f"{r['f1']:.1f}" in REPORT and f"{r['f1']:.1f}" in README


PUBLISHED_GPT4 = {"abt-buy-sampled-gs": 95.15, "amazon-google-sampled-gs": 75.61,
                  "dblp-scholar-sampled-gs": 88.44, "walmart-amazon-sampled-gs": 89.00,
                  "wdcproducts-80cc-seen-sampled-250-gs-2": 88.35}


def test_luna_matches_or_beats_published_gpt4():
    for ds, f1 in PUBLISHED_GPT4.items():
        r = load(f"llm/result_{ds}_domain-complex-force_gpt-6-luna.json")
        assert r["f1"] >= f1 - 0.5, f"Luna {ds}: {r['f1']} vs GPT-4 {f1}"
        assert f"{r['f1']:.2f}" in REPORT


def test_person_task_conclusions():
    """Section 6: on historical_50k only Luna beats the forest; on synthetic customers the best
    LLM and the forest are level. Both halves of the revised conclusion rest on these."""
    hist = {m: load(f"llm/result_historical50k-person-hard_domain-complex-force_{m}.json")["f1"]
            for m in ("gpt-6-luna", "jev-router", "gpt-4o-2024-08-06")}
    assert hist["gpt-6-luna"] > 76.5 > hist["gpt-4o-2024-08-06"]
    rf = load("customer_person_task_baselines.json")["random_forest"]["f1"]
    cust = {m: load(f"llm/result_customer-person-hard_domain-complex-force_{m}.json")
            for m in ("gpt-6-luna", "jev-router", "gpt-4o-2024-08-06", "gpt-4o-mini")}
    assert abs(cust["gpt-6-luna"]["f1"] - rf) < 1.5, "Luna and the forest are no longer level"
    assert f"{rf:.1f}" in REPORT and f"{rf:.1f}" in README
    for m, r in cust.items():
        assert r["n"] == 660 and r["empty_answers"] == 0, m
        assert f"{r['f1']:.1f}" in REPORT, m


def test_router_is_not_better_than_luna():
    """README: Jev is within ±2 F1 of Luna on every set and costs more on every set."""
    sets = list(PUBLISHED_GPT4) + ["historical50k-person-hard", "customer-person-hard"]
    for ds in sets:
        j = load(f"llm/result_{ds}_domain-complex-force_jev-router.json")
        l = load(f"llm/result_{ds}_domain-complex-force_gpt-6-luna.json")
        assert abs(j["f1"] - l["f1"]) <= 2, f"{ds}: Jev {j['f1']} vs Luna {l['f1']}"
        assert j["cost_usd"] > 2 * l["cost_usd"], ds


def test_jev_decision_model_claims():
    """Jev direct is cheap for all seven sets; its tuned cut-off beats the forest on the
    historical person pairs; criteria plus structured fields lift it further."""
    sets = list(PUBLISHED_GPT4) + ["historical50k-person-hard", "customer-person-hard"]
    res = {ds: load(f"llm/result_{ds}_domain-complex-force_jev-1.13.json") for ds in sets}
    assert sum(r["cost_usd"] for r in res.values()) < 0.15
    for ds, r in res.items():
        assert f"{r['f1_tuned_held_out']:.1f}" in REPORT, ds
    hist = res["historical50k-person-hard"]["f1_tuned_held_out"]
    assert hist > 76.5  # README leads with the criteria+structured figure instead


def test_jev_variant_claims():
    v = load("jev_variants.json")
    h, c = v["historical50k-person-hard"], v["customer-person-hard"]
    assert h["criteria"]["match"]["f1_at_0.5"] - h["baseline"]["match"]["f1_at_0.5"] > 3
    best = h["structured_criteria"]["match"]["f1_tuned_held_out"]
    luna = load("llm/result_historical50k-person-hard_domain-complex-force_gpt-6-luna.json")["f1"]
    assert best > luna, "criteria + structured Jev is no longer the best on historical_50k"
    assert f"{best:.1f}" in REPORT and f"{best:.1f}" in README
    for task in (h, c):
        assert task["structured_criteria"]["match"]["auc"] > task["baseline"]["match"]["auc"]
    assert c["multi"]["match"]["f1_at_0.5"] >= c["structured_criteria"]["match"]["f1_at_0.5"] - 1, \
        "extra questions now hurt the match answer"
    for variant in v.values():
        for name, r in variant.items():
            if isinstance(r, dict) and "match" in r:
                assert f"{r['match']['f1_at_0.5']:.2f}" in REPORT, name


def test_jev_cascade_claims():
    """Section 7: Jev in the band loses to Ditto and even to the forest alone, held out."""
    c = load("jev_cascade.json")
    h = {k: v["mean"] for k, v in c["split_half"].items()}
    assert h["forest_veto_then_ditto"] > h["forest_veto_then_mean_jev_ditto"] > h["forest_veto_then_jev"]
    assert h["forest_veto_then_jev"] < h["forest_only"], "Jev in the band now beats the forest"
    d = load("person_head_to_head.json")
    assert c["picked_on_test"]["ditto_only"]["f1"] == d["ditto_roberta"]["best_clustered"]["clustered"]["f1"], \
        "the cascade evaluation no longer reproduces section 7's Ditto number"
    for k, v in h.items():
        assert f"{v:.2f}" in REPORT, k
    assert f"{c['band_auc']['jev']:.3f}" in REPORT and f"{c['band_auc']['jev']:.3f}" in README


def test_ditto_on_customer_claims():
    """Section 9/11: Ditto beats the forest on customer records by under a point, held out,
    and recovers part - not all - of what perfect linkage adds to golden records."""
    c = load("customer_ditto.json")
    pipe = load("customer_pipeline.json")["pathology_aware"]["clustered"]["f1"]
    assert c["picked_on_test"]["forest_only"]["f1"] == pipe, "no longer the same pairs as the pipeline"
    h = {k: v["mean"] for k, v in c["split_half"].items()}
    gain = h["forest_veto_then_ditto"] - h["forest_only"]
    assert 0 < gain < 1.5, f"Ditto's held-out gain on customers is now {gain:.2f}"
    for k, v in h.items():
        assert f"{v:.2f}" in REPORT, k
    g = {k: v["first_seen"]["all_fields_correct"] for k, v in c["golden_records_on_test"].items()}
    assert g["forest_only"] < g["forest_veto_then_ditto"] < g["perfect_linkage"]
    for v in g.values():
        assert f"{v:.2f}" in REPORT


def test_llm_labelling_claims():
    """Section 7: Jev labels cost the forest under a point on customers, over 5 on historical_50k."""
    L = load("jev_labels.json")
    for name, (lo, hi) in (("customer", (0, 1.0)), ("historical", (5, 100))):
        m = {k: v["mean"] for k, v in L[name]["split_half"].items()}
        loss = m["forest_true_labels"] - m["forest_jev_hard_labels"]
        assert lo < loss < hi, f"{name}: labelling loss is now {loss:.2f}"
        for v in m.values():
            assert f"{v:.2f}" in REPORT, (name, v)
    assert L["customer"]["labels"]["accuracy"] > 98 > L["historical"]["labels"]["accuracy"]


def test_tool_validation_claims():
    """docs/guide.md: the tool's validation figures, and the label-count advice they support."""
    v = load("custmatch_validation.json")
    f = {k: r["all_records"]["f1"] for k, r in v.items() if isinstance(r, dict)}
    flat = " ".join(GUIDE.replace("**", "").split())
    assert f"{f['hand_1000']:.1f} clustered F1 from 1,000 labelled pairs" in _flat(README)
    assert f"{f['hand_500']:.2f} clustered F1 and 1,000 gave {f['hand_1000']:.2f}" in flat
    assert abs(f["hand_1000"] - f["hand_500"]) < 0.5, "the 500-label advice no longer holds"
    j = v["jev_2000_gated_on_300"]
    assert j["gate"]["accuracy"] >= 98 and f"{j['all_records']['f1']:.2f} clustered" in GUIDE
    assert f"{j['unlabelled_people_only']['f1']:.2f}" in GUIDE
    assert f"{v['hand_1000']['households']['f1']:.1f} F1" in GUIDE


def test_gap_claims():
    """docs/guide.md: active learning, the Splink benchmark, incremental matching and scale."""
    flat = " ".join(GUIDE.replace("**", "").split())
    v = load("custmatch_validation.json")
    a, r = v["active_1000"]["all_records"]["f1"], v["hand_1000"]["all_records"]["f1"]
    assert a > r and f"{a:.2f} clustered F1 against {r:.2f}" in flat
    for row in v["active_vs_random_1000_seeds"]:
        assert row["active_1000"] > row["random_1000"], row
        assert f"{row['active_1000']:.2f}" in flat and f"{row['random_1000']:.2f}" in flat
    sp = load("splink_vs_custmatch.json")
    best_splink = max(sp["splink_unsupervised"]["ceiling_threshold_picked_on_truth"]["f1"],
                      sp["splink_m_from_500_labels"]["threshold_from_labels"]["f1"])
    assert best_splink < sp["custmatch_1000_labels"]["f1"] - 5, "the Splink gap has narrowed"
    for x in (sp["splink_unsupervised"]["ceiling_threshold_picked_on_truth"]["f1"],
              sp["splink_unsupervised"]["threshold_from_labels"]["f1"],
              sp["splink_m_from_500_labels"]["threshold_from_labels"]["f1"]):
        assert f"{x:.2f}" in flat
    seeded = sp["custmatch_seeded_by_splink"]
    assert f"{seeded['all_records']['f1']:.2f}" in flat and f"{seeded['seed_label_accuracy']:.2f}%" in flat
    inc = load("custmatch_incremental.json")
    gap = inc["full_rerun"]["all_records"]["f1"] - inc["incremental"]["all_records"]["f1"]
    # name-rarity features are counted over the records present, so a forest saved from a run
    # over 90% of the file sees slightly different values once new records arrive: 0.55 in
    # validation with rich features on (0.04 without)
    assert abs(gap) < 1.0, f"incremental now differs from a rerun by {gap:.2f}"
    assert f"{inc['incremental']['all_records']['f1']:.2f} against {inc['full_rerun']['all_records']['f1']:.2f}" in flat
    sc = load("custmatch_scale.json")
    big = max(sc, key=lambda row: row["records"])
    assert big["records"] >= 1_000_000 and big["full_run_seconds"] < 180
    assert big["clustered_f1"] > sc[0]["clustered_f1"] - 0.5, "accuracy must not fall with scale"


def test_splink_datasets_benchmark_claims():
    """docs/guide.md: custmatch vs Splink on Splink's own datasets, with Splink's own settings."""
    flat = " ".join(GUIDE.replace("**", "").split())
    b = load("splink_datasets_benchmark.json")
    for ds, keys in (("febrl3", ["splink_ceiling", "custmatch_1000", "custmatch_active"]),
                     ("febrl4", ["splink_documented", "splink_ceiling", "custmatch_1000", "custmatch_active"]),
                     ("historical_50k", ["splink_documented", "splink_ceiling", "custmatch_1000", "custmatch_active"]),
                     ("fake_1000", ["splink_documented_loose_blocking", "custmatch_1000_tuned", "custmatch_active_tuned",
                                    "custmatch_1000", "custmatch_active", "splink_documented"])):
        for k in keys:
            assert f"{b[ds][k]['f1']:.2f}" in flat, (ds, k)
    # the conclusions the guide draws
    for ds in ("febrl3", "febrl4"):
        assert b[ds]["custmatch_active"]["f1"] >= b[ds]["splink_ceiling"]["f1"], ds
    h = b["historical_50k"]
    assert h["splink_documented"]["f1"] > h["custmatch_1000"]["f1"] > h["custmatch_active"]["f1"]
    assert abs(h["splink_documented"]["f1"] - h["custmatch_1000"]["f1"]) < 1.5
    f = b["fake_1000"]
    assert f["custmatch_active_tuned"]["f1"] > f["splink_documented_loose_blocking"]["f1"] > f["custmatch_1000_tuned"]["f1"]


def test_beat_splink_claims():
    """docs/guide.md/README: with rich features + the Splink feature and 1,000 random labels,
    custmatch beats or ties Splink on all five datasets."""
    flat = " ".join(GUIDE.replace("**", "").split())
    b = load("beat_splink.json")
    ref, rs = b["splink_reference"], b["rich_splink"]
    for ds, r in rs.items():
        assert r["random_1000"] >= ref[ds] - 0.05, (ds, r["random_1000"], ref[ds])
        assert f"{r['random_1000']:.2f}" in flat and f"{r['active_300_700']:.2f}" in flat, ds
    gap = rs["historical_50k"]["random_1000"] - ref["historical_50k"]
    assert f"by {gap:.1f} points on the dense historical_50k" in flat


def test_blocking_lessons_claims():
    """docs/guide.md's blocking-lessons table is generated from beat_splink.json; pin the conclusions."""
    flat = " ".join(GUIDE.replace("**", "").split())
    b = load("beat_splink.json")
    base, split, learned, recut = b["rich"], b["rich_split"], b["rich_learned99"], b["rich_recut30"]
    for ds in base:
        for v in (base, split, learned):
            assert f"{v[ds]['candidate_pairs']:,}" in flat, ds
        assert learned[ds]["candidate_pairs"] < base[ds]["candidate_pairs"], f"learned blocking no smaller on {ds}"
    assert split["fake_1000"]["random_1000"] > base["fake_1000"]["random_1000"], "splitting no longer fixes fake_1000"
    assert split["historical_50k"]["random_1000"] < base["historical_50k"]["random_1000"], "the documented dense-file cost is gone"
    assert abs(split["synthetic"]["random_1000"] - base["synthetic"]["random_1000"]) < 0.5, "fixed blocking no longer level on synthetic"
    assert recut["historical_50k"]["random_1000"] > base["historical_50k"]["random_1000"]
    for ds in ("febrl3", "febrl4", "fake_1000", "synthetic"):
        assert recut[ds]["random_1000"] == base[ds]["random_1000"], f"re-cut changed {ds}"


def test_household_research_claims():
    h = load("household_research.json")
    flat = " ".join(GUIDE.replace("**", "").split())
    pred, true = h["predicted"], h["true"]
    c0, c1 = pred["contacts"]["C0"], pred["contacts"]["C1"]
    assert f"from {c0['email_household']['precision']:.1f}% to {c1['email_household']['precision']:.1f}%" in flat
    assert true["contacts"]["C1"]["email_household"]["precision"] == 100.0
    assert pred["H1_C1"]["f1"] == pred["H0_C1"]["f1"], "standardising alone now changes households"
    assert f"from {pred['H0_C1']['f1']:.2f} to {pred['H2_C1']['f1']:.2f}" in flat
    assert f"+{pred['H4_C1']['f1'] - pred['H2_C1']['f1']:.2f} together" in flat
    assert pred["H4_C1"]["f1"] > pred["H0_C1"]["f1"]


def test_placeholder_claims():
    p = load("placeholder_research.json")
    flat = " ".join(GUIDE.replace("**", "").split())
    fp = p["false_positives_clean"]
    for ds in ("febrl3", "historical_50k", "fake_1000"):
        assert fp[ds]["names_flagged"] == fp[ds]["addresses_flagged"] == fp[ds]["dates_flagged"] == 0, ds
    assert f"{fp['synthetic']['names_flagged']} names on the synthetic file" in flat
    assert fp["synthetic"]["dates"] == {"12/04/2018": fp["synthetic"]["dates_flagged"]}
    assert f"and {fp['synthetic']['dates_flagged']} dates there, all one form default (12/04/2018" in flat
    d = p["detection_injected"]
    assert f"names P {d['names']['precision']:.2f} / R {d['names']['recall']:.0f}" in flat
    assert f"dates P {d['dates']['precision']:.0f} / R {d['dates']['recall']:.0f}" in flat
    for pre in ("", "clean_"):
        off, on = p[f"{pre}placeholders_off"], p[f"{pre}placeholders_on"]
        assert (f"clustered F1 {off['all_records']['f1']:.2f} -> {on['all_records']['f1']:.2f}; households "
                f"{off['households']['f1']:.2f} -> {on['households']['f1']:.2f}") in flat
    assert p["placeholders_on"]["all_records"]["f1"] > p["placeholders_off"]["all_records"]["f1"]
    assert p["clean_placeholders_off"]["all_records"]["f1"] - p["clean_placeholders_on"]["all_records"]["f1"] < 0.1


def test_unicode_form_claims():
    u = load("unicode_form.json")
    flat = " ".join(GUIDE.split())
    changelog = " ".join((ROOT / "CHANGELOG.md").read_text().split())
    changed = {k: v["values"] for k, v in u["changed_values"].items() if v["values"]}
    assert set(changed) == {"historical_50k", "synthetic"}, changed     # the only datasets it touches
    h, s = u["runs"]["historical_50k"], u["runs"]["synthetic"]
    for name, r in u["runs"].items():
        assert r["before"]["random_1000"] == r["after"]["random_1000"], name
    assert (f"only historical_50k ({changed['historical_50k']} values, mostly d’Urban-style apostrophes) and the "
            f"synthetic file ({changed['synthetic']}) change, and clustered F1 with 1,000 random labels did not move on "
            f"either ({h['before']['random_1000']:.2f}, {s['before']['random_1000']:.2f};") in flat
    assert (f"historical_50k went from {h['before']['active_300_700']:.2f} to {h['after']['active_300_700']:.2f} "
            f"on one draw") in changelog
    assert f"changes {changed['historical_50k']} values in historical_50k" in changelog


def test_bpid_claims():
    b = load("bpid.json")
    flat = " ".join(REPORT.split())
    changelog = " ".join((ROOT / "CHANGELOG.md").read_text().split())
    old, new, paper = b["shipped_0.1.2"], b["shipped"], b["paper"]
    pct = lambda v: f"{100 * v:.2f}"  # noqa: E731
    pct1 = lambda v: f"{100 * v:.1f}"  # noqa: E731
    assert f"| custmatch 0.1.2, 8,000 labels | {pct(old['matching_8000']['f1'])} |" in flat
    for n in (300, 1000, 8000):
        m = new[f"matching_{n}"]
        assert (f"| custmatch with dates read in more forms, {n:,} labels | {pct(m['f1'])} | {pct(m['precision'])} | "
                f"{pct(m['recall'])} |") in flat
    f1 = 100 * new["matching_8000"]["f1"]
    assert f"That added {f1 - 100 * old['matching_8000']['f1']:.1f} points of F1" in flat
    assert f"custmatch scores {f1:.2f}: {f1 - 100 * paper['random_forest']:.1f} points" in flat
    assert f"{100 * paper['ditto'] - f1:.2f} below Ditto and {100 * paper['sudowoodo'] - f1:.2f} below" in flat
    assert f1 > 100 * max(paper[k] for k in ("claude3_sonnet", "llama3_70b", "mistral_large", "gpt4_turbo"))
    assert f"added {100 * (b['dob_edit']['matching_8000']['f1'] - new['matching_8000']['f1']):.2f}" in flat
    fm, mm = new["errors"]["false_matches"], new["errors"]["missed_matches"]
    assert (f"Of one draw's {fm['count']:,} false matches, {pct1(fm['shared_email_or_phone'])}% share an email or "
            f"phone - {pct1(fm['shared_contact_names_differ'])}% while") in flat
    assert f"{pct1(fm['a_name_blank'])}% have no name on one side" in flat
    assert (f"Of the {mm['count']} missed matches, {pct1(mm['phone_agrees_only_in_last_10_digits'])}% share a phone "
            f"only") in flat
    assert f"{pct1(mm['shared_address_not_both_first'])}% share an address" in flat
    assert f"and {pct1(mm['a_name_blank'])}% have no name on one side" in flat
    bl, bl_old = new["blocking"], old["blocking"]
    assert (f"proposed {pct(bl['recall'])}% of the matches at {bl['candidate_pairs'] / 1e6:.1f} million candidates "
            f"in about {round(bl['seconds'])} seconds") in flat
    assert f"with 0.1.2's date reading the keys found {pct(bl_old['recall'])}%" in flat
    assert bl["recall"] > paper["blocking_best_recall"]
    assert (f"from {pct(old['matching_8000']['f1'])} to {pct(new['matching_8000']['f1'])} with 8,000 labels, and its "
            f"blocking recall from {pct(bl_old['recall'])}% to {pct(bl['recall'])}%") in changelog


def test_audit_claims():
    a = load("audit_check.json")
    flat = " ".join(GUIDE.split())
    t = a["truth"]
    assert f"true precision {t['precision']:.2f}, recall {t['recall']:.2f}, F1 {t['f1']:.2f}" in flat
    k = ("precision", "recall", "f1")
    assert ("estimates averaged " + ", ".join(f"{a[x]['mean_estimate']:.2f}" for x in k[:2]) +
            f" and {a['f1']['mean_estimate']:.2f}") in flat
    for x in k:                                      # unbiased: within a point of the truth on average
        assert abs(a[x]["mean_estimate"] - t[x]) < 1, x
        used, plain = a[x]["intervals_holding_the_truth"], a[x]["not_used"]["plain_bootstrap"]["holding_the_truth"]
        assert used >= plain and used >= a[x]["not_used"]["beta"]["holding_the_truth"], x
    held = [a[x]["intervals_holding_the_truth"] for x in k]
    assert f"held the truth {held[0]}, {held[1]} and {held[2]} times in 100" in flat
    for name, words in (("plain_bootstrap", "resampling the labels alone held it"), ("beta", "a Beta interval per stratum")):
        v = [a[x]["not_used"][name]["holding_the_truth"] for x in k]
        assert f"{words} {v[0]}, {v[1]} and {v[2]} times" in flat, name


def test_sparse_australian_claims():
    """docs/report/ "Sparse records": every figure comes from au_validation.json."""
    flat = " ".join(GUIDE.replace("**", "").split())
    r = load("au_validation.json")
    for name in ("baseline", "baseline_active", "country_au", "country_au_active"):
        m = r[name]["mean"]
        assert (f"{m['all_records']['f1']:.2f} +- {r[name]['f1_sd']:.2f} | {m['no_surname_records']['f1']:.2f} | "
                f"{m['households']['f1']:.2f}") in flat, name
    best = r["country_au_active"]["mean"]["all_records"]["f1"]
    for other in ("baseline_active", "country_au"):
        assert best > r[other]["mean"]["all_records"]["f1"], f"country + active no longer beats {other}"
    assert r["country_au_active"]["f1_sd"] < r["country_au"]["f1_sd"], "active learning no longer steadies the result"
    for step in ("phone", "dates", "email", "nicknames", "local_nicknames", "email_user", "postcode"):
        cost = best - r[f"active_without_{step}"]["mean"]["all_records"]["f1"]
        assert f"{cost:.2f}" in flat, step
    c = r["ceilings_country_au"]
    assert f"{c['blocking_and_closure']['f1']:.2f}" in flat and f"{c['households_from_true_identities']['f1']:.2f}" in flat
    assert f"{100 - c['blocking_pair_recall']:.0f}% of same-person record pairs" in flat
    assert f"{c['people_with_an_address']}% of people" in flat



def test_classifier_bakeoff_claims():
    """docs/report/ "Which classifier": the review table and its conclusions come from
    model_review.json; the removals it justified must hold."""
    import numpy as np
    flat = _flat(GUIDE)
    r = load("model_review.json")
    models = ["forest", "hgb", "extratrees", "lightgbm", "xgboost", "catboost", "tabpfn", "tabicl"]
    names = {"febrl3": "febrl3", "febrl4": "febrl4", "fake_1000": "fake_1000", "synthetic": "synthetic",
             "au": "Australian, sparse", "historical_50k": "historical_50k"}

    def mean(ds, m, lt):
        v = [x["f1"] for k, x in r[ds][m].items() if k.startswith(lt)]
        assert len(v) == 3, (ds, m, lt)
        return float(np.mean(v))
    for ds, label in names.items():
        cells = " | ".join(f"{mean(ds, m, 'random'):.2f} / {mean(ds, m, 'active'):.2f}" for m in models)
        assert f"| {label} | {cells} |" in flat, ds
    diff = {m: [mean(d, m, lt) - mean(d, "forest", lt) for d in names for lt in ("random", "active")] for m in models[1:]}
    for m in ("hgb", "extratrees", "lightgbm", "xgboost", "catboost"):
        assert np.mean(diff[m]) < 0, f"{m} now beats the forest on average - revisit its removal"
    assert 0 < np.mean(diff["tabpfn"]) < 1, "TabPFN's gain is no longer the small one the removal assumed"
    assert f"TabPFN {np.mean(diff['tabpfn']):+.2f} on average and never more than {-min(diff['tabpfn']):.2f} behind" in flat
    from custmatch import models as cm
    assert cm.MODELS == ("forest", "forest_catboost")


def test_forest_catboost_claims():
    """docs/report/ "Which classifier": the forest + CatBoost table and its summary come from
    ensemble.json; the opt-in model must still gain on average and lose nowhere by much."""
    import numpy as np
    flat = _flat(GUIDE)
    r = load("ensemble.json")
    models = ["forest", "catboost", "forest_catboost"]
    names = {"febrl3": "febrl3", "febrl4": "febrl4", "fake_1000": "fake_1000", "synthetic": "synthetic",
             "au": "Australian, sparse", "historical_50k": "historical_50k"}

    def mean(ds, m, lt):
        v = [x["f1"] for k, x in r[ds][m].items() if k.startswith(lt)]
        assert len(v) == 3, (ds, m, lt)
        return float(np.mean(v))
    for ds, label in names.items():
        cells = " | ".join(f"{mean(ds, m, 'random'):.2f} / {mean(ds, m, 'active'):.2f}" for m in models)
        assert f"| {label} | {cells} |" in flat, ds
    diff = {m: [mean(d, m, lt) - mean(d, "forest", lt) for d in names for lt in ("random", "active")] for m in models[1:]}
    for m in models[1:]:
        assert f"| {np.mean(diff[m]):+.2f} / {min(diff[m]):+.2f} |" in flat, m
    gain, worst = np.mean(diff["forest_catboost"]), min(diff["forest_catboost"])
    assert gain > 0.2 and worst > -0.1
    secs = {m: np.mean([v["seconds"] for ds in r for v in r[ds][m].values()]) for m in ("forest", "forest_catboost")}
    assert f"| Seconds per run (mean) | {secs['forest']:.0f} | | {secs['forest_catboost']:.0f} |" in flat
    said = f"{gain:+.2f} clustered F1 on average"
    for doc in (DOCS / "guide.md", ROOT / "CHANGELOG.md", ROOT / "custmatch/models.py"):
        text = _flat(doc.read_text())
        assert said in text and f"never more than {-worst:.2f} behind" in text, doc


def test_native_script_and_correlation_clustering_claims():
    import numpy as np
    flat = " ".join(GUIDE.replace("**", "").split())
    latin, script = load("au_validation.json"), load("au_validation_script.json")
    off = script["country_au_active_no_script"]["mean"]["native_script_people"]["f1"]
    on = script["country_au_active"]["mean"]["native_script_people"]["f1"]
    ceiling = latin["country_au_active"]["mean"]["native_script_people"]["f1"]
    assert off < on and on - off >= 0.5 * (ceiling - off), "transliteration no longer recovers over half the loss"
    assert _near(GUIDE, f"{100 * (on - off) / (ceiling - off):.0f}%", "transliteration recovers")
    assert f"{off:.2f} without transliteration and {on:.2f} with it; the same people scored {ceiling:.2f}" in flat



def test_household_postcode_join_claims():
    flat = " ".join(GUIDE.replace("**", "").split())
    c = load("au_validation.json")["ceilings_country_au"]
    base, join = c["households_from_true_identities"], c["households_from_true_identities_postcode_join"]
    assert (f"raised household F1 from {base['f1']:.2f} to {join['f1']:.2f} but cut precision from "
            f"{base['precision']:.2f} to {join['precision']:.2f}") in flat
    assert join["precision"] < base["precision"] - 2, "the postcode join no longer costs precision: revisit"


def _flat(text):
    return " ".join(text.replace("**", "").split())


def _near(text, number, *words, window=160):
    """True when `number` appears within `window` characters of every one of `words` - a claim
    in context, not a number that happens to occur somewhere else in the docs."""
    import re
    flat = _flat(text)
    for m in re.finditer(re.escape(number), flat):
        around = flat[max(0, m.start() - window): m.end() + window].lower()
        if all(w.lower() in around for w in words):
            return True
    return False


def test_readme_headlines():
    """The headline comparison (README, now the report's "How custmatch compares"), each number with
    the words around it (a bare number can match anywhere)."""
    readme = _flat(README)
    v = load("custmatch_validation.json")
    assert f"{v['hand_1000']['all_records']['f1']:.1f} clustered F1 from 1,000 labelled pairs" in readme
    assert f"{v['active_1000']['all_records']['f1']:.1f} with active learning" in readme
    assert f"households at {v['hand_1000']['households']['f1']:.1f} F1" in readme
    sp = load("splink_vs_custmatch.json")
    best = max(sp["splink_unsupervised"]["ceiling_threshold_picked_on_truth"]["f1"],
               sp["splink_m_from_500_labels"]["threshold_from_labels"]["f1"])
    assert f"(Splink's best: {best:.1f})" in readme
    import numpy as np
    rr = load("cdp_reassess.json")

    def ra(scen, src, k):
        vals = []
        for key in (scen, f"{scen}_s1", f"{scen}_s2"):
            x = rr[key]["runs"][0]
            vals.append(x["segment"][src][k] if src.startswith("limit") else x["custmatch"][src]["at_its_threshold"][k])
        return float(np.mean(vals))
    P = "limit1_block3_cookie1"
    assert (f"many sign-ups: precision / recall | {ra('high', 'clean_1000', 'precision'):.1f} / {ra('high', 'clean_1000', 'recall'):.1f} | "
            f"CDP-style exact matching, tuned for precision: {ra('high', P, 'precision'):.1f} / {ra('high', P, 'recall'):.1f}") in readme
    assert f"few sign-ups | {ra('low', 'clean_1000', 'precision'):.1f} / {ra('low', 'clean_1000', 'recall'):.1f} | {ra('low', P, 'precision'):.1f}" in readme
    assert (f"{ra('high', 'min_precision_097', 'wrong_person_email_pct'):.2f}% / {ra('low', 'min_precision_097', 'wrong_person_email_pct'):.2f}% | "
            f"CDP tuned for precision: {ra('high', P, 'wrong_person_email_pct'):.2f}% / {ra('low', P, 'wrong_person_email_pct'):.2f}%") in readme
    sep = load("separation.json")
    ds = ["au_customer", "au_customer_s1", "au_customer_s2"]
    sp = lambda m: float(np.mean([sep[d]["runs"][0][m]["merged_with_ex_pct"] for d in ds]))  # noqa: E731
    assert f"merged with the ex | {sp('custmatch'):.1f}% | CDP-style, tuned: {sp('cdp_tuned'):.1f}%" in readme
    nc = load("ncvr_leipzig.json")["postcode_initial_keys/people_200000"]["runs"]
    cm = [x["clustered"]["f1"] for x in nc]
    spl = float(np.mean([x["splink"]["m_from_labels"]["p_0.99"]["f1"] for x in nc]))
    assert len(nc) == 3 and f"3 draws: clustered F1 | {np.mean(cm):.1f} ({min(cm):.1f}-{max(cm):.1f}) | Splink, given the same labels: {spl:.1f}" in readme
    au = load("au_validation.json")
    assert (f"from {au['baseline']['mean']['all_records']['f1']:.1f} to "
            f"{au['country_au_active']['mean']['all_records']['f1']:.1f}") in readme


def test_forest_against_splink_over_three_draws():
    """README and guide: over three label draws the default forest beats Splink on four of five."""
    import numpy as np
    b, ref = load("matcher_bakeoff.json"), load("beat_splink.json")["splink_reference"]
    mean = {ds: float(np.mean([v["f1"] for k, v in b[ds]["forest"].items() if k.startswith("random_1000")]))
            for ds in ref}
    ahead = [ds for ds in ref if mean[ds] > ref[ds]]
    behind = [ds for ds in ref if ds not in ahead]
    assert len(ahead) == 4 and behind == ["febrl3"], (ahead, behind)
    gap = ref["febrl3"] - mean["febrl3"]
    for text in (_flat(GUIDE),):
        assert f"beats Splink's unsupervised result on four and is {gap:.1f} behind on febrl3" in text or \
               f"beats Splink's unsupervised result (no labels) on four and is {gap:.1f} behind on febrl3" in text
    order = ("febrl3", "febrl4", "fake_1000", "synthetic", "historical_50k")
    assert " / ".join(f"{mean[d]:.2f}" for d in order) in _flat(GUIDE)


def test_speed_claims_match_the_scale_runs():
    default, fixed = load("custmatch_scale.json")[-1], load("custmatch_scale_fixed.json")[-1]
    assert default["records"] > 1_000_000 - 10_000
    for text in (_flat(GUIDE),):
        assert f"about {default['full_run_seconds'] / 60:.1f} minutes" in text
        assert f"{round(fixed['full_run_seconds'])} s with \"blocking\": \"fixed\"" in text.replace("`", "")
    assert f"blocking takes {round(default['block_seconds'])} s of {round(default['full_run_seconds'])}" in _flat(GUIDE)
    assert "under 2 minutes" not in _flat((ROOT / "README.md").read_text()) + _flat(GUIDE)


def test_bootstrap_intervals_come_from_the_script():
    b = load("jev_bootstrap.json")["historical50k-person-hard/criteria"]
    lo, hi = b["f1_interval"]
    assert f"by {b['f1_at_0.5_change']:.1f} (95% interval +{lo:.1f} to +{hi:.1f}" in _flat(REPORT)
    sc = load("jev_bootstrap.json")["historical50k-person-hard/structured_criteria"]
    assert f"AUC +{sc['auc_change']:.3f} on historical_50k, interval +{sc['auc_interval'][0]:.3f} to +{sc['auc_interval'][1]:.3f}" in _flat(REPORT)
    st = load("jev_bootstrap.json")["historical50k-person-hard/structured"]
    assert f"{st['f1_at_0.5_change']:.1f} F1 at 0.5".replace("-", "−") in _flat(REPORT)


def test_country_profile_claims():
    """docs/guide.md "Country profiles": the benchmark table comes from matcher_bakeoff.json."""
    import numpy as np
    b = load("matcher_bakeoff.json")
    flat = _flat(GUIDE)

    def mean(ds, m, lt):
        return float(np.mean([v["f1"] for k, v in b[ds][m].items() if k.startswith(lt)]))
    for ds, tag, c in (("synthetic", "forest_country_us", "US"), ("febrl3", "forest_country_au", "AU"),
                       ("febrl4", "forest_country_au", "AU"), ("historical_50k", "forest_country_gb", "GB")):
        row = (f"| {ds} | `{c}` | {mean(ds, 'forest', 'random_1000'):.2f} / {mean(ds, 'forest', 'active_300_700'):.2f} | "
               f"{mean(ds, tag, 'random_1000'):.2f} / {mean(ds, tag, 'active_300_700'):.2f} |")
        assert row in flat, ds
        for lt in ("random_1000", "active_300_700"):
            assert abs(mean(ds, tag, lt) - mean(ds, "forest", lt)) < 1.5, f"a profile moved {ds} by more than noise"


def test_order_dependence_claims():
    """docs/guide.md "Stable matches": the table and conclusions come from order_dependence.json."""
    import numpy as np
    r = load("order_dependence.json")
    flat = _flat(GUIDE)

    def mean(m, k, couples=False):
        return float(np.mean([v[k] for key, v in r.items()
                              if key.startswith(m + "/") and key.endswith("/couples") == couples]))
    for m in ("rerun", "rerun_fixed", "rerun_fixed_ids", "rerun_hyst05", "add_fixed"):
        assert f"| {mean(m, 'churn'):,.0f} | {mean(m, 'returns'):.1f} | {mean(m, 'id_changes'):,.0f} | {mean(m, 'final_f1'):.2f} |" in flat, m
    assert mean("rerun_fixed", "churn") < 0.6 * mean("rerun", "churn"), "retraining no longer a big cause"
    assert _near(GUIDE, f"{100 * (1 - mean('rerun_fixed', 'churn') / mean('rerun', 'churn')):.0f}%", "accounts for")
    assert mean("rerun_hyst05", "churn") < 0.85 * mean("rerun_fixed_ids", "churn")
    assert mean("rerun_hyst05", "final_f1") > mean("rerun_fixed_ids", "final_f1") - 0.5
    assert mean("add_fixed", "churn") == 0
    moves = sum(v["couple_record_moves"] for k, v in r.items() if k.endswith("/couples"))
    assert f"moved between partners {moves} times" in flat


def test_record_dates_claims():
    """docs/guide.md "Record dates": the table, and the gain across all six runs, from record_dates.json."""
    import numpy as np
    r = load("record_dates.json")
    flat = _flat(GUIDE)
    f1 = lambda v: [x["all_records"]["f1"] for x in r[v]["runs"]]  # noqa: E731
    assert f"| Clustered F1, all records | {np.mean(f1('without_dates')):.2f} | {np.mean(f1('record_dates')):.2f} |" in flat
    gains = [a - b for a, b in zip(f1("record_dates"), f1("without_dates"))]
    for k in [k for k in r if k.startswith("data_seed")]:
        gains.append(r[k]["record_dates"]["runs"][0]["all_records"]["f1"] - r[k]["without_dates"]["runs"][0]["all_records"]["f1"])
    assert len(gains) == 6 and min(gains) > 0, "dates no longer help in every run"
    assert _near(GUIDE, f"{min(gains):.1f} to {max(gains):.1f} points", "all six runs")

def test_scenario_claims():
    """docs/report/ "Scenarios": the scorecard rows and conclusions come from scenarios.json."""
    r = load("scenarios.json")
    flat = _flat(GUIDE)
    V = ("default", "twin_guard", "active", "twin_guard_active")
    for n, m in r["default"]["scenarios"].items():
        row = f"| {m['description']} | {m['expected']} | " + " | ".join(f"{r[v]['scenarios'][n]['passed_pct']:.0f}%" for v in V) + " |"
        assert _flat(row) in flat, n
    assert r["default"]["scenarios"]["twins"]["passed_pct"] == 0
    assert all(x["passed_pct"] == 100 for n, x in r["twin_guard_active"]["scenarios"].items() if n != "no_surname_email")
    assert r["twin_guard_active"]["scenarios"]["no_surname_email"]["passed_pct"] >= 90
    assert r["default"]["background_f1"] - r["twin_guard"]["background_f1"] < 0.5


def test_cookie_claims():
    """docs/guide.md "Cookies and device IDs": the table and conclusions come from cookies.json."""
    import numpy as np
    r = load("cookies.json")
    flat = _flat(GUIDE)
    f1 = lambda v: float(np.mean([x["all_records"]["f1"] for x in r[v]["runs"]]))  # noqa: E731
    assert f"| Cookies, both guards (the default) | {f1('cookies'):.2f} |" in flat
    assert f"| No cookies | {f1('no_cookies'):.2f} |" in flat
    gain = [a["all_records"]["f1"] - b["all_records"]["f1"] for a, b in zip(r["cookies"]["runs"], r["no_cookies"]["runs"])]
    assert min(gain) > 0
    assert f1("cookies_no_name_check") < f1("no_cookies") < f1("cookies")
    assert f1("cookies_unprotected") < 20


def test_ncvr_claims():
    """docs/report/ "A public benchmark": the table rows come from ncvr_leipzig.json."""
    import numpy as np
    r = load("ncvr_leipzig.json")
    flat = _flat(GUIDE)
    for key in r:
        f1 = np.mean([x["clustered"]["f1"] for x in r[key]["runs"]])
        assert f"| {f1:.2f} |" in flat, key
    a, b = r["default_keys/people_20000"], r["postcode_initial_keys/people_20000"]
    assert b["mean_f1"] > a["mean_f1"] + 3
    a, b = r["default_keys/people_200000"], r["postcode_initial_keys/people_200000"]
    assert b["mean_ceiling_f1"] > a["mean_ceiling_f1"] + 5 and b["mean_f1"] > a["mean_f1"]


def test_cdp_comparison_claims():
    """docs/report/ "Deterministic CDP matching vs custmatch": rows come from cdp_comparison.json."""
    import numpy as np
    r = load("cdp_comparison.json")
    flat = _flat(GUIDE)
    f1 = lambda sc, m: float(np.mean([x[m]["all_records"]["f1"] for x in r[sc]["runs"]]))  # noqa: E731
    for m in ("segment_raw", "segment_normalised", "segment_tuned", "custmatch"):
        assert f"{f1('high', m):.1f} (" in flat and f"{f1('low', m):.1f} (" in flat, m
    assert f1("high", "custmatch") > f1("high", "segment_tuned") and f1("low", "custmatch") > f1("low", "segment_tuned")
    drop_seg = f1("high", "segment_tuned") - f1("low", "segment_tuned")
    drop_cm = f1("high", "custmatch") - f1("low", "custmatch")
    assert drop_seg > drop_cm
    assert f"lost about {drop_seg:.0f} points from high to low sign-ups, custmatch about {drop_cm:.0f}" in flat


def test_ncvr_splink_and_anonlink_claims():
    """docs/guide.md NCVR: the Splink and anonlink-idea tables come from ncvr_leipzig.json."""
    import numpy as np
    r = load("ncvr_leipzig.json")
    flat = _flat(GUIDE)
    f = lambda k, m="f1": float(np.mean([x["clustered"][m] for x in r[k]["runs"]]))  # noqa: E731
    for k in ("postcode_initial_keys_unique_sources", "lsh_blocking"):
        assert f"| {f(k + '/people_20000'):.2f} | {f(k + '/people_20000', 'precision'):.2f} | {f(k + '/people_200000'):.2f} |" in flat, k
    seed1 = lambda k: next(x for x in r[k]["runs"] if x.get("seed", 1) == 1)["clustered"]["precision"]  # noqa: E731
    assert seed1("postcode_initial_keys_unique_sources/people_200000") > seed1("postcode_initial_keys/people_200000") + 1
    runs = r["postcode_initial_keys/people_200000"]["runs"]
    sp = [x["splink"]["m_from_labels"]["p_0.99"]["f1"] for x in runs]
    cm = [x["clustered"]["f1"] for x in runs]
    assert len(runs) == 3 and all(c > x for c, x in zip(cm, sp)), "custmatch no longer leads Splink on every draw"
    assert _near(GUIDE, "custmatch led on every draw", "200,000 people")


def test_cdp_reassessment_claims():
    """docs/report/ "Reassessed for marketing": every table row, and the comparisons in the text,
    from cdp_reassess.json (three generated populations, one label draw each)."""
    import numpy as np
    r = load("cdp_reassess.json")
    flat = _flat(GUIDE)
    for scen, label in (("high", "Many sign-ups"), ("low", "Few sign-ups")):
        keys = [scen, f"{scen}_s1", f"{scen}_s2"]
        seg = [r[k]["runs"][0]["segment"] for k in keys]
        cm = lambda v: [r[k]["runs"][0]["custmatch"][v]["at_its_threshold"] for k in keys]  # noqa: E731
        rows = {"CDP-style, tuned for F1": [x["limit1_block5"] for x in seg],
                "CDP-style, tuned for precision": [x["limit1_block3_cookie1"] for x in seg],
                "custmatch, default threshold": cm("clean_1000"), "custmatch, min_precision 0.97": cm("min_precision_097")}
        for meth, rs in rows.items():
            mean = lambda k: float(np.mean([x[k] for x in rs]))  # noqa: E731
            row = f"| {label} | {meth} | {mean('precision'):.1f} ("
            assert row in flat.replace("`", ""), row
            assert f"| {mean('recall'):.1f} | {mean('wrong_person_email_pct'):.2f}% |" in flat, (scen, meth)
        m = {k: float(np.mean([x["recall"] for x in v])) for k, v in rows.items()}
        assert 15.5 <= m["custmatch, default threshold"] - m["CDP-style, tuned for F1"] <= 26
        assert 5.5 <= m["CDP-style, tuned for precision"] - m["custmatch, min_precision 0.97"] <= 10.5
        e = {k: float(np.mean([x["wrong_person_email_pct"] for x in v])) for k, v in rows.items()}
        assert e["custmatch, min_precision 0.97"] <= 0.11 * e["CDP-style, tuned for precision"]
        st = {k: float(np.mean([x["false_pairs_strangers"] for x in v])) for k, v in rows.items()}
        assert 1.9 <= st["custmatch, default threshold"] / st["CDP-style, tuned for F1"]
        assert st["custmatch, default threshold"] / st["CDP-style, tuned for precision"] <= 7.5
    assert _near(GUIDE, "16 to 26 points more", "custmatch at its default")


def test_separation_claims():
    """docs/report/ "Couples who separate": every table row comes from separation.json."""
    import numpy as np
    r = load("separation.json")
    flat = _flat(GUIDE).replace("`", "")
    ds = ["au_customer", "au_customer_s1", "au_customer_s2"]
    m = lambda k, f: float(np.mean([r[d]["runs"][0][k][f] for d in ds]))  # noqa: E731
    for k in ("cdp_tuned", "cdp_tuned_newest", "custmatch", "custmatch_contact_safety", "custmatch_newest",
              "custmatch_min_precision", "custmatch_newest_min_precision"):
        assert (f"| {m(k, 'merged_with_ex_pct'):.2f}% | {m(k, 'email_reaches_ex_matched_pct'):.2f}% | "
                f"{m(k, 'email_reaches_ex_own_record_pct'):.2f}% | {m(k, 'post_to_ex_address_pct'):.2f}% |") in flat, k
    for d in ds:                                    # the CDP merges the leaver with the ex more in every population
        x = r[d]["runs"][0]
        assert x["custmatch"]["merged_with_ex_pct"] < x["cdp_tuned"]["merged_with_ex_pct"] / 3, d
    assert m("custmatch_contact_safety", "email_reaches_ex_matched_pct") < m("custmatch", "email_reaches_ex_matched_pct") / 2


def test_threshold_rule_claims():
    """docs/guide.md "Choosing among thresholds": the table comes from threshold_rules.json, and the
    rule chosen on the development cases must hold on the held-out public benchmarks."""
    r = load("threshold_rules.json")["summary"]
    flat = _flat(GUIDE)
    for g in ("development", "held_out_public_benchmarks"):
        m = r[g]["middle_0.5"]
        assert f"| middle of near-ties | {m['mean_regret']:.2f} | {m['max_regret']:.2f} |" in flat, g
        assert m["mean_regret"] < r[g]["current"]["mean_regret"] and m["max_regret"] < r[g]["current"]["max_regret"], g


def test_dense_file_claims():
    """docs/report/ "Dense files": the historical_50k blocking ceilings, the three-draw rerun with
    similarity blocking, and the learning curve."""
    import numpy as np
    flat = _flat(GUIDE)
    hb, hl = load("historical_blocking.json"), load("historical_labels.json")
    bk = load("matcher_bakeoff.json")["historical_50k"]["forest_similarity_blocking"]
    d, l = hb["default"], hb["lsh_names_birth_place"]
    assert l["ceiling"]["f1"] > d["ceiling"]["f1"] + 3
    assert _near(flat, f"{d['ceiling']['f1']:.2f}", "perfect matcher")
    assert _near(flat, f"{l['ceiling']['f1']:.2f}", "ceiling", f"{100 * (l['candidates'] / d['candidates'] - 1):.0f}% more")
    m = {lt: float(np.mean([v["f1"] for k, v in bk.items() if k.startswith(lt)])) for lt in ("random_1000", "active_300_700")}
    assert len(bk) == 6
    assert m["random_1000"] > d["mean_random"] and m["active_300_700"] > d["mean_active"], "similarity blocking no longer helps"
    assert (f"forest scores {m['random_1000']:.2f} with 1,000 random labels and {m['active_300_700']:.2f} with active ones"
            in flat)
    assert f"against {d['mean_random']:.2f} and {d['mean_active']:.2f} with the exact keys alone" in flat
    lc = {(v, n): float(np.mean([hl[f"{v}_{n}_{s}"]["f1"] for s in (1, 2)]))
          for v in ("default", "lsh_names_birth_place") for n in (1000, 3000, 6000)}
    for v in ("default", "lsh_names_birth_place"):
        assert " / ".join(f"{lc[(v, n)]:.2f}" for n in (1000, 3000, 6000)) in flat, v
    best = max(v["best_in_hindsight"]["f1"] for v in hl.values())
    assert best < l["ceiling"]["f1"] - 5, "the matcher is no longer the limit"
    assert _near(flat, f"{best:.2f}", "hindsight")
    assert f"({hb['gb']['mean_f1']:.2f} against {d['mean_f1']:.2f})" in flat


def test_match_weight_claims():
    """docs/report/ "A match weight of custmatch's own": the comparison table, the gains, and the
    scenario cost that keeps it opt-in."""
    import numpy as np
    flat = _flat(GUIDE)
    r, sc = load("match_weight.json"), load("scenarios.json")
    names = {"febrl3": "febrl3", "febrl4": "febrl4", "fake_1000": "fake_1000", "synthetic": "synthetic",
             "au": "Australian, sparse", "historical_50k": "historical_50k"}

    def mean(ds, m, lt):
        v = [x["f1"] for k, x in r[ds][m].items() if k.startswith(lt)]
        assert len(v) == 3, (ds, m, lt)
        return float(np.mean(v))
    diffs = []
    for ds, label in names.items():
        cells = [f"{mean(ds, m, 'random'):.2f} / {mean(ds, m, 'active'):.2f}" for m in ("forest", "forest_match_weight")]
        assert f"| {label} | {cells[0]} | {cells[1]} |" in flat, ds
        diffs += [mean(ds, "forest_match_weight", lt) - mean(ds, "forest", lt) for lt in ("random", "active")]
    g_au = mean("au", "forest_match_weight", "random") - mean("au", "forest", "random")
    g_h = mean("historical_50k", "forest_match_weight", "random") - mean("historical_50k", "forest", "random")
    assert g_au > 2 and g_h > 0.5 and min(diffs) > -0.5, (g_au, g_h, min(diffs))
    assert f"{g_au:+.2f} on the sparse Australian file and {g_h:+.2f} on historical_50k" in flat
    assert f"at most {-min(diffs):.2f} behind anywhere" in flat
    # the reason it is opt-in: the web sign-up case fails with the weight, holds without it (active)
    sign_up = {v: [run["scenarios"]["no_surname_email"] for run in sc[v]["runs"]]
               for v in ("active", "active_match_weight")}
    assert min(sign_up["active"]) >= 90 and max(sign_up["active_match_weight"]) <= 50, sign_up
    for v in ("match_weight", "active_match_weight"):
        assert all(run["scenarios"][s] == 100 for run in sc[v]["runs"] for s in ("moved", "married_name")), v


def test_large_scale_claims():
    """docs/report/ "Larger files" and README: 2, 5 and 10 million records, before and after the
    memory redesign, from scale_large.json."""
    r = load("scale_large.json")
    flat, readme = _flat(GUIDE), _flat(README)
    for k in ("scale_2m", "scale_5m", "scale_10m"):
        a, b = r[f"{k}_before_memory_redesign"], r[k]
        x = b["result"]
        t = sorted([a["result"]["full_run_seconds"] / 60, x["full_run_seconds"] / 60])
        span = f"{t[0]:.0f} min" if round(t[0]) == round(t[1]) else f"{t[0]:.0f} to {t[1]:.0f} min"
        assert (f"| {x['records'] / 1e6:.1f} million | {x['run_candidate_pairs'] / 1e6:.1f}M | {span} | "
                f"{a['peak_memory_gb']:.1f} / {b['peak_memory_gb']:.1f} GB |") in flat, k
        assert x["clustered_f1"] == a["result"]["clustered_f1"], f"the redesign changed {k}'s result"
        assert x["clustered_f1"] > 96, f"accuracy fell at {k}"
    a, b = r["scale_10m_before_memory_redesign"], r["scale_10m"]
    t = sorted([a["result"]["full_run_seconds"] / 60, b["result"]["full_run_seconds"] / 60])
    assert f"ten million {t[0]:.0f} to {t[1]:.0f} minutes and {b['peak_memory_gb']:.0f} GB" in readme


def test_guide_points_at_real_sections():
    """The guide sends readers to sections by name - "X" below, or "X" in a report chapter - and each
    must exist in the file named, as a heading or a bold-led paragraph."""
    import re
    raw = (DOCS / "guide.md").read_text()
    guide = " ".join(raw.split())
    for name in re.findall(r'"([^"]{4,80})" below', guide):
        assert re.search(rf"^#+ {re.escape(name)}", raw, re.M) or f"**{name}**" in guide, name
    pointers = re.findall(r'"([^"]{4,80})" in \[report/([\w-]+\.md)\]', guide)
    pointers += [(t, f) for f, t in re.findall(r'report/([\w-]+\.md)\)? "([^"]{4,80})"', guide)]
    assert pointers, "the guide no longer points into the report"
    for name, f in pointers:
        chapter = (DOCS / "report" / f).read_text()
        assert re.search(rf"^#+ {re.escape(name)}", chapter, re.M) or f"**{name}" in " ".join(chapter.split()), (name, f)


def test_full_ncvr_claims():
    """docs/report/ "Larger files": the full 5-million-record NCVR run, from scale_large.json."""
    r = load("scale_large.json")["ncvr_all"]["result"]
    flat = _flat(GUIDE)
    assert f"{r['records']:,} records of {r['people']:,} people" in flat
    assert f"It scored {r['clustered']['f1']:.2f} clustered F1" in flat
    assert any("randomly drawn labels are matches" in w for w in r["warnings"]), "the few-random-matches warning no longer fires"


def test_address_book_stratified_label_claims():
    """docs/report/06 "Random labels that still hold matches", the guide and the changelog: plain
    against stratified 300 random labels, ten draws on each of three synthetic address books."""
    r = load("address_book.json")
    text = _flat((DOCS / "report" / "06-blocking-and-scale.md").read_text())
    flat = _flat(GUIDE + (ROOT / "CHANGELOG.md").read_text())
    cell = {}
    for v in ("dirty", "clean"):
        for b in ("random_300", "stratified_300"):
            runs = [x for k, x in r.items() if k.split("/")[1:3] == [v, b] and "/draw" in k]
            assert len(runs) == 30, (v, b)
            f1 = [x["f1"] for x in runs]
            cell[v, b] = (fmean(f1), min(f1), fmean([x["random_matches"] for x in runs]))
    for v, row in (("dirty", "As generated: clustered F1, mean (lowest draw)"), ("clean", "Relay records removed, switchboard blanked")):
        p, q = cell[v, "random_300"], cell[v, "stratified_300"]
        assert f"| {row} | {p[0]:.2f} ({p[1]:.2f}) | {q[0]:.2f} ({q[1]:.2f}) |" in text, v
    assert (f"| Random matches among the 300 | {cell['dirty', 'random_300'][2]:.1f} | "
            f"{cell['dirty', 'stratified_300'][2]:.1f} / {cell['clean', 'stratified_300'][2]:.1f} |") in text
    assert f"from {cell['dirty', 'random_300'][0]:.2f} to {cell['dirty', 'stratified_300'][0]:.2f} over 30 draws" in flat
    gains = []
    for book in (1, 2, 3):
        for v in ("dirty", "clean"):
            m = {b: fmean([r[f"s{book}/{v}/{b}/draw{d}"]["f1"] for d in range(1, 11)]) for b in ("random_300", "stratified_300")}
            gains.append(m["stratified_300"] - m["random_300"])
    assert min(gains) > 0, "stratified labels no longer win on every address book"
    assert f"(by {min(gains):.1f} to {max(gains):.1f})" in text


def test_address_book_cards_claims():
    """docs/report/06 "Address-book cards": Profile B before and after, from address_book.json."""
    r = load("address_book.json")
    text = _flat((DOCS / "report" / "06-blocking-and-scale.md").read_text())
    rows = {"random_300": "300 random", "stratified_300": "300 stratified", "random150_active150": "150 random + 150 active",
            "stratified300_active150": "300 stratified + 150 active", "all": "Every candidate pair labelled (3 runs)"}
    cells = {}
    for budget, row in rows.items():
        out = []
        for tag in ("/before", ""):
            runs = [v for k, v in r.items() if k.split("/")[1:3] == ["cards", budget]
                    and (k.endswith(tag) if tag else k.count("/") == 3)]
            assert len(runs) == (3 if budget == "all" else 15), (budget, tag)
            f1 = fmean([v.get("f1", 0.0) for v in runs])       # a run that stops scores 0
            stopped = sum("error" in v for v in runs)
            found = fmean([int(v["duplicates_found"].split()[0]) if "duplicates_found" in v else 0 for v in runs])
            out.append(f"{f1:.2f} ({stopped}) | {found:.1f} of 11")
            cells[budget, tag] = f1
        assert f"| {row} | {out[0]} | {out[1]} |" in text, budget
    assert cells["random150_active150", ""] > cells["random150_active150", "/before"] + 30
    ph = {t: [v["f1"] for k, v in r.items() if "/cards/stratified300_active150/" in k and k.endswith(t)]
          for t in ("/placeholders_ignored", "/placeholders_compared")}
    assert all(len(v) == 15 for v in ph.values())
    assert f"changed nothing ({fmean(ph['/placeholders_ignored']):.2f} against {fmean(ph['/placeholders_compared']):.2f}" in text


def test_full_ncvr_reruns_claims():
    """docs/report/ "Larger files": the full NCVR reruns after the memory redesign."""
    r = load("scale_large.json")
    flat = _flat(GUIDE)
    peak = lambda x: max(v["peak_memory_gb_so_far"] for v in x["progress"].values())  # noqa: E731
    old, new, fx = r["ncvr_all"], r["ncvr_all_default_memory_redesign"], r["ncvr_all_fixed_blocking"]
    ad, nothing = r["ncvr_all_adaptive_blocking"], r["ncvr_all_current_default"]
    for x in (old, new, fx, ad, nothing):
        assert f"| {x['result']['clustered']['f1']:.2f} | {x['result']['clustered']['precision']:.1f} / {x['result']['clustered']['recall']:.1f} |" in flat
        assert f"{peak(x):.1f} GB |" in flat
    assert peak(new) < 0.5 * peak(old), "the memory redesign no longer halves full-scale memory"
    assert abs(new["result"]["clustered"]["f1"] - old["result"]["clustered"]["f1"]) < 1, "the redesign changed accuracy"
    assert fx["result"]["clustered"]["f1"] > new["result"]["clustered"]["f1"]
    assert ad["result"]["clustered"]["f1"] > fx["result"]["clustered"]["f1"], "adaptive blocking no longer beats fixed at full size"
    assert f"it scored {ad['result']['clustered']['f1']:.2f}, the best full-file result" in flat
    assert f"Only {ad['result']['random_matches']} of its 1,000 random labels were matches" in flat
    assert f"cut peak memory by {100 * (1 - peak(new) / peak(old)):.0f}%" in flat
    # nothing set: the switch to adaptive and stratified labels by themselves, 300 random + 700 active
    assert nothing["result"]["config_changes"] == {"blocking": None}
    assert any(w.startswith('blocking switched to "adaptive"') for w in nothing["result"]["warnings"])
    f1 = nothing["result"]["clustered"]["f1"]
    assert f"scored {f1:.2f} (the table's last row): {f1 - new['result']['clustered']['f1']:.2f} above the default" in flat
    assert f"but {ad['result']['clustered']['f1'] - f1:.2f} below the 1,000-random-label run" in flat
    assert f"{nothing['result']['random_matches']} of its 300 random labels were matches, against " \
           f"{old['result'].get('random_matches', 11)} and {new['result']['random_matches']} in plain draws of 300" in flat


def test_tier1_speedups_match_the_results():
    """The guide's Tier 1 paragraph quotes research/results/scale_large.json."""
    import statistics
    r = load("scale_large.json")
    flat = " ".join(GUIDE.split())
    local = r["tier1_local_1m"]
    before, after = statistics.median(local["before_seconds"]), statistics.median(local["after_seconds"])
    assert f"a median of {before:.0f} s before, {after:.0f} s after" in flat
    a, b = r["scale_10m"], r["scale_10m_tier1"]
    mins = (a["result"]["full_run_seconds"] / 60, b["result"]["full_run_seconds"] / 60)
    assert f"{mins[0]:.0f} minutes before, {mins[1]:.0f} after" in flat
    fa = next(v for k, v in a["result"]["stage_seconds"].items() if k.startswith("forest"))
    fb = next(v for k, v in b["result"]["stage_seconds"].items() if k.startswith("forest"))
    assert f"the forest stage {fa:.0f} s to {fb:.0f} s" in flat
    assert b["result"]["clustered_f1"] == a["result"]["clustered_f1"]
    side = r["scale_2m_tier1_same_image"]
    pb, pa = side["before"]["peak_memory_gb"], side["after"]["peak_memory_gb"]
    assert f"({pb:.1f} to {pa:.1f} GB at two million" in flat
    assert f"rose about {100 * (pa / pb - 1):.0f}%" in flat


def test_tier2_speedups_match_the_results():
    """The guide's second-round paragraph quotes research/results/scale_large.json."""
    r = load("scale_large.json")
    flat = " ".join(GUIDE.split())
    a, c = r["scale_10m"], r["scale_10m_tier2"]
    last = lambda x: next(v for k, v in x["result"]["stage_seconds"].items() if k.startswith("households"))  # noqa: E731
    assert f"took {c['result']['full_run_seconds'] / 60:.0f} minutes, the last stage {last(c):.0f} s (from {last(a):.0f} s" in flat
    assert f"peak memory back to {c['peak_memory_gb']:.1f} GB" in flat
    assert c["result"]["clustered_f1"] == a["result"]["clustered_f1"]


def test_head_to_head_and_home_data_claims():
    """docs/report/ "Compared with other open-source tools" and the README rows: custmatch,
    Splink 5 and dedupe from head_to_head.json and home_data_benchmark.json."""
    import numpy as np
    flat, readme = _flat(GUIDE), _flat(README)
    h = load("head_to_head.json")
    big, small = h["copies_50"], h["copies_1"]
    c, sp = big["custmatch"], big["splink_m_from_labels_capped"]
    assert (f"| {c['clustered']['f1']:.2f} / {c['wall_seconds']:.1f} / {c['peak_memory_gb']:.2f} GB | "
            f"Splink 5, large blocks skipped: {sp['clustered']['f1']:.2f} / {sp['splink_seconds']:.1f} / "
            f"{sp['peak_memory_gb']:.2f} GB |") in readme
    for key, label in (("m_from_labels_capped", "Splink, blocks over 60 records skipped, m from labels"),
                       ("em_capped", "Splink, blocks over 60 records skipped, EM"),
                       ("m_from_labels", "Splink, blocking as above with no cap, m from labels"),
                       ("em", "Splink, blocking as above with no cap, EM")):
        r = big[f"splink_{key}"]
        assert (f"| {label} | {r['splink_seconds']:.1f} | {r['peak_memory_gb']:.2f} GB | "
                f"{r['candidate_pairs_above_0.001'] / 1e6:.1f} M | {r['clustered']['precision']:.2f} | "
                f"{r['clustered']['recall']:.2f} | {r['clustered']['f1']:.2f} |") in flat, key
    assert c["clustered"]["f1"] - max(v["clustered"]["f1"] for k, v in big.items() if k.startswith("splink_") and isinstance(v, dict)) > 15
    d = small["dedupe"]
    assert (f"| {small['custmatch']['clustered']['f1']:.2f} / {small['custmatch']['wall_seconds']:.1f} / "
            f"{small['custmatch']['peak_memory_gb']:.2f} GB | dedupe: {d['clustered']['f1']:.2f} / "
            f"{d['dedupe_seconds']:.0f} / {d['peak_memory_gb']:.1f} GB |") in readme
    assert f"it scored {d['clustered']['f1']:.2f} against custmatch's {small['custmatch']['clustered']['f1']:.2f}" in flat

    r = load("home_data_benchmark.json")
    for ds in ("fake_1000", "febrl3", "febrl4", "historical_50k", "dedupe_csv_example", "dedupe_patent_example"):
        runs = [r[ds][k]["f1"] for k in ("dedupe", "dedupe_run2", "dedupe_run3")]
        cell = f"{np.mean(runs):.2f}" + (f" ({min(runs):.2f}-{max(runs):.2f})" if len(set(runs)) > 1 else "")
        chapter9 = (DOCS / "report" / "09-against-other-tools.md").read_text()
        row = next(line for line in chapter9.splitlines() if line.startswith(f"| {ds.removeprefix('dedupe_')} ("))
        assert f"{r[ds]['custmatch_1000']['f1']:.2f}" in row and cell in row.replace("**", ""), (ds, row)
        for key, v in r[ds].items():
            if key.startswith("custmatch_1000_max_block"):        # the config the run's warning advised
                assert f"{v['f1']:.2f} with `\"max_block\": {v['config']['max_block']}`" in row.replace("**", ""), key
            elif key.startswith("custmatch_1000_"):                # other configs, in the text below the table
                assert f"`{json.dumps(v['config'])[1:-1]}`) reached only {v['f1']:.2f}" in flat, key
        splink = r[ds] if ds.startswith("dedupe_") else load("splink_datasets_benchmark.json")[ds]
        assert f"{splink['splink_from_labels']['f1']:.2f}" in row, ds


def test_large_block_check_and_contact_junk_claims():
    """docs/guide.md "How it decides": the large-block check kept nothing and changed nothing on
    the person benchmarks, and lifted the patent file; the nameless fake-mobile scenario failed
    every draw before contact profiling read the other contact."""
    r = load("large_block_check.json")
    for ds in ("febrl3", "febrl4", "fake_1000", "synthetic", "au", "historical_50k"):
        assert r[ds]["on"]["blocks_kept"] == 0 and r[ds]["on"]["f1"] == r[ds]["off"]["f1"], ds
    off, on = r["patents"]["off"]["f1"], r["patents"]["on"]["f1"]
    flat = _flat(GUIDE)
    assert f"lifted clustered F1 from {off:.2f} to {on:.2f}" in flat
    assert f"{on:.2f} ({off:.2f} without the large-block check)" in flat
    assert f"from {off:.2f} to {on:.2f} clustered F1" in _flat((ROOT / "CHANGELOG.md").read_text())
    s = load("scenarios.json")
    assert s["default_before_contact_junk"]["scenarios"]["fake_mobile_many_emails"]["passed_pct"] == 0
    assert all(s[v]["scenarios"]["fake_mobile_many_emails"]["passed_pct"] == 100
               for v in ("default", "twin_guard", "active", "twin_guard_active"))


def test_patterned_number_claims():
    """docs/report/ "Scenarios": a bought gold number failed every draw while patterns were junk
    on sight, and passes every draw now."""
    s = load("scenarios.json")
    assert s["default_before_patterned_numbers"]["scenarios"]["gold_number"]["passed_pct"] == 0
    assert all(s[v]["scenarios"]["gold_number"]["passed_pct"] == 100
               for v in ("default", "twin_guard", "active", "twin_guard_active"))
    assert "With random labels it passed {:.0f}% of repeats in this build".format(
        s["default"]["scenarios"]["no_surname_email"]["passed_pct"]) in _flat(GUIDE)
    assert "Active learning lifted the background F1 to {:.2f}".format(s["active"]["background_f1"]) in _flat(GUIDE)


def test_suspect_record_claims():
    """docs/guide.md "Suspect records": the real-name exclusion, the zero false flags and the
    time come from suspect_research.json, and the module's list is the one derived there."""
    from custmatch import suspect
    r = load("suspect_research.json")
    flat = _flat(GUIDE)
    assert r["real_names_match_the_module"] and len(r["real_names"]) == len(suspect.REAL_NAMES)
    assert f"{len(r['real_names'])} of the list's {r['single_words']} single words" in flat
    assert all(v["flagged"] == 0 for v in r["ncvr"].values())
    assert all(v["flagged"] == 0 for v in r["held_out"].values())
    assert f"It costs {r['million']['seconds']:.1f} seconds at a million records" in flat
    assert f"{r['million']['seconds']:.1f} s at a million records" in (ROOT / "CHANGELOG.md").read_text()


def test_realtime_coherence_claims():
    """docs/report/ "The service": the real-time coherence check and the merge policies, from
    server_replay.json (five seeds)."""
    r = load("server_replay.json")
    on, off = r["mean_strong_coherence_0.5"], r["mean_strong_coherence_0"]
    design = _flat(REPORT)
    assert (f"real-time precision went from {off['realtime_precision']:.2f} to {on['realtime_precision']:.2f} "
            f"for recall {off['realtime_recall']:.2f} to {on['realtime_recall']:.2f}, F1 "
            f"{off['realtime_f1_vs_truth']:.2f} to {on['realtime_f1_vs_truth']:.2f}") in design
    seeds = {x["seed"] for x in r["runs"]}
    f1 = lambda c, s: next(x["realtime_f1_vs_truth"] for x in r["runs"]  # noqa: E731
                           if x["merge"] == "strong" and x["coherence"] == c and x["seed"] == s)
    better = sum(f1(0.5, s) > f1(0, s) for s in seeds)
    assert better >= 4 and on["realtime_f1_vs_truth"] > off["realtime_f1_vs_truth"], "the check no longer helps"
    assert f"better on {['no', 'one', 'two', 'three', 'four', 'all'][better]} of five seeds" in design
    assert f"the batch run {on['batch_f1_vs_truth']:.2f}" in design
    assert f"gave {r['mean_always_coherence_0.5']['realtime_f1_vs_truth']:.2f}" in design
    assert f"never merging {r['mean_never_coherence_0.5']['realtime_f1_vs_truth']:.2f}" in design
    assert f"with {on['agreement_f1']:.2f} agreement with the batch run" in design
    strict = r["mean_strong_coherence_0.51"]
    assert (f"lifted precision to {strict['realtime_precision']:.2f} but cut recall to "
            f"{strict['realtime_recall']:.2f} (F1 {strict['realtime_f1_vs_truth']:.2f})") in design
    assert strict["realtime_f1_vs_truth"] < on["realtime_f1_vs_truth"]


def test_readme_headline_tiles():
    """README "Results": the headline tiles and "Where it is stronger", from the results files."""
    readme = _flat((ROOT / "README.md").read_text())
    big = load("scale_large.json")["scale_10m_tier2"]
    assert big["result"]["records"] > 10_000_000
    assert f"{big['result']['full_run_seconds'] / 60:.0f} minutes on one 16-core machine, {big['result']['clustered_f1']:.2f} clustered F1" in readme
    assert f"{big['result']['clustered_f1']:.2f} at ten million" in readme
    h = load("head_to_head.json")
    m, s = h["copies_50"]["custmatch"], h["copies_50"]["splink_m_from_labels_capped"]
    assert f"{m['wall_seconds']:.0f} seconds on a laptop, {m['clustered']['f1']:.2f} clustered F1" in readme
    assert f"custmatch scores {m['clustered']['f1']:.2f} clustered F1 where Splink scores {s['clustered']['f1']:.2f}" in readme
    assert f"({s['splink_seconds']:.1f} seconds at a million records)" in readme
    small = h["copies_1"]
    assert (f"about as accurate on 20,134 records ({small['dedupe']['clustered']['f1']:.2f}), took "
            f"{small['dedupe']['dedupe_seconds']:.0f} seconds and {small['dedupe']['peak_memory_gb']:.1f} GB there against "
            f"custmatch's {small['custmatch']['wall_seconds']:.1f} seconds and {small['custmatch']['peak_memory_gb']:.2f} GB") in readme
    import numpy as np
    r = load("ensemble.json")
    diff = {(ds, lt): np.mean([v["f1"] for k, v in r[ds]["forest_catboost"].items() if k.startswith(lt)])
            - np.mean([v["f1"] for k, v in r[ds]["forest"].items() if k.startswith(lt)])
            for ds in r for lt in ("random", "active")}
    assert f"+{np.mean(list(diff.values())):.2f} F1 on average, +{diff[('au', 'random')]:.1f} on sparse records" in readme
    assert f"+{diff[('historical_50k', 'random')]:.1f} on dense genealogy records" in readme
    assert f"never more than {-min(diff.values()):.2f} behind" in readme


def test_steward_learning_claims():
    """docs/report/ "The service" and CHANGELOG.md: decisions that train the next run, from server_learning.json."""
    r = load("server_learning.json")
    learn, only = r["mean_f1_by_round_learn"], r["mean_f1_by_round_override_only"]
    assert learn[-1] > only[-1] > learn[0], "decisions no longer teach the model more than overrides alone"
    said = f"{learn[0]:.1f} -> {learn[-1]:.1f} clustered F1 over"
    assert said in _flat(REPORT) and said in _flat((ROOT / "CHANGELOG.md").read_text())
    assert f"against {only[-1]:.1f} when they only override their own pairs" in _flat(REPORT)


def test_readme_memory_claims():
    """README "Where it is stronger": memory at a million and ten million records."""
    readme = _flat((ROOT / "README.md").read_text())
    h = load("head_to_head.json")
    m, s = h["copies_50"]["custmatch"], h["copies_50"]["splink_m_from_labels_capped"]
    big = load("scale_large.json")["scale_10m_tier2"]
    per_million = big["peak_memory_gb"] / big["result"]["records"] * 1e6
    assert f"A million records peak at {m['peak_memory_gb']:.2f} GB on a laptop (Splink: {s['peak_memory_gb']:.2f} GB)" in readme
    assert f"at {big['peak_memory_gb']:.1f} GB, about {per_million:.1f} GB per million" in readme
    assert f"needed {h['copies_1']['dedupe']['peak_memory_gb']:.1f} GB for 20,134 records" in readme


def test_readme_public_benchmarks():
    """README "On other people's data": every figure is one chapter 9 reports (and tests above tie
    chapter 9 to the results files)."""
    import re
    readme = (ROOT / "README.md").read_text()
    table = readme[readme.index("**On other people's data.**"):readme.index("**At scale, on synthetic customer data.**")]
    rows = [l for l in table.splitlines() if l.startswith("| ") and not l.startswith(("| Dataset", "| ---"))]
    assert len(rows) == 8
    chapter = (DOCS / "report" / "09-against-other-tools.md").read_text()
    for row in rows:
        for number in re.findall(r"\b\d{2}\.\d{2}\b", row):
            assert number in chapter, (number, row)
    assert "within a quarter of a point of what its blocking allows" in readme
    assert 99.46 - 99.22 <= 0.25


def test_label_curve_claims():
    """docs/report/04-labels.md "How many labels": the tables and their conclusions, from label_curve.json."""
    r = load("label_curve.json")
    text = _flat((DOCS / "report" / "04-labels.md").read_text())
    sets = ("fake_1000", "febrl3", "febrl4", "historical_50k", "ncvr_20k", "au_customer")
    for d in sets:
        for c in ("random_100", "random_300", "random_1000", "active_200", "active_300", "active_1000"):
            assert f"{r[d][c]['mean']:.2f}" in text, (d, c)
        z = r[d]["zero_label"]
        for k in ("top_lambda", "weight_10", "weight_20", "best_in_hindsight"):
            assert f"{z[k]['f1']:.2f}" in text, (d, k)
    gap = {d: r[d]["active_1000"]["mean"] - r[d]["active_300"]["mean"] for d in sets}
    assert sum(g <= 1.6 for g in gap.values()) == 5 and gap["au_customer"] > 1.6
    beat = [d for d in sets if r[d]["active_300"]["mean"] > r[d]["random_1000"]["mean"]]
    assert len(beat) == 4 and set(sets) - set(beat) == {"febrl4", "historical_50k"}
    assert 0 < r["febrl4"]["random_1000"]["mean"] - r["febrl4"]["active_300"]["mean"] < 0.1
    assert 14.07 in r["ncvr_20k"]["random_100"]["f1_by_seed"] and 7.54 in r["febrl4"]["random_50"]["f1_by_seed"]
    assert None in r["au_customer"]["random_50"]["f1_by_seed"]
    assert round(r["ncvr_20k"]["active_300"]["mean"] - r["ncvr_20k"]["zero_label"]["best_in_hindsight"]["f1"]) == 21
    assert "trails 300 labels by 21 points on NCVR" in text


def test_local_llm_claims():
    """docs/report/04-labels.md "A local LLM": the table and its conclusions, from local_llm_labels.json."""
    r = load("local_llm_labels.json")
    text = _flat((DOCS / "report" / "04-labels.md").read_text())
    for m in ("qwen3:8b", "qwen3:14b"):
        for d in ("customer", "historical"):
            for k in ("random", "hard"):
                assert f"{r[m][d][k]['llm']['accuracy']:.1f}%" in text, (m, d, k)
                assert r[m][d][k]["unanswered"] == 0
    for d in ("customer", "historical"):
        for k in ("random", "hard"):
            assert f"{r['qwen3:8b'][d][k]['forest']['accuracy']:.1f}%" in text, (d, k)
    s, f = r["qwen3:8b"], lambda d, k: s[d][k]["llm"]["accuracy"] - s[d][k]["forest"]["accuracy"]
    assert f("customer", "random") == 0 and round(-f("historical", "random"), 1) == 6.4
    assert round(f("customer", "hard"), 1) == 7.8 and round(f("historical", "hard"), 1) == 13.6
    for d in ("customer", "historical"):
        c = s[d]["hard"]["llm_confident"]
        assert f"{c['accuracy']:.2f}%" in text and f"{c['share']:.0f}%" in text
        for k in ("random", "hard"):
            assert r["qwen3:14b"][d][k]["llm"]["accuracy"] < s[d][k]["llm"]["accuracy"]
    big = r["qwen3:14b"]["historical"]
    assert (round(big["random"]["llm"]["recall"]), round(big["hard"]["llm"]["recall"])) == (39, 24)
    assert min(big[k]["llm"]["precision"] for k in ("random", "hard")) > 90


def test_embedding_blocking_claims():
    """docs/report/06-blocking-and-scale.md "Nearest neighbours by spelling": both tables, from the results."""
    from statistics import median
    c, p = load("embedding_blocking.json"), load("embedding_pipeline.json")
    text = _flat((DOCS / "report" / "06-blocking-and-scale.md").read_text())
    rows = ("keys", "keys_lsh", "keys+char_names_k5", "keys+char_names_place_k5", "keys+char_names_place_k10")
    for d in ("historical_50k", "ncvr_20k"):
        for k in rows:
            assert f"{c[d][k]['ceiling']:.2f}" in text and f"{c[d][k]['pairs']:,}" in text, (d, k)
        lsh = [r["f1"] for r in p[d]["keys_lsh"]["runs"]]
        for k, v in p[d].items():
            f1 = [r["f1"] for r in v["runs"]]
            assert len(f1) == 10 and f"{v['mean_f1']:.2f} | {min(f1):.2f}-{max(f1):.2f}" in text, (d, k)
            if k.startswith("keys_emb"):
                wins = sum(a > b for a, b in zip(f1, lsh))
                assert f"{max(f1):.2f} | {wins} of 10" in text, (d, k)
        secs = c[d]["keys+char_names_place_k10"]["embedding_seconds"]
        assert f"{secs:.0f} s on {'historical_50k' if d.startswith('hist') else 'NCVR'}" in text
    assert f"{c['historical_50k']['keys_lsh+char_names_place_k5']['ceiling']:.2f}" in text
    gap = [a["f1"] - b["f1"] for a, b in zip(p["historical_50k"]["keys_emb10"]["runs"], p["historical_50k"]["keys_lsh"]["runs"])]
    assert f"by {fmean(gap):.2f} points on average" in text
    for k in ("keys_emb5", "keys_emb10"):
        n = [a["f1"] - b["f1"] for a, b in zip(p["ncvr_20k"][k]["runs"], p["ncvr_20k"]["keys_lsh"]["runs"])]
        assert min(n) > 1.0 and f"{fmean(n):.2f} {'on average with 5' if k == 'keys_emb5' else 'with 10'}" in text
    for d, want in (("historical_50k", "98.66 against 98.92 on historical_50k"), ("ncvr_20k", "100.00 on NCVR")):
        a, e = c[d]["keys+ann_names_place_k10"]["ceiling"], c[d]["keys+char_names_place_k10"]["ceiling"]
        assert e - a < 0.3 and f"{a:.2f}" in want and want in text
    h = p["historical_50k"]
    assert f"ties LSH there, {h['keys_ann10']['mean_f1']:.2f} against {h['keys_lsh']['mean_f1']:.2f}" in text
    assert abs(h["keys_ann10"]["mean_f1"] - h["keys_lsh"]["mean_f1"]) < 0.1
    sc = load("embedding_scale.json")
    for n in ("100000", "1000000", "5000000"):
        r = sc[n]
        row = (f"| {int(n):,} | {r['vectors_seconds']:.1f} s | {r['ann_seconds']:.1f} s | {r['ann_pairs']:,} | "
               f"{r['peak_gb_after_ann']:.2f} GB |")
        assert row in text, n
        if "keys_seconds" in r:
            assert f"| {r['keys_seconds']:.1f} s | {r['keys_lsh_seconds']:.1f} s | {r['keys_lsh_pairs'] / r['keys_pairs']:.1f} times |" in text
    per_m = [sc[n]["ann_seconds"] / int(n) * 1e6 / 60 for n in ("1000000", "2000000", "5000000")]
    assert all(4.3 < x < 4.8 for x in per_m)                      # "about four and a half minutes a million"
    million = max(load("custmatch_scale.json"), key=lambda r: r["records"])
    assert sc["1000000"]["ann_seconds"] > million["full_run_seconds"]   # "more than the rest of a run"
    stage = lambda k, f: median(f(r["stage_seconds"]) for r in p["ncvr_20k"][k]["runs"])  # noqa: E731
    blk = lambda r: next(v for n, v in r.items() if n.startswith("blocking"))  # noqa: E731
    run = lambda r: sum(r.values())  # noqa: E731
    assert f"{stage('keys_lsh', blk):.2f} s instead of {stage('keys', blk):.2f} s on NCVR" in text
    assert f"(labels in hand) {stage('keys_lsh', run):.1f} s instead of {stage('keys', run):.1f} s" in text
    assert f"neighbours, {stage('keys_emb10', run):.1f} s" in text


def test_auto_adaptive_and_stratified_label_claims():
    """docs/report/06-blocking-and-scale.md "Splitting when it is needed" and "Random labels that
    still hold matches", from ncvr_leipzig.json, beat_splink.json and home_data_benchmark.json."""
    r = load("ncvr_leipzig.json")
    text = _flat((DOCS / "report" / "06-blocking-and-scale.md").read_text())
    ad = "postcode_initial_keys_adaptive"
    plain, strat = r[f"{ad}/people_200000"], r[f"{ad}/people_200000/stratified"]
    one, one_s = r[f"{ad}/people_1000000"]["runs"][0], r[f"{ad}/people_1000000/stratified"]["runs"][0]
    k, k_s = r[f"{ad}/people_1000000/random_1000"]["runs"][0], r[f"{ad}/people_1000000/random_1000/stratified"]["runs"][0]
    assert f"| 200,000 (3 draws) | {plain['mean_f1']:.2f} | {strat['mean_f1']:.2f} |" in text
    assert (f"| 1,000,000, 300 random labels | {one['clustered']['f1']:.2f} (2 random matches) | "
            f"{one_s['clustered']['f1']:.2f} ({one_s['random_matches']}) |") in text
    assert (f"| 1,000,000, 1,000 random labels | {k['clustered']['f1']:.2f} ({k['random_matches']}) | "
            f"{k_s['clustered']['f1']:.2f} ({k_s['random_matches']}) |") in text
    assert strat["mean_f1"] > plain["mean_f1"] and one_s["clustered"]["f1"] > one["clustered"]["f1"] + 10
    assert k_s["clustered"]["f1"] >= k["clustered"]["f1"], "stratified labels now cost at a million people"
    # the true default switches to both by itself, and lands where the explicit settings do
    assert r["current_default/people_200000"]["mean_f1"] == strat["mean_f1"]
    b = load("beat_splink.json")
    # three label draws per dataset: mean and range, plain and stratified, random and active
    seeds = ["", "/seed2", "/seed3"]
    for ds in b["rich_stratified"]:
        cells = []
        for m in ("random_1000", "active_300_700"):
            for v in ("rich_rerun", "rich_stratified"):
                f = [b[v + s][ds][m] for s in seeds]
                lo, hi = f"{min(f):.2f}", f"{max(f):.2f}"
                cells.append(f"{fmean(f):.2f}" + ("" if lo == hi == "99.99" else f" ({lo}-{hi})"))
        assert f"| {ds} | " + " | ".join(cells) + " |" in text, ds
    hist = {v: fmean([b[v + s]["historical_50k"]["active_300_700"] for s in seeds]) for v in ("rich_rerun", "rich_stratified")}
    assert f"by {hist['rich_rerun'] - hist['rich_stratified']:.2f} on average" in text
    h = load("home_data_benchmark.json")
    assert (f"the childcare sites from 86.95 to {h['dedupe_csv_example']['custmatch_1000']['f1']:.2f}, the patent "
            f"applicants from 75.88 to {h['dedupe_patent_example']['custmatch_1000']['f1']:.2f}") in text


def test_ncvr_memory_claims():
    """docs/report/06 "Memory and time on one machine", the guide and the changelog: peak memory and time of
    one NCVR run before and after this release's changes, and with macOS's allocator cache off,
    from ncvr_memory.json."""
    r = load("ncvr_memory.json")
    text = _flat((DOCS / "report" / "06-blocking-and-scale.md").read_text())
    flat = _flat(GUIDE + (ROOT / "CHANGELOG.md").read_text())
    label = {"numpy/before": "Before", "numpy": "Now", "numpy/malloc_large_cache_0": "Now, `MallocLargeCache=0` (macOS)"}
    gb = lambda x: f"{x['peak_memory_gb']:.1f}".removesuffix(".0")      # noqa: E731
    for n in (1428005, 5000000):
        runs = {k: r[f"records_{n}/{k}"] for k in label}
        first = runs["numpy/before"]
        assert runs["numpy"]["malloc_large_cache"] is None and runs["numpy/malloc_large_cache_0"]["malloc_large_cache"] == "0"
        for k, x in runs.items():
            assert x["clustered"] == first["clustered"] and x["candidate_pairs"] == first["candidate_pairs"], (n, k)
            block = next(s for name, s in x["stage_seconds"].items() if name.startswith("blocking"))
            assert (f"| {label[k]} | {x['peak_memory_gb']:.1f} GB | {block:.0f} s | {x['seconds'] / 60:.1f} min |") in text, (n, k)
        off, on = runs["numpy/malloc_large_cache_0"], runs["numpy"]
        assert off["peak_memory_gb"] < on["peak_memory_gb"] < first["peak_memory_gb"], n
        # "the peak falls by a fifth to over a third, and the run takes 10 to 20% longer"
        assert 0.20 <= 1 - off["peak_memory_gb"] / on["peak_memory_gb"] <= 0.40, n
        assert 1.10 <= off["seconds"] / on["seconds"] <= 1.20, n
    full = {k: r[f"records_5000000/{k}"] for k in label}
    one = {k: r[f"records_1428005/{k}"] for k in label}
    assert (f"a run takes {full['numpy']['seconds'] / 60:.1f} minutes against {full['numpy/before']['seconds'] / 60:.1f}, "
            f"at a peak of {gb(full['numpy'])} GB against {gb(full['numpy/before'])} GB") in flat
    assert (f"lowers the peaks to {gb(full['numpy/malloc_large_cache_0'])} GB and "
            f"{gb(one['numpy/malloc_large_cache_0'])} GB") in flat
    assert (f"lowers those peaks to {gb(full['numpy/malloc_large_cache_0'])} GB and "
            f"{gb(one['numpy/malloc_large_cache_0'])} GB") in flat
    duck = r["records_5000000/duckdb"]
    assert duck["peak_memory_gb"] > on["peak_memory_gb"]
    assert f"peaked at {duck['peak_memory_gb']:.1f} GB on the full file" in text


def test_fold_trees_claims():
    """docs/report/06 "Fewer trees in the fold models": from fold_trees.json."""
    r = load("fold_trees.json")
    text = _flat((DOCS / "report" / "06-blocking-and-scale.md").read_text())
    runs = [b for ds, seeds in r.items() if ds != "ncvr_1m" for s in seeds.values() for b in s.values()]
    change = [b["100"]["f1"] - b["300"]["f1"] for b in runs]
    same = sum(abs(c) < 1e-9 for c in change)
    assert f"100 trees left {same} of the {len(runs)} runs' clustered F1 unchanged, the rest within " \
           f"{min(change):.2f} to +{max(change):.2f}" in text
    n = r["ncvr_1m"]["active_300_700"]
    assert (f"({n['300']['forest_seconds']:.1f} s to {n['100']['forest_seconds']:.1f} s) but moved the threshold "
            f"from {n['300']['threshold']:g} to {n['100']['threshold']:g} and cost "
            f"{n['300']['f1'] - n['100']['f1']:.2f} points ({n['300']['f1']:.2f} to {n['100']['f1']:.2f})") in text
