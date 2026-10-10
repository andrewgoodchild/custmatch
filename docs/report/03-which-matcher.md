# 3. Which matcher

Random forests, gradient boosting, transformers (Ditto) and tabular foundation models, on the
same pairs. The forest is custmatch's default: cheap, robust with a thousand labels, and on
customer-style records within a point of a transformer. Zero-shot language models as matchers
are in [chapter 10](10-reproductions.md).

## Person matching: FEBRL4 and historical_50k

`research/scripts/febrl_person_matching.py`, `research/scripts/historical50k_dedupe.py`. No published F1 exists
for these two under a common protocol (the survey found no 2023–26 paper reporting
transformer or LLM F1 on FEBRL or NCVR), so these are new baselines rather than reproductions.
Scoring is pairwise P/R/F1 against **all** true pairs, so blocking misses count against recall.

### FEBRL4 (5,000 × 5,000 records, 5,000 true links)

| Method | P | R | F1 |
|---|---|---|---|
| Deterministic rules (exact SSN-id, or name+DOB) | 100.0 | 95.6 | 97.7 |
| recordlinkage ECM (unsupervised Fellegi–Sunter) | 100.0 | 100.0 | **99.99** |
| Splink 4 (unsupervised EM, fuzzy levels) | 100.0 | 100.0 | **99.99** |
| Random forest trained on FEBRL3, tested on FEBRL4 | 99.8 | 100.0 | 99.9 |

**FEBRL4 is saturated.** Everything scores ≈100, so it cannot rank methods. It is fine as a
smoke test for a pipeline, useless as a benchmark. Use it to check that your blocking and
comparison setup is wired correctly, then move on.

### historical_50k (50,578 records, 5,156 people, 303,961 true pairs)

Nine blocking passes (name pairs, surname+DOB, postcode, birth place+surname, DOB+birth place,
surname+occupation, first-3-letter name prefixes), skipping blocks larger than 150 records.
Blocking retrieves 918,609 candidate pairs and covers **81.7%** of true pairs, which is the
recall ceiling for every matcher below.

| Method | P | R | F1 |
|---|---|---|---|
| Deterministic rules (name+DOB, full name+DOB, name+postcode) | 84.7 | 20.7 | 33.2 |
| Splink 4, unsupervised Fellegi–Sunter, threshold 0.5 | 98.9 | 43.6 | **60.6** |
| Random forest on similarity features, held-out half of the clusters | 83.3 | 76.1 | **79.6** |
| — the same RF, scored only against pairs blocking retrieved | 83.3 | 93.0 | 87.9 |

This is the benchmark to use for customer matching, not FEBRL. It separates the methods
cleanly, and the ordering is the one practitioners should expect: rules are precise and miss
most of the duplicates; unsupervised Fellegi–Sunter roughly doubles recall at similar
precision; supervised learning on the same features adds another 19 F1 — but needs labels.

Note what the numbers are really saying: Splink's precision is 98.9 while its recall is 43.6,
with this report's own configuration - a fixed 0.5 threshold and seven blocking rules. Run with
its documented settings ([chapter 9](09-against-other-tools.md), "On Splink's own datasets") Splink scores 82.79
clustered F1 on this dataset; the numbers here understate it. Its loss here looks like blocking
and threshold rather than scoring, though Splink's own blocking ceiling was not measured. The last row makes the point
bluntly — the same random-forest predictions score 79.6 F1 against all true pairs but 87.9
against the pairs blocking actually retrieved. Before tuning a matcher on data like this,
measure the recall ceiling of your blocking; otherwise you are optimising the wrong half of
the pipeline. Widening the blocking from six passes to nine moved every method up (Splink
57.8 → 60.6, RF 77.0 → 79.6) without touching the matcher at all.

## Which classifier

The random forest (scikit-learn, 300 trees) is custmatch's default pairwise classifier; the
one alternative is the forest averaged with CatBoost (below). Until 0.1.0
`"matcher_model"` also offered histogram gradient boosting (`"hgb"`), CatBoost, TabPFN v2 (a
tabular foundation model that needs torch and a GPU) and `"auto"` (fit several, keep the best
cross-fitted estimate). A review on six datasets removed them: each model trained on the same
label files - per dataset three draws of 1,000 random labels and of 300 random + 700 active ones
- with the forest as the reference, run in parallel on Modal (`research/scripts/modal_models.py`,
`research/results/model_review.json`). Lighter alternatives were tried in the same run: ExtraTrees,
LightGBM and XGBoost, and TabICL, an open tabular foundation model like TabPFN. The review ran
while the match weight was on by default (since made opt-in), with the current threshold rule;
TabPFN with its default four ensemble members. Clustered F1, random / active labels:

| Dataset | Forest | hgb | ExtraTrees | LightGBM | XGBoost | CatBoost | TabPFN v2 | TabICL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| febrl3 | 99.99 / 99.99 | 99.77 / 99.94 | 99.90 / 99.99 | 98.94 / 99.94 | 99.63 / 99.93 | 99.60 / 99.93 | 99.92 / 99.99 | 99.94 / 99.99 |
| febrl4 | 99.98 / 99.99 | 99.83 / 99.99 | 99.97 / 99.99 | 99.35 / 99.99 | 99.46 / 99.99 | 99.81 / 99.99 | 99.99 / 99.99 | 99.98 / 99.99 |
| fake_1000 | 93.20 / 99.03 | 90.89 / 97.71 | 92.02 / 98.48 | 69.53 / 98.15 | 89.77 / 98.19 | 90.70 / 98.80 | 95.63 / 99.06 | 96.39 / 99.01 |
| synthetic | 97.50 / 98.87 | 97.11 / 98.73 | 97.54 / 98.70 | 97.24 / 98.66 | 96.85 / 98.75 | 97.58 / 98.72 | 97.58 / 98.72 | 97.85 / 98.94 |
| Australian, sparse | 77.72 / 82.03 | 69.97 / 80.88 | 67.65 / 79.36 | 69.79 / 80.95 | 77.10 / 79.12 | 76.66 / 82.14 | 78.74 / 82.17 | 79.46 / 80.42 |
| historical_50k | 89.11 / 88.52 | 77.05 / 85.97 | 88.22 / 86.63 | 75.21 / 86.26 | 87.40 / 89.66 | 90.29 / 90.63 | 90.55 / 89.88 | 91.33 / 90.81 |
| Mean / worst against the forest | | -2.34 / -12.06 | -1.46 / -10.07 | -4.33 / -23.67 | -0.84 / -3.43 | -0.09 / -2.50 | +0.52 / -0.15 | +0.68 / -1.60 |
| Seconds per run (mean) | 23 | 24 | 19 | 16 | 12 | 14 | 161 (GPU) | 267 (GPU) |

None of the lighter models matches the forest: hgb, ExtraTrees and LightGBM fall far behind with
random labels on the dense and sparse files, and XGBoost is behind on average. CatBoost wins only
on historical_50k and loses on fake_1000 and the sparse file. The two foundation models are the
only consistent gains - TabPFN +0.52 on average and never more than 0.15 behind, most with
random labels (fake_1000 +2.42, historical_50k +1.44) - but half a point on average does not pay for
torch, a GPU and runs 7 to 12 times as long; TabICL gains a little more on average and loses
1.60 on the sparse file with active labels. An earlier bake-off on a laptop
(`research/scripts/matcher_bakeoff.py`, the threshold rule before 0.1.0, TabPFN with one ensemble member)
reached the same picture: no model won everywhere. The forest stays the default; a config naming
a removed model is refused with a message saying so.

**The forest and CatBoost, averaged** (`"matcher_model": "forest_catboost"`, opt-in). Bagged
trees and boosted trees make different mistakes, so their average can beat both. The same review
setup - six datasets, the same label files, three draws of each kind, on Modal
(`research/scripts/modal_models.py --models forest,catboost,forest_catboost --out research/results/ensemble.json`)
- with the forest and CatBoost rerun alongside, on the current code. Clustered F1, random /
active labels:

| Dataset | Forest | CatBoost | Forest + CatBoost |
| --- | --- | --- | --- |
| febrl3 | 99.62 / 99.99 | 99.60 / 99.99 | 99.59 / 99.99 |
| febrl4 | 99.90 / 99.99 | 99.94 / 99.99 | 99.91 / 99.99 |
| fake_1000 | 92.66 / 99.05 | 91.54 / 99.03 | 92.97 / 99.03 |
| synthetic | 97.54 / 98.96 | 97.49 / 98.54 | 97.61 / 98.93 |
| Australian, sparse | 71.73 / 81.49 | 71.62 / 81.83 | 73.53 / 81.65 |
| historical_50k | 87.45 / 88.10 | 88.60 / 89.55 | 88.74 / 89.45 |
| Mean / worst against the forest | | +0.10 / -1.12 | +0.41 / -0.03 |
| Seconds per run (mean) | 18 | | 23 |

The average gains +0.41 clustered F1 on average and is never more than 0.03 behind the forest
on a dataset's mean (single draws go to -0.26 on febrl3 and -0.76 on the sparse file); it
gained on historical_50k in all six draws (+0.96 to +1.71). The sparse file's gain varies
between draws (-0.76 to +3.54), so its size is less certain than its sign. The active labels
were chosen by the forest, which if anything favours the forest. It stays opt-in because it
costs a 100 MB dependency and about 30% more run time, it has not yet been timed at a million
records, and the service's per-feature score breakdown covers the forest only.

---

[← Match whole customers, not pairs](02-whole-customers.md) · [Contents](README.md) · [Labels →](04-labels.md)
