# 6. Blocking and scale

Blocking decides which pairs are ever compared; a pair it never proposes can never be matched.
This chapter covers what blocking costs and loses, why dense files score lower, and how far
one machine goes - from a laptop to ten million records.

## Blocking, and what Splink, dedupe and Zingg taught

The leading open-source matchers ([chapter 9](09-against-other-tools.md)) each do something custmatch's first
blocker did not. These are now options - each measured on all five benchmark datasets with rich
features (`research/scripts/beat_splink.py`), clustered F1 with 1,000 random labels and, in brackets,
300 random + 700 active:

| Dataset | Old blocking: pairs / F1 random (active) | `"blocking": "fixed"` (split) | `"blocking": "learned"`, 99% | Re-cut over 30 (`max_cluster`) | Learned + disagreement, active |
| --- | --- | --- | --- | --- | --- |
| fake_1000 | 50,129 / 92.46 (99.08) | 7,329 / 95.54 (98.17) | 1,683 / 93.60 (95.66) | 92.46 (99.08) | 95.66 |
| febrl3 | 65,001 / 99.42 (99.99) | 64,828 / 99.86 (99.99) | 6,563 / 99.38 (99.99) | 99.42 (99.99) | 99.99 |
| febrl4 | 61,269 / 99.84 (99.99) | 59,949 / 99.82 (99.99) | 4,823 / 98.06 (99.97) | 99.84 (99.99) | 99.94 |
| historical_50k | 547,050 / 87.02 (86.20) | 672,502 / 84.85 (86.20) | 315,711 / 83.76 (86.19) | 87.79 (86.69) | 87.69 |
| synthetic | 48,759 / 97.28 (98.61) | 48,884 / 97.36 (98.74) | 24,032 / 97.66 (97.64) | 97.28 (98.61) | 98.89 |

These ran before 0.1.2. Rerun on 0.1.2's code, historical_50k scores 85.53 (86.64) with the old
blocking and 84.95 (87.13) with `"fixed"` (`rich_rerun`, `rich_split_0_1_2`); the other four
datasets give the same figures as above.

- **`"blocking": "fixed"`** (Zingg): the same kind of keys as rules, but a block larger than
  `max_block` is split with the next key (first letters of the first name, birth year, street...)
  instead of dropped; records without that key (a two-letter first name) stay together as their
  own piece. It fixed fake_1000's 173-record London block without any hand tuning
  (95.54 against 92.46 with random labels; with active learning 98.17 against 99.08). It is
  level on the synthetic file and the febrl sets, and costs 2.17 on the dense historical_50k with random
  labels (0.58 on 0.1.2's code), where splitting keeps more pairs than dropping. Use it when large blocks are ordinary
  common values (a city), not shared contacts.
- **`"blocking": "learned"`** (dedupe): a greedy set cover picks the cheapest rules from a library
  (exact, first three letters, Soundex, year and year-month of birth, and pairs of these) that
  cover `blocking_recall` (default 0.95; 0.99 above) of your labelled matches. It generates 2 to
  30 times fewer candidate pairs, for F1 changes from +1.1 to -3.3 with random labels - a speed
  option for large files. Like dedupe it learns on a sample (labelled records plus 50,000 others), which at 1M
  records chose rules giving 679,400 pairs. The rules are learned from all your labelled matches, including the
  ones that later judge the threshold, so the cross-fitted estimate in `summary.json` is
  somewhat optimistic with learned blocking: matches the rules miss are, by construction, rare
  among your labels and less rare in the rest of the file.
- **`sample --active ... --strategy disagreement`** (dedupe): half of each active batch comes from
  plausible pairs blocking did *not* propose that the forest calls matches; with learned
  blocking, those labels teach the blocker what it misses. In this single draw it scored
  98.89 on the synthetic file and 87.69 on historical_50k (plain active learning
  98.61 and 86.20); before the blocker fixes it scored 79.45 on historical_50k.
  A result that flips between versions is not a conclusion either way.
- **`"max_cluster": 30`** (dedupe): a customer larger than this is re-clustered at stricter
  thresholds until the pieces fit. +0.77 on historical_50k and no change elsewhere, where no customer
  reaches 30 records. Every run also writes a per-record `confidence` in `clusters.csv`: the
  mean score of its links, or for a record left on its own, one minus its strongest link.
- **Diagnostics** (Splink): every run writes `blocking_report.json` - pairs and new pairs per
  rule, the largest blocks by key, blocks split or dropped - and `custmatch blocking CONFIG`
  prints it before you label anything. `summary.json` has `stage_seconds` for each stage.
- **Speed** (dedupe, Splink): the new blocker builds integer keys and pairs the records of each
  block in vectorised numpy steps (until 0.1.0 a DuckDB self-join, about 2 s faster at a million
  records, dropped to save the dependency). At 1M records blocking takes 7.3 s and a whole run
  with `"fixed"` 94 s, against 101 s with the default blocker (`research/results/custmatch_scale*.json`).

None of these is the default: each wins somewhere and loses somewhere. The default blocking is
the one the report measured.

**A match weight of custmatch's own** (`"match_weight_feature": true`, opt-in;
custmatch/matchweight.py). Until 0.1.0 an option ran Splink's unsupervised model and gave its
match weight to the forest. custmatch now computes such a weight itself, with no Splink: each
field of a pair gets a comparison level (missing, disagree, close, exact), u is counted on a
million random record pairs, m is fitted by expectation maximisation over all candidate pairs
without labels, and an exact agreement is weighed by how common the agreeing value is (term
frequency). The forest learns from your labels how far to trust it. Any pair can be scored, so
the in-cluster check, `custmatch add` and `explain` reuse the saved model.

Three lessons shaped it. EM fits both classes on the candidates - with the non-match class fixed
at random-pair rates, the match class took every candidate, because blocking makes non-matching
candidates agree far more often than random records do. Fields a person changes - surname,
address, email, phone, and extra fields such as postcode or city - count only when they agree,
as in Fellegi-Sunter practice for time-varying fields and the many-valued features of MDM
systems: the first version, which counted every disagreement, learned from generated files where
nobody moves that a different address or phone meant a different person, and the "moved house"
and "married name" scenarios failed in up to all of their draws. And a bare initial counts as no
evidence rather than as a rare name, which had merged sisters recorded as "L. Jones".

Forest alone against forest with the weight, on the same labels - three draws, random / active
labels (`research/scripts/modal_models.py --models forest,forest_match_weight`, `research/results/match_weight.json`):

| Dataset | Forest | + match weight |
| --- | --- | --- |
| febrl3 | 99.62 / 99.99 | 99.99 / 99.99 |
| febrl4 | 99.90 / 99.99 | 99.97 / 99.99 |
| fake_1000 | 92.66 / 99.05 | 93.71 / 99.03 |
| synthetic | 97.54 / 98.96 | 97.66 / 98.97 |
| Australian, sparse | 71.73 / 80.72 | 75.56 / 81.44 |
| historical_50k | 87.45 / 88.09 | 88.51 / 88.71 |

It gains most where labels are thin: +3.83 on the sparse Australian file and +1.06 on
historical_50k with random labels, and is at most 0.02 behind anywhere. But it costs a
behaviour the forest alone gets right in nearly every repeat (`research/scripts/scenarios.py`, three draws,
share of each scenario's 10 cases handled correctly):

| Scenario | Forest | + weight | Forest, active | + weight, active |
| --- | --- | --- | --- | --- |
| Moved house | 100% | 100% | 100% | 100% |
| A new surname after marriage | 100% | 100% | 100% | 100% |
| Sisters at one address, first initial only | 100% | 100% | 100% | 20 to 100% |
| Only a first name and email (a web sign-up) | 90 to 100% | 0 to 10% | 90 to 100% | 0 to 30% |

A record with only a first name and an email has few fields to agree on, so its total weight is
low however strong the email match, and the forest learns to trust the total. Giving it the
weight per field as well did not help. So the weight stays opt-in: use it when labels are few and
mostly random and your records are full; leave it off where web sign-ups with a first name and
an email are common.

## Dense files: why historical_50k scores lower

historical_50k is the hardest benchmark here: about ten records per person, half of them without
a surname or a birth date, names such as "tom baron chudleigh" or just "sir", and corrupted
dates. Two limits add up (`research/scripts/historical_blocking.py`, `research/scripts/historical_labels.py`):

- **Blocking.** With exact keys alone, a perfect matcher could reach only 93.34
  clustered F1: many people's records share no key at all (a blank surname, Tom against Thomas,
  a mistyped date). Similarity blocking on the names within a birth place
  (`"lsh_blocking": {"fields": ["first_name", "last_name"], "with": "birth_place"}`) lifts that
  ceiling to 96.81 for 13% more candidate pairs. The historical_50k benchmark
  config has used it since 0.1.0: over three label draws the forest scores 86.60 with
  1,000 random labels and 87.99 with active ones (`research/scripts/matcher_bakeoff.py`),
  against 85.68 and 87.02 with the exact keys alone under the same threshold rule.
- **The matcher.** More labels help only a little. With 1,000 / 3,000 / 6,000 random labels the
  forest scored 86.08 / 86.24 / 87.44 with the exact keys and
  87.91 / 89.17 / 88.53 with similarity blocking (two draws each), and
  even the best threshold in hindsight reached only 89.83. The remaining errors carry little
  evidence: wrongly joined pairs mostly share a surname and a birth place and nothing else, and
  many true pairs disagree on place or date.

Tried without gain: the GB country profile, for its nicknames (86.73 against
86.35); comparing whole names by word overlap; and accepting weaker links only where a third
record backs them up (worse: 84 to 86). On a dense file of your own where one person's records
often end up as several customers, try similarity blocking on the names, with a field that keeps
blocks small, and compare the cross-fitted F1 in `summary.json`. Labels cannot show what blocking
misses: they are drawn from the candidate pairs.

## Nearest neighbours by spelling: better than LSH on name-only data, at a cost

Similarity blocking finds pairs by chance agreement on hashed letter pairs. A direct alternative
is to give each record its nearest neighbours by spelling: character 2-4-gram TF-IDF vectors of
the names, alone or with the place (birth place, town), and each record's k most similar records
by cosine added to the exact keys' pairs. No model is downloaded. Two steps, on historical_50k
and the Leipzig NCVR file (20,000 people, default keys):

**The ceiling** (`research/scripts/embedding_blocking.py`, `research/results/embedding_blocking.json`):
clustered F1 if every true candidate pair were accepted and joined up.

| Blocking | historical_50k | Pairs | NCVR | Pairs |
| --- | ---: | ---: | ---: | ---: |
| Keys alone | 93.34 | 547,050 | 95.74 | 423,539 |
| Keys + LSH on the names within the place | 96.81 | 619,385 | 98.85 | 467,112 |
| Keys + 5 neighbours, names | 96.75 | 591,019 | 99.82 | 465,686 |
| Keys + 5 neighbours, names and place | 98.61 | 595,164 | 99.99 | 472,323 |
| Keys + 10 neighbours, names and place | 98.92 | 689,521 | 100.00 | 541,994 |

Names with the place do better than names alone, and on top of LSH the neighbours add little
(98.67 with 5 on historical_50k), so they would replace LSH rather than join it.

**The score** (`research/scripts/embedding_pipeline.py`, `research/results/embedding_pipeline.json`):
the whole pipeline, labels drawn and the forest trained over the same candidate pairs - 100
random and 900 active labels, answered from the truth, ten seeds. Mean clustered F1, the range
over seeds, and how many seeds beat LSH:

| Blocking | historical_50k | Range | Beats LSH | NCVR | Range | Beats LSH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Keys alone | 86.01 | 81.47-87.23 | - | 95.66 | 95.64-95.67 | - |
| Keys + LSH | 86.62 | 84.07-88.49 | - | 98.70 | 98.67-98.72 | - |
| Keys + 5 neighbours, names and place | 86.69 | 83.04-89.79 | 4 of 10 | **99.85** | 99.83-99.87 | 10 of 10 |
| Keys + 10 neighbours, names and place | **87.95** | 85.97-90.36 | 8 of 10 | 99.80 | 99.74-99.85 | 10 of 10 |
| The same 10 neighbours, found approximately | 86.59 | 84.69-88.91 | 6 of 10 | 99.81 | 99.75-99.85 | 10 of 10 |

On NCVR the higher ceiling becomes a higher score with no doubt: over a point above LSH on every
seed, 1.15 on average with 5 neighbours and 1.11 with 10. On historical_50k, 10 neighbours lead LSH by 1.32 points
on average and on 8 of 10 seeds, but that lead did not survive finding the neighbours
approximately, as a product must (below): the approximate search ties LSH there, 86.59 against
86.62. Many of historical_50k's records are equally close to many others, and which of them
become neighbours decided the lead. So the gain is shown on name-only data (NCVR), not on dense
data.

These runs found a bug: the same seed did not always give the same score. scikit-learn's
predict_proba adds the trees' votes from several threads in the order they finish, so a
probability could differ in its last bit between runs - enough to reorder tied pairs in active
learning, so different pairs were asked. custmatch now adds the trees in a fixed order
(`custmatch/models.py`, `p_match`), and the numbers above are reproducible.

**What it costs.** On these small files, little: LSH's own blocking takes 1.04 s instead of
0.26 s on NCVR, and the run (labels in hand) 10.1 s instead of 8.7 s, mostly from scoring the
extra pairs; with 10 neighbours, 10.2 s. Finding the neighbours is the expensive part. Comparing
every record with every other took 52 s on historical_50k and 26 s on NCVR, and grows with the
square of the records. An approximate search (pynndescent's NN-descent, a graph of 16 neighbours
with the nearest 10 kept) keeps almost all of the ceiling - 98.66 against 98.92 on historical_50k,
100.00 on NCVR - and grows in a straight line. On the Leipzig NCVR file
(`research/scripts/embedding_scale.py`, `research/results/embedding_scale.json`; a 10-core laptop):

| Records | Vectors | Approximate neighbours | Pairs they add | Peak memory | Keys | Keys + LSH | LSH's pairs against the keys' |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 2.0 s | 20.3 s | 819,651 | 1.07 GB | 0.8 s | 4.7 s | 1.6 times |
| 1,000,000 | 18.3 s | 270.2 s | 8,163,687 | 2.44 GB | 9.0 s | 71.4 s | 4.4 times |
| 5,000,000 | 113.9 s | 1357.9 s | 40,393,085 | 7.58 GB | - | - | - |

So the neighbours cost about four and a half minutes a million records, where the keys take
seconds - more than the rest of a run (see below) - but add at most ten pairs a record. LSH is
quicker to find, but on the real file its blocks grow with the file: at a million records it
more than quadruples the candidate pairs, and those pairs are what the forest spends its time
scoring. Neither is in custmatch's defaults. Embedding blocking is not in custmatch at all: its
gain is shown only on name-only data, it would add pynndescent (and numba) as a dependency, and
it would make a large run several minutes longer. Worth adding as an option for name-only files
if users with such files ask; until then, `lsh_blocking` is the option for messy names.

## Larger files

**Speed.** On a million records - the synthetic file copied 50 times with copy-specific names
and contacts, so blocks do not grow as a real file's would: a throughput test on partitioned
data - custmatch took 111 seconds with default settings and 100 with `"blocking": "fixed"`
(`research/results/custmatch_scale*.json`). In an earlier run, not saved to `research/results/`, Splink took about
28.5 minutes on the same file. Most of that gap is candidate pairs (about 24.7 million for
Splink against 2.4 million): the junk contact values are shared across all 50 copies, and
custmatch drops those hub blocks while Splink, as configured, did not - an unequal setup, so
compare per-pair speed, where the two were within about 1.5x. A faster language would not change this picture: the forest and clustering
already run in compiled code, and the time goes on how many pairs are compared.

**Larger files** (`research/scripts/modal_scale.py`, `research/results/scale_large.json`): the same copies, on a
cloud machine with 16 cores and 128 GiB, with 1,000 random labels, run before and after the memory
redesign (numpy blocking, chunked features and predictions). The same code ran up to 40% slower on
one cloud run than the other, in stages the redesign did not touch, so the times are a range:

| Records | Candidate pairs | Full run (two runs) | Peak memory, before / after the redesign | Households, golden records, outputs | Clustered F1 |
| --- | --- | --- | --- | --- | --- |
| 2.0 million | 4.9M | 6 min | 7.5 / 7.7 GB | 43% | 97.04 |
| 5.0 million | 12.2M | 16 to 23 min | 17.6 / 17.6 GB | 45% | 97.55 |
| 10.1 million | 24.4M | 37 to 57 min | 33.4 / 33.7 GB | 47% | 96.89 |

Ten million records fit on one machine, with the same accuracy. On these copies - about 2.4
candidate pairs per record - memory is dominated by the records and the working tables around them,
so the redesign, which cut the cost per candidate pair from 288 to 155 bytes on 7 million NCVR pairs
(14 pairs per record; `research/scripts/memory_profile.py`), left it about where it was: roughly 3.3 GB per
million records, so a 16 GB laptop tops out near four million. (The record table itself is about
200 MB per million: pandas 3 stores text as Arrow strings.) The last stage - households, golden
records and writing the outputs - is over half the run at ten million and grows faster than the
file, the place to optimise next.

**Faster stages, same results (Tier 1, 0.1.0).** Golden records for the default rule are chosen for
every customer at once; household addresses are cleaned once per distinct value and their fuzzy
street comparison runs only between streets that could merge (it grew with the square of the
file); small batches are scored tree by tree instead of through the forest's thread pool; feature
columns and contact weights are prepared once per run. Every output stayed identical. On a laptop
at a million records, three runs each: a median of 98 s before, 73 s after. On Modal at ten
million: 57 minutes before, 38 after, the forest stage 942 s to 433 s - measured on different days,
and Modal's machines vary: run side by side at two million, stages whose code did not change took
43% longer on one container than the other. Peak memory rose about 11% (7.6 to 8.5 GB at two
million, side by side), from caching converted columns during scoring.

A second round, also with identical outputs, rewrote the household address merging (each street
once, in numpy steps, instead of every similar pair - a thousand near-identical streets had half a
million), factorised blocking keys column by column, and assembled golden records and link
evidence column by column. Ten million records then took 31 minutes, the last stage 796 s (from
1612 s originally), with the same clustered F1 and peak memory back to 34.0 GB: releasing the
scoring caches before the last stage removed the rise.

**One real file of millions: the full NCVR benchmark** (`research/scripts/modal_ncvr_full.py`): all five
sources, 5,000,000 records of 3,500,840 people with real name and postcode frequencies,
labelled as a user would (300 random, then 700 by active learning). It ran on the same machine
and took 3.2 hours, most of it the seven active-learning rounds (21 to 22 minutes
each, every round rescoring all 69 million candidate pairs) - the final run
was 28 minutes, at 55 GB. It scored **84.53** clustered F1 (precision
90.2, recall 79.6), against 95.63 on 200,000 of the same people. Two
things that were fine on the sample broke at full size, and `summary.json` warned of both:

- **Blocks grow with the file.** The keys chosen on 200,000 people (postcode plus an initial)
  make blocks of a few dozen there; with 3.5 million people many exceed `max_block` and are
  skipped, so true pairs are never compared and recall falls.
- **Matches are rarer among random pairs.** Only 11 of the 300 random labels were matches, too
  few to judge the threshold (the warning asks for at least 30).

Tune blocking and the label count on the full file, not a sample. Two reruns after the memory
redesign:

| Full NCVR run | Clustered F1 | Precision / recall | Random matches among the labels | Peak memory |
| --- | --- | --- | --- | --- |
| As above, before the memory redesign | 84.53 | 90.2 / 79.6 | 11 | 55.1 GB |
| The same, after it | 84.33 | 89.8 / 79.5 | 13 | 18.6 GB |
| Oversized blocks split (`"blocking": "fixed"`), 1,000 random labels | 87.19 | 87.7 / 86.7 | 30 | 26.4 GB |
| Split on three fields (`"blocking": "adaptive"`), 1,000 random labels, on an 18-core 64 GB laptop | 88.45 | 86.7 / 90.2 | 11 | 33.6 GB |

The redesign cut peak memory by 66% at the same accuracy. Splitting blocks instead of
dropping them, with enough random matches to judge the threshold, won back 2.9 of the points lost
against the 200,000-people sample - recall rose from 79.5 to 86.7 - and no warning fired; the
rest of the gap is still open. (Both reruns' times are left out: they were slowed by a defect in the
new blocker, since fixed, that scanned every block entry for each of about 50,000 skipped blocks.
On the full file the fixed blocker takes about 5 to 8 minutes, against about 9 for the old one.)

**Where the rest of the gap is** (`research/scripts/ncvr_blocking_scale.py`,
`research/results/ncvr_blocking_scale.json`): blocking alone, with no labels or model, on samples
of the same file. The ceiling is every true candidate pair accepted and closed.

| People (records) | Default: pair completeness / ceiling F1 | `"fixed"` | `"adaptive"` | Candidate pairs, default / fixed / adaptive |
| --- | --- | --- | --- | --- |
| 200,000 (285,812) | 93.78 / 97.35 | 97.32 / 99.20 | 98.47 / 99.85 | 7.1M / 7.9M / 8.9M |
| 1,000,000 (1,428,005) | 86.42 / 93.13 | 95.22 / 98.13 | 97.80 / 99.56 | 22.4M / 34.8M / 46.9M |
| 3,500,840 (5,000,000) | 79.61 / 88.96 | 93.37 / 97.13 | 96.74 / 99.05 | 69.1M / 125.6M / 191.8M |

With the default blocker, recall at full size is the blocking ceiling: 79.5 against a pair
completeness of 79.6. With `"fixed"` it is not. The ceiling there is 97.1, but the run scored
87.2, so ten points went to the matcher and the labels. The pairs `"fixed"` still misses are
mostly ones where the field it split on has the typo. For example, a (postcode, first initial)
block is split on the first three letters of the first name, so it loses "thoams gravel"
against "thomas kravel". Splitting never used the file's own fields, town and postcode.
`"blocking": "adaptive"` splits each oversized block on up to three refinements at once, each
on a different field and the extra fields included, and keeps the pieces of all of them. A pair
is lost only if it disagrees on every one. At full size that lifts pair completeness from 93.4%
to 96.7% for 53% more pairs.

End to end, on 1,000,000 people
([chapter 9](09-against-other-tools.md), "A public benchmark"), adaptive splitting with 300
random labels scored 80.84. That number says nothing about the blocker: only 2 of its 300
random labels were matches, because extra candidate pairs make matches rarer among random
pairs, and the warning fired. With 1,000 random labels it scored 94.04, against 93.64 for
`"fixed"` and 90.29 for the default with 300. Precision, not recall, is now what limits it
(91.8). On the full five million (the last row of the table above) it scored **88.45**, the best
full-file result: recall rose from 86.7 to 90.2 and precision fell from 87.7 to 86.7. Only 11 of
its 1,000 random labels were matches and the warning fired; the 700 active labels held 458. The
blocking ceiling is 99.05, so about ten points are still lost after blocking, and precision is now
the limit there too. It ran on a laptop (18 cores, 64 GB) in 2.2 hours, the final run 22 minutes;
the process's peak footprint was 51 GB, against the 33.6 GB the run records. On the five small benchmarks
(`research/scripts/beat_splink.py`, `rich_adaptive` against `rich_split_0_1_2`) it is level or
ahead with active labels. With random labels it gains 2.11 on historical_50k and loses 0.32 to
0.75 on febrl3, fake_1000 and the synthetic file. It is therefore opt-in, for files whose
blocks outgrow `max_block`. Both rule blockers also got faster with identical output: a sort
in place of a scan per oversized block, and an integer sort in place of a pandas groupby.
`"fixed"` blocking at 200,000 people went from 9.7 to 6.3 seconds, and `"adaptive"` on the full
file from 555 to 274 seconds at a 26 GB peak.

**Splitting when it is needed, by default.** The default blocker now switches to `"adaptive"`
by itself when its keys would leave more than 1% of the records only in skipped blocks
(`blocking.AUTO_ADAPTIVE_SHARE`), and `summary.json` says so. Measured with the default keys:
at most 0.45% on the five small benchmarks (synthetic; febrl3 0.06%, historical_50k 0.02%,
febrl4 and fake_1000 none) and 0.13% on BPID, so none of them changes; 1.1% on NCVR at 200,000
people, 4.2% at a million and 8% at five million. dedupe's two organisation files switch too,
and gain: the childcare sites from 86.95 to 89.08, the patent applicants from 75.88 to 77.37.
After a switch the large-block check still runs on the blocks the default keys skipped, so a
large organisation stays one customer; `add` and a run reusing the model block the same way.
`"blocking": "core"` keeps the old behaviour.

**Random labels that still hold matches** (`labels.stratified_pick`). With every split block
compared, matches are rarer among the candidate pairs, and a plain random draw holds a handful:
2 of 300 at a million people, 11 of 1,000 on the full file. The threshold's recall is judged on
those. From 2 million candidate pairs up (`labels.STRATIFY_FROM`), `custmatch sample` now draws
half the random labels uniformly and half spread over the pairs' agreement patterns - which of
the fields agree exactly - each pattern's share the square root of its size, and writes each
label's weight, the candidate pairs it stands for. The threshold's estimates are weighted by
it, so they stay unbiased. A first version stratified on how many fields agree, leaning towards
more; it cost 2.4 points at a million people with 1,000 random labels, because there pairs
agreeing on three of four fields were 0.01% matches (relatives sharing surname, town and
postcode) and pairs agreeing on two 1.25%. With `"adaptive"` blocking and 300 random + 700
active labels:

| NCVR people | Plain random draw | Stratified by agreement pattern |
| --- | --- | --- |
| 200,000 (3 draws) | 97.63 | 97.98 |
| 1,000,000, 300 random labels | 80.84 (2 random matches) | 94.85 (8) |
| 1,000,000, 1,000 random labels | 94.04 (21) | 94.91 (40) |

On the five small benchmarks it helped with 1,000 random labels on one draw each (febrl3 99.42
to 99.92, febrl4 99.84 to 99.94, historical_50k 85.53 to 88.04, fake_1000 92.46 to 93.60,
synthetic 97.28 to 97.78; `rich_stratified` against `rich_rerun`) and was level with active
labels (historical_50k 86.64 to 86.13, synthetic 98.61 to 99.11, the rest within 0.05). Those
files have under a million candidate pairs, so by default they still draw plainly until more
draws confirm it; `custmatch sample --stratified` draws this way at any size.

## What had to change to run this on a laptop, and what it costs

| Deviation | Reason | Effect on comparability |
|---|---|---|
| Ditto `max_len` 128 instead of the paper's 256 | At 256 a training step takes **76 s** on this machine (it exceeds RAM and swaps); at 128 it takes **0.9 s** | Truncates the longest records. Mainly affects Abt-Buy (long descriptions) and Company; the structured sets serialise well under 128 tokens |
| Ditto eval batch `batch_size*2` instead of `*16` | The 512-row eval batch was the single biggest memory spike | None — inference only |
| `apex` fp16 removed, device `mps`, `AdamW` from `torch.optim` | apex is CUDA-only | fp32 instead of fp16; if anything slightly more accurate, and slower |
| spaCy `en_core_web_lg` freed after preprocessing | It stayed resident (~1 GB) through training | None |
| One run per dataset (`run_id 0`) instead of the paper's averaged repeats | Time | Single-run variance: treat single runs as ±2 F1 ([chapter 10](10-reproductions.md)); on Beer and iTunes one flipped prediction moves F1 3–7 points |
| Local Llama-3.1-8B-Instruct (4-bit) for the first LLM pass | The `OPENAI_API_KEY` in the environment returned 401 | Different model class; compared against published 8B numbers. GPT-4o and GPT-4o-mini were later rerun directly ([chapter 10](10-reproductions.md)); GPT-4 remains cited |
| GPT models via OpenRouter, served by Azure, not OpenAI's API | The only key available; the account requires zero data retention, which excludes OpenAI's own endpoint | Same pinned snapshots; results land within 1.5 F1 of published on all ten runs |
| GPT-6 Luna and Jev Router without `temperature`, 2,000-token answer cap (8,000 for re-asks) | Reasoning models reject `temperature`; their thinking counts against the cap | Not bit-reproducible; Jev's model choice can change over time. Cut-off replies were re-asked, 10 of 128 still empty and scored "No" |

**Machine note.** Throughput here is dominated by memory, not by the GPU. With Docker
Desktop's VM holding 10 GB of the 16 GB, any job of mine above ~4 GB pushes the machine into
swap and everything slows by 10–60×. Two of my own jobs running at once was enough to take a
13.5-second Ditto epoch to ten minutes. If you want to rerun this, run one job at a time, or
stop Docker first. Two specifics worth knowing if you hit the same wall:

- PyTorch's MPS caching allocator grew to ~11 GB of driver memory while only ~2 GB of tensors
  were live. `torch.mps.empty_cache()` per epoch keeps it bounded.
- Ditto pads with token id 0, which is `<s>` for RoBERTa, and passes no attention mask. So any
  change that increases padding actively hurts accuracy: bucketing the padded length to a
  multiple of 32 cost ~30 points of validation F1 on Beer before I reverted it.
- **The `--dk` knowledge injector caches its output as `<file>.dk` and reuses it whenever the
  file exists and is non-empty — including a half-written one from an interrupted run.** This
  silently cost me a Walmart-Amazon run: the cached "training set" held 203 of 6,144 pairs, the
  run finished in 16 minutes instead of two hours and scored 28.5 F1, and nothing in the output
  said anything was wrong. I patched `transform_file` to treat a cache as valid only when its
  line count matches the input. **If you kill a Ditto run during preprocessing, delete the
  `.dk` files before restarting**, or check them:
  `for f in data/er_magellan/*/*/*.txt; do [ -f "$f.dk" ] && echo "$(wc -l <"$f") $(wc -l <"$f.dk") $f"; done`

---

[← Customer data: fake contacts, households and sparse records](05-customer-data.md) · [Contents](README.md) · [Golden records →](07-golden-records.md)
