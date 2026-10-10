# 9. Against other tools

custmatch against Splink, dedupe and others, on the same records and labels - including each
tool's own datasets - and on a public benchmark of real voter names.

## Compared with other open-source tools

A feature comparison as of 2026-09-26; Splink and dedupe were also run, below. Zingg was not.
custmatch is the only one of these that tunes its threshold on the clustered result, and the
only one with built-in household and junk-contact handling. Since this comparison it has gained
active learning, per-link explanations, incremental matching and a million-record run; it still
lacks a no-labels start and distributed scale.

**At a million records, against Splink 5** (`research/scripts/head_to_head.py --copies 50`,
`research/results/head_to_head.json`). The synthetic customer file copied 50 times (1,006,700 records),
one laptop (Apple silicon, 16 GB), one tool at a time, the same 1,000 labels. Splink 5.0.0 on
DuckDB, configured as its documentation recommends: blocking rules mirroring custmatch's keys,
name and email comparisons with term-frequency adjustment, u by random sampling, m by EM or
from half the labels, the threshold chosen on the labels, Splink's own clustering. Splink's time
is its own training, prediction and clustering, not loading or writing results:

| 1,006,700 records | Seconds | Peak memory | Candidate pairs | Precision | Recall | Clustered F1 |
| --- | --- | --- | --- | --- | --- | --- |
| custmatch (the whole run: also households, golden records, outputs) | 71.0 | 3.64 GB | 2.4 M | 98.22 | 97.75 | **97.99** |
| Splink, blocks over 60 records skipped, m from labels | **21.6** | 5.96 GB | 14.5 M | 79.75 | 78.03 | 78.88 |
| Splink, blocks over 60 records skipped, EM | 37.0 | 5.70 GB | 14.5 M | 75.28 | 71.16 | 73.16 |
| Splink, blocking as above with no cap, m from labels | 503.7 | 5.59 GB | 24.7 M | 66.47 | 78.03 | 71.79 |
| Splink, blocking as above with no cap, EM | 531.5 | 5.33 GB | 24.8 M | 75.28 | 71.16 | 73.16 |

Without a cap, placeholder contacts (`a@a.com`, `9999999999`, each on about 20,000 records)
give Splink's email and phone rules 2.4 billion comparisons, and DuckDB spills to disk. Skipping
those blocks - what checking Splink's `n_largest_blocks` suggests - makes it the faster tool:
22 seconds against custmatch's 38 for loading, blocking and the forest (the rest of custmatch's
71 is households, golden records and outputs, which Splink does not produce). custmatch is 19
points more accurate, on both precision and recall: Splink's model cannot discount shared
household values and placeholders, which this file is full of. At 20,134 records (one copy) the
gap was 13 points (97.38 against 84.19) and the two took 5.4 and 4.4 seconds.

**On each other's home data, with dedupe** (`research/scripts/home_data_benchmark.py`,
`research/results/home_data_benchmark.json`). Splink's four bundled datasets and dedupe's two examples
(github.com/dedupeio/dedupe-examples). All three tools learn from the same labelled pairs
(custmatch's sampler, seed 1, answered from the truth). dedupe 3.0.3 trains on them through its
training file - not its interactive console - and clusters at its documented threshold, 0.5; it
uses its examples' settings on its own data and fields mirroring Splink's comparisons on
Splink's. It samples at random without a seed, so it ran three times: the mean, and the range
where runs differed. Splink uses its documented settings on its own data
(`research/scripts/splink_datasets_benchmark.py`, the threshold from the labels) and plain string
comparisons on dedupe's, which document none. custmatch uses its defaults. Clustered F1:

| Dataset | Records | custmatch | Splink | dedupe (3 runs) |
| --- | --- | --- | --- | --- |
| fake_1000 (Splink's) | 1,000 | 87.11 | 58.36; **92.52** with its tutorial's looser blocking | 70.26 (70.12-70.53) |
| febrl3 (Splink's) | 5,000 | 99.42 | **99.63** | 98.11 |
| febrl4 (Splink's) | 10,000 | **99.84** | 99.83 | 99.11 |
| historical_50k (Splink's) | 50,578 | **87.02** | 76.82; 82.79 at its documented threshold | 80.02 (77.51-81.43) |
| csv_example (dedupe's: childcare sites) | 3,337 | 86.95 | 64.59 | **91.37** (86.72-93.70) |
| patent_example (dedupe's: patent applicants) | 2,379 | 75.88 (6.93 without the large-block check); **87.26** with `"max_block": 300` | 80.19 | 49.66 |

dedupe's two examples hold organisations, not people: custmatch reads the organisation's name as
a surname, with no first name. On the patent file one applicant has 676 records, and custmatch
skips any block of more than 60 records (`max_block`) as a likely hub: before the large-block
check ("How it decides") only 3.3% of the true pairs were ever compared and it scored 6.93. The
check, on by default, sees that the oversized company-name blocks are one entity each and
compares them: 75.88. Its run also warned about the blocks it still skipped ("... raise
"max_block" if these values are legitimately common"); following that, `"max_block": 300`
scored 87.26, ahead of Splink and dedupe. Splitting the oversized blocks instead (`"blocking": "fixed"`) reached only 14.12: this
file has no second key to split them by. Splink compares every pair in its blocks, whatever
their size. dedupe wins on its childcare sites.
On people, custmatch leads on the noisy and dense files and the three are close on the febrl
sets. dedupe's errors are missed matches - its precision was 96.9 to 100 everywhere, its recall
as low as 33.0 at the default threshold.

dedupe is the closest rival on accuracy and much the slowest: on the synthetic customer file at
20,134 records it scored **97.75** against custmatch's 97.38, in 499 seconds and 8.5 GB
against 5.4 seconds and 0.35 GB, and its peak memory was 4.0 to 9.2 GB on the six datasets of
1,000 to 50,578 records. A million records was not attempted on a 16 GB laptop.

Caveats. The synthetic file was generated for this project, so its difficulties - placeholders,
households, sparse records - are the ones custmatch was designed around. An expert could
configure Splink or dedupe better than these documented setups, and the configurations are in
the scripts for anyone to improve. custmatch and Splink ran once per dataset, on one label draw.

**Benchmarked against Splink** on the same synthetic file, scored identically
(`research/scripts/splink_vs_custmatch.py`), clustered F1:

| | Clustered F1 |
| --- | --- |
| Splink, no labels, at p > 0.9 / p > 0.99 | 1.41 / 29.76 |
| Splink, no labels, best threshold (picked on the truth) | 83.91 |
| Splink, threshold chosen on the same 1,000 labels | 83.03 |
| Splink, m-probabilities from 500 labels | 84.43 |
| custmatch, 1,000 random labels | **97.28** |

Splink runs in 7 seconds but cannot separate household members: same surname, same address
and a shared family email outweigh a different first name, and unsupervised training learns
households rather than people. Its match probabilities also pile up at 1.0; thresholds must be
set on match weight (the best was about 20, probability 0.999999).

| Tool | Licence | Matching | Threshold tuned on | Labels | Scale |
| --- | --- | --- | --- | --- | --- |
| custmatch | Apache 2.0 | forest with rich features; optional match-weight feature and forest + CatBoost ensemble | clustered result | random or active-learning batches | one machine, 1M records in under 2 min |
| [Splink](https://github.com/moj-analytical-services/splink) | MIT | Fellegi-Sunter (unsupervised) | pairs | none | DuckDB, Spark |
| [Zingg](https://github.com/zinggAI/zingg) | AGPL-3.0 + paid tiers | supervised | not documented | active learning | Spark, Snowflake |
| [dedupe](https://github.com/dedupeio/dedupe) | MIT | logistic regression | user-set | active learning | one machine |
| [recordlinkage](https://github.com/J535D165/recordlinkage) | BSD-3 | EM or supervised | not found | optional | small to medium |
| [pyJedAI](https://github.com/AI-team-UoA/pyJedAI) | Apache-2.0 | unsupervised or supervised, embeddings | not found | optional | one machine |
| [fastLink](https://github.com/kosukeimai/fastLink) (R) | GPL-3 | Fellegi-Sunter | not found | none | one machine |

dedupe here is the open-source library (3.0.3, released August 2024; its repository was last
pushed in July 2025). Dedupe.io, the hosted service built on it, shut down on 31 January 2023.

**On Splink's own datasets, with Splink's own settings** (`research/scripts/splink_datasets_benchmark.py`).
Splink runs its official v4.0.17 example notebooks verbatim; custmatch gets the same fields and
Splink's blocking rules through its config. Clustered F1 against each dataset's ground truth:

| Dataset | Splink, documented | Splink, best threshold (picked on truth) | custmatch, 1,000 random labels | custmatch, 300 random + 700 active |
| --- | --- | --- | --- | --- |
| febrl3 (5k, dedupe) | no threshold documented | 99.79 | 99.92 | **99.99** |
| febrl4 (5k + 5k, link) | 99.58 | 99.95 | 99.97 | **99.99** |
| historical_50k (50k, dense) | **82.79** | 82.87 | 82.08 | 78.00 |
| fake_1000 (1k, very noisy), tutorial's loose blocking | 92.52 | 92.52 | 88.19 | **94.45** |

That table is custmatch before its two matcher options. custmatch's fake_1000 row uses a config
tailored as Splink's tutorial is: the same loose blocking, `max_block` 200 and fuzzy date of
birth and email. With its default config it scored 76.00 / 78.38, and Splink with the
tutorial's first, tighter blocking scored 58.11.

**With rich features and the Splink feature** (removed in 0.1.0; `research/scripts/beat_splink.py`, same labels and seeds;
random / active labels):

| Dataset | Splink | custmatch, rich + Splink feature, 1,000 random | custmatch, rich + Splink feature, 300 + 700 active |
| --- | --- | --- | --- |
| febrl3 | 99.79 (best threshold) | **99.99** | 99.95 |
| febrl4 | 99.58 | **99.99** | **99.99** |
| historical_50k | 82.79 | 86.57 | **87.19** |
| fake_1000 | 92.52 | 93.89 | **98.35** |
| synthetic customers | 84.43 | 96.77 | **98.86** |

In this single label draw, with 1,000 random labels, custmatch scored at or above Splink on
every dataset, by 3.8 points on the dense
historical_50k; active learning added more everywhere except the dense file and febrl4. Over
three draws (`research/scripts/matcher_bakeoff.py`, rich features, no Splink feature; the threshold rule
before 0.1.0) the forest averaged
99.50 / 99.83 / 93.36 / 97.54 / 86.29 on febrl3 / febrl4 / fake_1000 / synthetic /
historical_50k: ahead of Splink on four, 0.29 behind on febrl3. Two caveats: Splink here uses
no labels and custmatch 1,000, and the rich features were developed on these same datasets, so
none of them is a held-out test of the design. Rich features alone (the default) score 99.42 / 99.84 / 87.02 / 92.46 / 97.28 on
the same five. Two ideas did not work: drawing half of each active batch at random did not fix
dense files, and a near-zero-label route (100 hand labels gating 900 LLM labels from Jev) scored
97.21 / 75.26 / 68.15 / 96.93 on febrl4 / historical_50k / fake_1000 / synthetic (before the
sparse-junk fix; not rerun, as it costs LLM calls) - behind Splink on
three, because a 100-pair check is too small to catch a labeller that misses matches. Splink
still needs no labels at all; custmatch needs about 1,000 to beat it.

Senzing is a commercial product rather than open source; check its current licence terms for
evaluation limits. Ditto and Magellan are research code without recent releases.

Gaps closed: the Splink benchmark, active learning, explanations, incremental matching, a
million records on one machine (contact profiling was vectorised: 7 s to 0.1 s at 20k), and
beating Splink on dense data (rich features, then the Splink feature, now custmatch's own match weight).

Gaps still open:

- **Starting with no labels.** Seeding custmatch with Splink's 2,000 most confident calls gave
  labels that were 99.85% correct and a forest that scored 19.93: the seeds are only the easy
  extremes, never the hard household cases. Labels must be representative, not just right.
  Use 300 hand labels plus active learning, or gated LLM labels.
- **Distributed scale.** At 1M records blocking takes 14 s of 101 (default settings). Ten million
  records ran on one 128 GiB machine in 37 minutes
  (33 GB at peak); beyond what fits in one machine's memory - roughly 30
  million records at that rate - a distributed backend would be needed.
- **Active learning on dense files** is inconsistent: on historical_50k it lost to random labels
  with rich features alone (86.20 against 87.02) and won with the Splink feature on (87.19
  against 86.57). One draw each. Over three draws with similarity blocking it won on average
  (87.99 against 86.60), and with the match weight it won narrowly (88.71 against
  88.51). On dense files, compare both on your own data.

## A public benchmark: North Carolina voters (NCVR)

The Leipzig NCVR benchmark (Saeedi et al., University of Leipzig; CC BY 4.0) holds real names
from the North Carolina voter roll in five sources of a million records each, with duplicates
whose values were corrupted by a generator (20% of records, two changes each: "mar6ha litttle",
"millers ceeek", postcode "28|73"). Four fields only - first name, surname, town and postcode -
so it is a test of sparse, name-only matching at scale; its errors are synthetic. The data is
not in this repository (it is real people's names); `research/scripts/ncvr_leipzig.py` says where to
download it. Each run takes a random sample of people with all their records across the five
sources, 300 random + 700 active labels from the truth, and scores against the true people;
the ceiling accepts every true pair blocking proposes.

| Sample, blocking keys | Draws | Records | Clustered F1 | Precision | Recall | Ceiling F1 | Pairs compared | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20,000 people, default keys | 3 | 28,534 | 95.51 | 99.74 | 91.64 | 95.64 | 0.4M | 10 s |
| 20,000 people, + postcode and initial keys | 3 | 28,534 | 99.22 | 99.57 | 98.88 | 99.46 | 0.5M | 12 s |
| 200,000 people, default keys | 1 | 285,812 | 89.77 | 99.00 | 82.11 | 90.36 | 2.2M | 51 s |
| 200,000 people, + postcode and initial keys | 3 | 285,713 | 95.63 | 98.13 | 93.27 | 97.42 | 7.2M | 121 s |

- **The matcher is not the limit; blocking is.** custmatch scores within about half a point of
  the ceiling - the best any matcher could do with the pairs blocking proposed - at 99.7%
  precision on 20,000 people, and within two points on 200,000. The missing recall is pairs
  never compared.
- **The common miss:** both names corrupted, town and postcode intact. Postcode alone cannot
  catch it: a postcode holds thousands of people, so the block is skipped as too big. A key of
  postcode plus a name's first letter can:

```json
  "extra_blocking": [["postcode"], ["postcode", "first_name:first1"], ["postcode", "last_name:first1"],
                     ["town", "last_name:soundex"]]
```

  `extra_blocking` keys take the blocking transforms: `first1`, `first3`, `soundex`, `year`.
- **Larger files need it more.** At 200,000 people more blocks grow past the size limit and
  recall falls with the default keys; the narrower keys recover most of the pairs (ceiling
  90.4 to 97.4) at three times the comparisons, and F1 rose from 89.8 to
  95.6. The default keys were run on one draw at this size, and the full benchmark (five
  million records) was not run.

**Against Splink.** Splink 4 (`--splink`) ran on the same samples with the same blocking keys -
DuckDB has no soundex, so its town key uses the surname's initial, which proposes more pairs -
name comparisons with term-frequency adjustment, town by Jaro-Winkler and postcode by edit
distance. Clustered F1, mean (range) over the draws:

| Postcode and initial keys | Draws | custmatch | Splink, m from all labels, p > 0.99 | Splink, m from all labels, threshold from labels | Splink, m from the 300 random labels, p > 0.99 | Splink unsupervised, best in hindsight |
| --- | --- | --- | --- | --- | --- | --- |
| 20,000 people | 3 | 99.22 (99.0-99.4) | 98.01 (97.8-98.1) | 97.68 (96.9-98.4) | not run | 84.97 (84.4-85.6) |
| 200,000 people | 3 | 95.63 (94.7-96.2) | 93.94 (93.7-94.2) | 88.34 (85.9-92.0) | 79.24 (79.2-79.4) | 63.95 (63.6-64.2) |

- **custmatch led on every draw:** by 1.0 to 2.3 points at 200,000 people. Before 0.1.0's
  threshold rule one draw's threshold came out too strict (85.44 F1, though blocking had
  proposed the pairs), and Splink led on average; see "How it decides".
- **Splink's labels were not its own.** "m from all labels" uses all 1,000, 700 of which
  custmatch's model chose as the most informative; given only the 300 random ones Splink scored
  about 79. Splink's own unsupervised training (EM) did not cope with four corrupted fields.
- On 20,000 people (three draws) custmatch led by about 1 point too. (The 20,000-people Splink figures are from the run before the
  Wikidata name list was removed; the custmatch figures beside them are current.)

Why custmatch's advantage is small or absent here, when it was 13 points on the customer file
(97.2 against Splink's 84.4 given labels): NCVR is the kind of problem Fellegi-Sunter models
handle well - four independent fields with generated typos. custmatch's main advantages need
customer data to work on: junk and shared contacts recognised, households, nicknames and
country cleaning, and the threshold chosen on whole customers to stop chaining. NCVR has no
emails, phones or households. On clean, name-only data, Splink given good labels is as good or
better.

**Two ideas from anonlink** (CSIRO Data61's linkage toolkit):

| Variant | F1, 20,000 people (3 draws) | Precision | F1, 200,000 people (1 draw) | Precision | Ceiling, 200,000 |
| --- | --- | --- | --- | --- | --- |
| Postcode and initial keys | 99.22 | 99.57 | 96.04 | 97.93 | 97.35 |
| + one record per source | 99.35 | 99.86 | 96.47 | 99.19 | 97.35 |
| Similarity (LSH) blocking instead | 98.58 | 99.55 | 94.65 | 96.90 | 96.48 |
| LSH blocking + one record per source | 98.71 | 99.85 | 95.42 | 98.96 | 96.48 |

- **One record per source per person** (`"unique_sources": true`, or a list of sources). Each
  NCVR source is a list of different voters, so a customer holding two records from one source
  is two people; it is cut at its weakest link, as a steward's `different` would cut it, and
  `add` refuses such joins. Precision rose by 0.3 points on 20,000 people and by 1.3 on 200,000,
  where same-name strangers are commoner; no customer is larger than five records, one per
  source. Use it for a source that truly holds one record per person: a voter roll, a loyalty
  scheme, one account per customer.
- **Similarity blocking** (`"lsh_blocking"`): MinHash locality-sensitive hashing of the names'
  letter pairs, so names with typos in both still meet without hand-picked keys - the idea of
  anonlink's blocklib, on plain text. It found nearly as many true pairs as the postcode keys
  (ceiling 96.48 against 97.35 at 200,000 people) with fewer comparisons, and scored
  close to them: 98.58 against 99.22 on 20,000 people. (Before 0.1.0's threshold rule one draw
  dropped to 89.94, which made it look worse than it is.) Hand-picked keys still did a little
  better here; similarity blocking needs no knowledge of the file.

**What that exposed: thresholds from few random matches.** custmatch chooses the threshold on
the randomly drawn labels only - the actively chosen ones are deliberately hard and would bias
it. On NCVR only about one random candidate pair in 25 is a match, so 300 random labels hold a
dozen or so, and the recall estimate cannot tell a wide range of thresholds apart. Until 0.1.0
the strictest of them was taken, which cost real recall on some draws; now the middle one is
(guide "How it decides"), which fixed those draws. With few random matches, label more random
pairs: `summary.json` warns below 30.

---

[← Matching for marketing: against a CDP](08-against-a-cdp.md) · [Contents](README.md) · [Reproducing published results →](10-reproductions.md)
