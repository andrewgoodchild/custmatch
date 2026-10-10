# 4. Labels

Labels are the most important input, and the price of entry: Splink can start with none,
custmatch cannot. This chapter measures how many it needs - about 300, most of them chosen by
active learning - whether it could start with none (not reliably), and whether a language model
could supply them (only where its labels are near-perfect; custmatch tried it and dropped it).

## How many labels, and can it start with none?

`research/scripts/label_curve.py`, `research/results/label_curve.json`. custmatch's defaults
trained on 50 to 1,000 labelled pairs, drawn two ways: all at random, or 100 at random and then
batches of 100 chosen by active learning (the workflow of `custmatch sample --active` and the
dashboard). Three seeds each, labels answered from the truth; below 200 labels - where the
pipeline normally refuses to run - the floor was lowered for the experiment. Mean clustered F1:

| Dataset | 100 random | 300 random | 1,000 random | 200 (100 + 100 active) | 300 (100 + 200 active) | 1,000 (100 + 900 active) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fake_1000 (Splink's) | 81.76 | 83.15 | 86.94 | 84.17 | 87.17 | 88.71 |
| febrl3 (Splink's) | 96.88 | 99.71 | 99.62 | 99.30 | 99.84 | 99.99 |
| febrl4 (Splink's) | 92.54 | 98.55 | 99.92 | 97.34 | 99.84 | 99.99 |
| historical_50k (Splink's) | 80.48 | 79.17 | 85.68 | 83.35 | 83.47 | 84.81 |
| NCVR, 20,000 people | 65.89 | 94.28 | 97.08 | 96.36 | 98.50 | 99.40 |
| Synthetic Australian sign-ups (sparse) | 54.14 | 69.12 | 72.82 | 65.86 | 77.09 | 82.65 |

- **About 300 labels, chosen well, are the real price.** With active learning, 300 labels come
  within 1.6 points of 1,000 chosen the same way on five of six datasets. The sparse sign-up file
  is the exception (77.09 against 82.65): its hard cases need more examples.
- **Choosing labels is worth more than adding them:** 300 chosen by active learning beat 1,000
  random on four of six datasets. febrl4 is a tie (0.08 behind) and historical_50k the
  exception, where dense records reward random draws (as in [chapter 6](06-blocking-and-scale.md)).
- **The 200-label floor is in the right place.** Below it, a run can fail outright: one draw of
  100 random NCVR labels scored 14.07, one of 50 febrl4 labels 7.54, and one of 50 sign-up labels
  held no match at all.

**With no labels.** custmatch's own Fellegi-Sunter match weight (`custmatch/matchweight.py`: m
fitted by EM on the candidate pairs, no labels) scored every candidate pair, and customers were
cut at label-free thresholds: the top share of pairs EM estimated to be matches (its lambda), and
fixed weights of 10 and 20 bits. The best weight in hindsight is an upper bound no real run could
know. Clustered F1, beside 300 labels:

| Dataset | EM's match share | Top share | Weight 10 | Weight 20 | Best in hindsight | 300 labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fake_1000 (Splink's) | 0.31 | 78.52 | 82.88 | 42.68 | 84.46 | 87.17 |
| febrl3 (Splink's) | 0.10 | 99.95 | 97.15 | 99.99 | 98.77 | 99.84 |
| febrl4 (Splink's) | 0.08 | 99.91 | 95.37 | 99.99 | 97.17 | 99.84 |
| historical_50k (Splink's) | 0.39 | 4.65 | 0.15 | 87.14 | 87.33 | 83.47 |
| NCVR, 20,000 people | 0.55 | 0.01 | 0.02 | 51.90 | 77.04 | 98.50 |
| Synthetic Australian sign-ups (sparse) | 0.09 | 25.05 | 21.27 | 65.38 | 73.16 | 77.09 |

No label-free rule holds across datasets. Taking EM's share is near-perfect on FEBRL and
collapses where EM decides most candidate pairs match (historical_50k, NCVR: one giant
customer); a fixed weight suits one dataset and fails the next; and even the best threshold in
hindsight trails 300 labels by 21 points on NCVR. So custmatch keeps its 200-label floor rather
than offering a zero-label mode that is right on some files and badly wrong on others, with no
way to tell which. Splink's unsupervised mode works in practice because a person sets the
threshold by looking at example pairs - labelling by another name.

## Can an LLM replace the hand labels? Only where its labels are near-perfect

`research/scripts/jev_labels.py`. The use of LLMs the recent literature supports (Steiner & Bizer 2026) is
labelling training pairs for a supervised matcher. Here Jev — criteria + structured fields, nothing
tuned on true labels — labels every training pair, and the forest (same features, same settings)
trains on Jev's labels instead of the truth. Hard labels take P > 0.5; soft labels enter each pair
twice, as a match weighted p and a non-match weighted 1 − p; a third variant picks Jev's cut-off on
300 randomly chosen hand-labelled pairs. Same test pairs and held-out clustering protocol as above.

| | Customer pipeline (15,444 pairs labelled, $0.41) | historical_50k (12,000 pairs, $0.28) |
|---|---|---|
| Jev label accuracy at 0.5 | **98.9%** (P 98.3, R 99.0) | 91.3% (P 97.7, R 73.4) |
| Forest on true labels | **98.53** | **84.68** |
| Forest on Jev hard labels | 97.99 | 74.09 |
| Forest on Jev soft labels | 98.03 | 72.87 |
| Forest on Jev labels, cut-off from 300 hand labels | 97.99 (cut-off 0.5) | 71.88 (cut-off 0.35: P 91.8, R 86.0) |

**On customer-style records it works: no hand labels, half a point lost.** Jev labels 98.9% of the
training pairs correctly, and the forest trained on those labels scores 97.99 held out against
98.53 on the truth, for $0.41. The 300 hand labels pick 0.5 anyway.

**On contact-poor records it fails, and more labels do not fix it.** Jev's labels are 97.7% precise
but miss 27% of true matches — the over-strictness of [chapter 10](10-reproductions.md) again — and the forest learns that
strictness: 74.09 against 84.68. Lowering the cut-off with 300 hand labels recovers recall (86%) at
the cost of precision (92%), and the forest gets *worse* (71.88): false match labels teach it to merge
people, which closure punishes far more than missed ones. Precision first, again — now in the
labels.

So LLM labelling is a real saving exactly where the task is already easy for the LLM, and not
where a supervised matcher is most needed. Measure the LLM's label accuracy on a few hundred of
your own pairs first: here ~99% was enough and ~91–93% was not.

## Tried in custmatch, and dropped

custmatch once had a `label` command built on this: Jev labelled pairs through OpenRouter, but only
after labelling a few hundred of your own hand-labelled pairs at least 98% correctly. It was tried
on synthetic data to see what the hype was about. In validation (before rich features), 2,000 Jev
labels gated on 300 hand labels cost $0.06 and scored 95.85 clustered F1 on all records, 97.34 on
people outside the labelled pairs - no better than active learning with hand labels, which keeps
the data in house. Sending customer records to a third party is a security risk, so the command was
removed; the code lives on in `research/scripts/jev_labelling.py` for the scripts that produced these
numbers.

---

[← Which matcher](03-which-matcher.md) · [Contents](README.md) · [Customer data: fake contacts, households and sparse records →](05-customer-data.md)
