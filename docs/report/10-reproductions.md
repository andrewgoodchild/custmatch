# 10. Reproducing published results

custmatch is built on published results; these chapters check they hold. Magellan's
classical baseline, Ditto's transformer and zero-shot LLM matching (Peeters, Steiner & Bizer)
were rerun from their authors' code, with what reproduced and what did not.

## What reproduced

| Target | Published | Here | |
|---|---|---|---|
| Magellan (SIGMOD'18), 12 datasets | 37.4–100 | matched or beaten on all 12 | ✅ |
| Ditto (PVLDB'21) DBLP-Scholar | 95.80 | 95.3 | ✅ |
| Ditto DBLP-ACM | 99.10 | 98.3 | ✅ |
| Ditto Walmart-Amazon | 83.73 | 84.8 | ✅ |
| Ditto Fodors-Zagats | 100.00 | 97.7 | ✅ |
| Ditto Abt-Buy (full Ditto) | 89.33 | 85.1 | ⚠️ 4.2 below |
| Ditto Beer / iTunes-Amazon | 90.46 / 97.80 | 80.0 / 82.4 | ⚠️ 14 and ~27 test positives |
| Ditto Amazon-Google | 74.67 | 35.3 → **74.0** with warmup | ✅ after fixing a training collapse |
| LLM (EDBT'25) GPT-4o-mini, 5 sets | 70.98–90.95 | 71.98–91.98 | ✅ all within 1.1 |
| LLM (EDBT'25) GPT-4o, 5 sets | 73.56–90.47 | 74.75–90.62 | ✅ all within 1.5 |
| LLM (EDBT'25) GPT-4, 5 sets | 75.61–95.15 | — | not run (~$21); GPT-6 Luna scores 80.86–95.08 on the same sets for $0.33 |

Two of eleven Ditto runs collapsed to the degenerate "everything matches" solution — F1 sitting
exactly on 2p/(1+p) for the dataset's positive rate, with nothing in the output to say so.
Ditto trains with `num_warmup_steps=0`, the known trigger for this BERT fine-tuning failure
mode; adding 10% warmup took Amazon-Google from 35.3 to 74.0, against the paper's 74.67.

## Reproduction 1 — Magellan (Mudgal et al., SIGMOD 2018)

`scripts/magellan_baseline.py`. Per-attribute similarity features (exact, token Jaccard,
3-gram Jaccard, Levenshtein, Jaro, Jaro–Winkler, Monge–Elkan, numeric differences), then the
best of decision tree / random forest / logistic regression / naive Bayes / SVM chosen on the
validation split and refitted on train+valid. Test F1 on the match class.

| Dataset | Paper (Magellan) | This run | Δ |
|---|---|---|---|
| Structured/Amazon-Google | 49.1 | **59.9** | +10.8 |
| Structured/Beer | 78.8 | **86.7** | +7.9 |
| Structured/DBLP-ACM | 98.4 | **98.6** | +0.2 |
| Structured/DBLP-Scholar | 92.3 | **93.8** | +1.5 |
| Structured/Fodors-Zagats | 100.0 | **100.0** | 0.0 |
| Structured/iTunes-Amazon | 91.2 | **94.7** | +3.5 |
| Structured/Walmart-Amazon | 71.9 | **78.0** | +6.1 |
| Dirty/DBLP-ACM | 91.9 | **94.7** | +2.8 |
| Dirty/DBLP-Scholar | 82.5 | **88.3** | +5.8 |
| Dirty/iTunes-Amazon | 46.8 | **67.9** | +21.1 |
| Dirty/Walmart-Amazon | 37.4 | **43.6** | +6.2 |
| Textual/Abt-Buy | 43.6 | **68.0** | +24.4 |

**Verdict: reproduced, and then some.** Every dataset matches or beats the published figure.
The classical baseline in the 2018 paper is weaker than a classical baseline has to be: with
modern string-similarity features applied to *every* attribute (including long text), Abt-Buy
goes from 43.6 to 68.0 and Dirty/iTunes-Amazon from 46.8 to 67.9. Runtime: about 8 minutes
for all 12 datasets on CPU. This matters for the field's headline claim — several "deep
learning beats classical" gaps shrink substantially when the classical side is tuned.

## Reproduction 2 — Ditto (Li et al., PVLDB 2021)

`scripts/run_ditto_all.sh`, using the authors' own code from `megagonlabs/ditto` with the
minimum changes needed for Apple Silicon (see [chapter 6](06-blocking-and-scale.md)). Configuration follows the paper:
RoBERTa-base, lr 3e-5, MixDA with the per-dataset operator from the repo's
`run_all_er_magellan.py`, `--dk general`, 40 epochs on the small datasets and 15 otherwise,
test F1 taken at the best-validation epoch with the threshold tuned on validation.

Paper reference values are Table 5 of the Ditto paper (verified from the PDF): the **Ditto**
column is the full system, **Baseline** is plain RoBERTa fine-tuning with no optimisations.

I ran two configurations, because the first one turned out to be broken (see below):
**DA+DK** is full Ditto with MixDA, and **DK only** drops MixDA, matching the paper's
`Ditto(DK)` column.

| Dataset | Test positives | Paper: Ditto | Paper: Ditto(DK) | Mine: DA+DK | Mine: DK only |
|---|---|---|---|---|---|
| Structured/Beer | 14 | 94.37 | 90.46 | 55.6 | 80.0 |
| Structured/iTunes-Amazon | ~27 | 97.06 | 97.80 | 63.8 | 82.4 |
| Structured/Fodors-Zagats | ~22 | 100.00 | 100.00 | **97.7** | — |
| Structured/Walmart-Amazon | 193 | 86.76 | 83.73 | 17.2 | **84.8** |
| Structured/Amazon-Google | 234 | 75.58 | 74.67 | — | 35.3 (**74.0** with warmup) |
| Structured/DBLP-ACM | 444 | 98.99 | 99.10 | — | **98.3** |
| Structured/DBLP-Scholar | 1,070 | 95.60 | 95.80 | — | **95.3** |
| Textual/Abt-Buy | 206 | 89.33 | 81.69 | **85.1** | — |

**Four of eight reproduce within about 1–4 F1**: DBLP-Scholar 95.3 vs 95.80, DBLP-ACM 98.3 vs
99.10, Walmart-Amazon 84.8 vs 83.73 (above the published figure) and Fodors-Zagats 97.7 vs 100.
Abt-Buy was run as full Ditto (DA+DK), so its comparison is the paper's 89.33: 85.1 is 4.2
below (it happens to sit above the paper's Ditto(DK) 81.69, a different configuration). On the
datasets with enough test positives to measure, **Ditto's method holds up.**

Three do not as run (Amazon-Google recovers with warmup, below), and they fail in a specific, recognisable way.

### The failures are training collapses, not bad scores

Walmart-Amazon under MixDA scored **17.2**, and Amazon-Google without MixDA scored **35.3**.
These are not "somewhat worse" numbers. 17.2 is exactly 2p/(1+p) at Walmart-Amazon's 9.4%
positive rate — the score for labelling every pair a match — and Amazon-Google sat at 0.185,
its own all-positive score, for 10 of 15 epochs. The model had collapsed to the degenerate
solution and the threshold search then did the only thing it could.

I eliminated the mundane explanations:

- **Not the data.** Both DK-injected training files have every row and the right label split
  (Walmart-Amazon 5,568/576; Amazon-Google 6,175/699). All 18 cache files verified line-for-line.
- **Not truncation.** Only 16.8% of Walmart-Amazon pairs exceed 128 tokens, against 64.2% for
  Abt-Buy, which trained fine; Beer has 0% over and still underperformed.
- **Not a broken augmenter.** Across 400 sampled rows, `drop_col` never lost the separator and
  never emptied a side.

**MixDA makes collapse more likely but is not the cause.** The cleanest evidence is a
controlled A/B on Walmart-Amazon — same data, same seed, same settings, MixDA the only
difference: **17.2 with, 84.8 without.** Turning it off also recovered Beer (55.6 → 80.0) and
iTunes-Amazon (63.8 → 82.4). But Amazon-Google collapsed with MixDA already off, so the
augmentation cannot be the whole story.

What fits all of it is ordinary fine-tuning instability — the failure mode Mosbach et al.
(ICLR 2021) document for BERT, where a fraction of runs fall into a degenerate solution. Ditto
trains with `num_warmup_steps=0`, and no warmup is precisely the condition that literature
identifies as the trigger. MixDA plausibly raises the odds by making the early gradient signal
noisier. Two of my eleven runs collapsed outright and two more landed well below the paper —
which is also consistent with the paper reporting the best of 4 configurations × 5 seeds, a
protocol that quietly discards exactly these runs.

### The fix: 10% warmup

The Ditto patch (`patches/ditto-mps-and-fixes.patch`) adds `DITTO_WARMUP_FRAC`, a linear warmup
over that fraction of training steps. With the paper's recipe unchanged otherwise, I reran
Amazon-Google, the run that collapsed with MixDA already off, with 10% warmup. Same data, seed,
15 epochs and `--dk general`; warmup is the only change:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|
| No warmup (test F1) | 27.8 | 18.5 | 23.3 | 18.5 | 18.5 | 18.5 | 33.8 | 35.3 |
| 10% warmup (test F1) | 57.7 | 64.0 | 72.2 | 72.3 | 71.9 | **74.0** | 72.9 | 71.8 |

At the best-validation epoch the warmup run scores **74.0 against the paper's 74.67**, up from
35.3. The no-warmup run spent 5 of its first 7 epochs on the all-positive score 18.5 and never
recovered past 35.3; the warmup run never touched it. Amazon-Google therefore reproduces too:
the recipe was fine, but the schedule was not.

One caveat on how far to lean on the exact figure: an earlier, interrupted warmup run
(`results/ditto/Structured_Amazon-Google_warmup_partial.log`) had the same settings but diverged
from epoch 1 (63.7 vs 57.7) and reached 72.3 by epoch 2. MPS training is not bit-reproducible
across runs even with a fixed seed, so treat single-run numbers here as ±2 F1. The collapse is
different: it either happens or it does not, and with warmup it did not.

I did not rerun the MixDA collapses (Walmart-Amazon, Beer, iTunes-Amazon) with warmup, so
whether warmup alone makes MixDA safe is still untested.

### What this means if you are reproducing Ditto

1. **Check for the collapse explicitly.** Compare each run's F1 against 2p/(1+p) for that
   dataset's positive rate. If it is sitting on that number, the run is dead, not merely weak —
   and nothing in Ditto's output will tell you.
2. **Run more than one seed.** The published numbers are best-of-20; a single run is not
   comparable to them, and the small datasets swing hardest.
3. **Treat `--da` as suspect** until you have a no-`--da` control.

### Where the small datasets stand

Beer and iTunes-Amazon improved a lot with MixDA off (55.6 → 80.0 and 63.8 → 82.4) but stay
below the published 90.46 and 97.80. Their test splits hold **14** and **~27** positive pairs,
so one flipped prediction moves F1 by 3–7 points, and the published figure is the best of
twenty runs. Fodors-Zagats, the third tiny dataset, reached 97.7 against 100 — it is the easy
one, near-duplicate restaurant records that Magellan also solves perfectly.

The takeaway is not that these datasets are hard but that **F1 on a test set with 14 positives
is not a measurement.** Any claim resting on Beer or iTunes-Amazon, in any paper, is carrying
a confidence interval several times wider than the differences usually being argued over.

### A bug in Ditto's augmentation

Walmart-Amazon crashed 13 seconds in, inside the authors' own code:

```
File "ditto_light/dataset.py", line 80, in __getitem__
    left, right = combined.split(' [SEP] ')
ValueError: not enough values to unpack (expected 2, got 1)
```

The MixDA `drop_col` operator can delete the `[SEP]` marker that separates the two records, and
the split then returns one piece. It is data-dependent, which is why Beer and iTunes-Amazon
survived the same operator and Walmart-Amazon did not. I patched it to fall back to the
un-augmented pair when the separator is gone (`git -C external/ditto diff`) — the augmented
example is then simply identical to the original, which is harmless to MixDA.

## Reproduction 3 — zero-shot LLM matching (Peeters, Steiner & Bizer, EDBT 2025)

`scripts/llm_zeroshot_mlx.py`. Their exact downsampled test sets and their `domain-complex-force`
prompt ("Do the two product descriptions refer to the same real-world product? Answer with
'Yes' if they do and 'No' if they do not."), temperature 0, and their answer parsing (strip
punctuation, lowercase, match on "yes"). The model is **Llama-3.1-8B-Instruct, 4-bit, via MLX**,
because the environment's OpenAI key was dead at the time — so this is a like-for-like
*protocol* reproduction with a much smaller model. The GPT rows were rerun later; see
"The GPT rows, rerun" below.

| Test set (pairs / positives) | This run: Llama-3.1-8B-4bit | Published: GPT-4o-mini | Published: GPT-4 | Published: Llama-3.1-70B | Published: Ditto |
|---|---|---|---|---|---|
| Abt-Buy (1,206 / 206) | **69.10** (P 55.4, R 91.8) | 90.95 | 95.15 | 89.84 | 91.31 |
| Walmart-Amazon (1,193 / 193) | **57.86** (P 41.5, R 95.3) | 86.36 | 89.00 | 84.85 | 86.39 |
| WDC Products 80% cc (1,239 / 250) | **70.44** (P 57.1, R 92.0) | 80.84 | 88.35 | 83.67 | 84.90 |

GPT-4o-mini / GPT-4 columns are the `domain-complex-force` rows of the authors' own results
workbook; the 70B and Ditto columns are their Table 4 (best prompt per model). Run time was
about 16 minutes per test set on the laptop, at 1.3 pairs/second.

**What reproduces:** the protocol, cleanly — prompts, parsing, and test sets all replay, and
the *ordering* of the published results is confirmed: frontier models ≫ open 8B model.

**What does not:** an 8B model is not a small GPT-4. It sits 20–29 F1 points below GPT-4o-mini
on the identical pairs, and below Ditto trained on the same data. The failure is one-sided and
consistent: **recall 92–95%, precision 42–57%.** It says "Yes" to almost everything that looks
vaguely similar.

Two cautions for anyone citing 8B zero-shot numbers:

- TailorMatch reports Llama-3.1-8B zero-shot at 56.57 / 42.04 / 53.36 on these three datasets,
  which is 12–17 points below what I measured. That is **not** a contradiction: they score the
  full test splits (Abt-Buy 206 positives in 1,916 pairs, ~11%) while these are the downsampled
  sets (~17–20% positive). A matcher that over-predicts matches is flattered by a denser
  positive rate. F1 is not portable across test sets with different base rates, and the
  literature's 8B numbers are not comparable to each other unless the split is identical.
- The prompt matters as much as the model. In the authors' own results GPT-4o-mini swings from
  90.95 to 60.81 on Abt-Buy across their eight prompt variants.

### The GPT rows, rerun

`scripts/llm_zeroshot_openrouter.py`, the same prompts, parsing and test sets, with the models
pinned to the snapshots in the authors' workbook and served via OpenRouter (Azure).
`gpt-4o-mini` is requested unpinned because the pinned name is only served by OpenAI directly,
which a zero-data-retention account cannot use; 2024-07-18 is its only release.

| Test set | GPT-4o-mini published | GPT-4o-mini here | GPT-4o published | GPT-4o here |
|---|---|---|---|---|
| Abt-Buy | 90.95 | 91.98 | 90.47 | 90.62 |
| Amazon-Google | 70.98 | 71.98 | 73.56 | 74.75 |
| DBLP-Scholar | 86.11 | 86.56 | 89.76 | 89.81 |
| Walmart-Amazon | 86.36 | 86.60 | 84.04 | 84.55 |
| WDC Products 80% cc | 80.84 | 81.36 | 87.64 | 89.06 |
| Cost, all five | $0.08 | $0.09 | $1.38 | $1.48 |

**All ten reproduce, each within 1.5 F1 and all slightly above the published figure.** No
answer was unparseable on any product set. The EDBT 2025 headline — that frontier LLMs match
products zero-shot at Ditto's level without any training data — holds up as published.

GPT-4o shows the same one-sided bias as the 8B model, only milder: recall 93–99% and
precision 62–86% on products. That matters for the next subsection.

### Newer models: GPT-6 Luna, and Jev

Same protocol, 2026 models that did not exist when the paper was written. **GPT-6 Luna** is
OpenAI's low-cost reasoning model ($0.10/$0.50 per million tokens). **Jev** is TypeSafe's
*decision* model: it does not generate text, it answers typed questions with probabilities
(below). **Jev Router** (`typesafe/jev-router`) is a product built on it that uses Jev to pick
another model, and a reasoning effort, per request; over these runs it sent 58% of pairs to Luna
itself, and the rest mostly to DeepSeek V4.1 Flash, Gemini 3.8 Flash and occasionally Claude
Sonnet 5 or GPT-6 Sol. Neither accepts `temperature`, so these rows are not bit-reproducible.

| Test set | GPT-4 published | GPT-6 Luna | Jev Router |
|---|---|---|---|
| Abt-Buy | 95.15 | 95.08 | **96.21** |
| Amazon-Google | 75.61 | **80.86** | 80.66 |
| DBLP-Scholar | **88.44** | 88.11 | 87.03 |
| Walmart-Amazon | 89.00 | 92.46 | **94.39** |
| WDC Products 80% cc | 88.35 | **92.18** | 91.80 |
| Cost, all five | $16.30 | **$0.33** | $1.41 |

**Luna matches GPT-4 (within 0.4) or beats it (by 3.5–5.3) on every set, at 1/50th of the cost** — the EDBT 2025 ceiling has
moved to a model priced for high volume. **Jev is not better than Luna and costs 4–10× more**:
it wins two sets and loses two by 1–2 points, which is inside the run-to-run noise of an
unpinned model, and it inherits DeepSeek's long reasoning. That reasoning mattered for the
setup: with a 2,000-token answer cap, 128 of Jev's replies — all from DeepSeek — were cut off
mid-thought and came back empty, and on the person pairs 44 of those 55 were true matches. The
table re-asks only those at an 8,000-token cap (Luna's 5 cut-off replies likewise); 10 were
still empty. Anyone running reasoning models through this protocol should set the cap high
and count empty replies, because the parser silently scores them as "No".

**Jev as the matcher.** `scripts/jev_decisions.py` calls Jev directly through OpenRouter's
Decisions API: the two records as `state`, one yes/no ("noul") question whose wording is the
task's own prompt sentence, no criteria or examples. It returns P(match) per pair, from one
pinned snapshot (`typesafe/jev-1.13-20260917`), at $0.042 per million input tokens and nothing
for output — **$0.11 for all seven test sets**, a quarter of Luna and ~1/16th of GPT-4o.

| Test set | Jev at P > 0.5 | Jev, cut-off tuned on the other half | AUC | GPT-6 Luna | GPT-4o |
|---|---|---|---|---|---|
| Abt-Buy | 89.70 | 89.8 | 0.994 | **95.08** | 90.62 |
| Amazon-Google | 73.79 | 72.6 | 0.950 | **80.86** | 74.75 |
| DBLP-Scholar | **90.02** | 89.1 | 0.988 | 88.11 | 89.81 |
| Walmart-Amazon | 87.78 | 86.9 | 0.986 | **92.46** | 84.55 |
| WDC Products 80% cc | 86.90 | 85.7 | 0.977 | **92.18** | 89.06 |
| historical_50k person pairs | 75.88 | **85.0** | 0.948 | 84.70 | 67.20 |
| Synthetic customer pairs | 94.95 | 94.3 | 0.991 | **97.14** | 95.28 |

The tuned column picks the cut-off on a random half of each test set and scores the other half,
both ways, averaged. On products Jev sits at GPT-4o's level and below Luna. On person pairs its
probabilities are the useful part: at 0.5 it is too strict on historical_50k, like GPT-4o, but
its ranking is good (AUC 0.948), so a cut-off of 0.25 learned from 330 labelled pairs takes it
to **85.0 — level with Luna, ahead of the forest, for $0.0095**. That is the property [chapter 2](02-whole-customers.md)
says matters for clustering: a score you can threshold, rather than a verdict you cannot. None
of the text models offer it.

**Jev's options, one at a time.** The runs above used none of Jev's API beyond a bare yes/no
question. `scripts/jev_variants.py` changes one thing at a time on the two person tasks: written
`criteria` for yes and for no (fixed in advance from domain knowledge — clerical drift in names,
dates and contacts counts as the same person; relatives and housemates do not), records as
JSON objects with named fields instead of strings, both together, a three-way `choice`
(same person / same household / different), and one call asking four questions at once
(match, same household, and whether each record's contact is a placeholder). $0.12 for all
eight runs.

| historical_50k person pairs | F1 at 0.5 | F1, tuned on the other half | AUC |
|---|---|---|---|
| Jev, bare question (above) | 75.88 | 85.0 | 0.948 |
| + criteria | **82.02** | 84.6 | 0.961 |
| + structured fields | 66.14 | 84.5 | 0.958 |
| + both | 77.40 | **87.1** | **0.966** |

| Synthetic customer pairs | F1 at 0.5 | F1, tuned on the other half | AUC |
|---|---|---|---|
| Jev, bare question (above) | 94.95 | 94.3 | 0.991 |
| + criteria | 96.98 | 96.0 | 0.995 |
| + structured fields | 95.85 | 97.0 | 0.997 |
| + both | 96.92 | 96.7 | **0.998** |
| three-way choice, structured | 95.88 | 96.6 | 0.997 |
| four questions in one call, structured + criteria | **97.13** | **97.2** | 0.997 |

A paired bootstrap (2,000 resamples of the same pairs) separates signal from the ±1.5 a
250-positive test can resolve:

- **Criteria is the one change that clearly helps on both tasks.** On historical_50k it lifts
  F1 at 0.5 by 6.1 (95% interval +3.4 to +9.1; `scripts/jev_bootstrap.py`) — the guidance about tolerable drift removes
  most of the over-strictness that previously needed a tuned cut-off — and raises AUC on both.
- **Structured fields sharpen the ranking but make Jev stricter still** (−9.7 F1 at 0.5 on
  historical_50k), so they pay off only with a tuned cut-off.
- **Both together give the best ranking on both tasks** (AUC +0.017 on historical_50k, interval
  +0.006 to +0.029). With the cut-off tuned on 330 of the 660 test pairs, **87.1 F1 on
  historical_50k is the highest score on that task here** — GPT-6 Luna 84.7, the forest 76.5 —
  for $0.016, with the caveats in "In short": best of four variants on the same pairs, against
  an untuned forest.
- **Extra questions in the same call cost nothing in match quality.** The four-question call
  scores the best F1 at 0.5 on customer pairs (97.1) and also returns same-household at 97.1
  F1 (tuned) and placeholder-contact at 93.5 F1 per record. The dedicated contact profiling of
  [chapter 5](05-customer-data.md) still wins on the last: it finds junk values at 100% precision and recall because it
  sees how often a value recurs across the whole file, which a single pair cannot show.
- **The three-way choice matches the yes/no question on matching** and scores 97.7 F1 on
  same-household. Neither household figure is comparable to [chapter 5](05-customer-data.md)'s 90.2, which is for
  whole households across the full file, not for hard pairs.

So the uplift is real and cheap: **criteria plus structured fields**, with the cut-off learned
from a few hundred labelled pairs, and extra questions riding along free.

### The part that matters for customer matching

I built the same kind of test from person records — `scripts/make_person_llm_task.py` samples
660 hard pairs from historical_50k (250 positives where the two records' names differ;
410 negatives that share a surname and have >0.75 name similarity) — and ran the identical
prompt, then scored the **same 660 pairs** with classical matchers (`scripts/person_task_baselines.py`).

| Method on the same 660 person pairs | P | R | F1 | Accuracy |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct, zero-shot | 80.0 | **1.6** | **3.1** | 62.6 |
| GPT-4o-mini, zero-shot ($0.009) | 89.7 | 38.4 | 53.8 | 75.0 |
| GPT-4o (2024-08-06), zero-shot ($0.14) | **99.2** | 50.8 | 67.2 | 81.2 |
| Jev Router, zero-shot ($0.65) | 96.8 | 72.4 | 82.8 | 88.6 |
| Jev 1.13 decision model, P > 0.5 ($0.0095) | 91.5 | 64.8 | 75.9 | 84.4 |
| GPT-6 Luna, zero-shot ($0.07) | 95.0 | 76.4 | **84.7** | **89.6** |
| Rule: name Jaro–Winkler > 0.9 AND same DOB | 91.7 | 13.2 | 23.1 | 66.7 |
| Random forest on similarity features, disjoint training clusters | 62.4 | 98.8 | **76.5** | 77.0 |

The 8B model answered "No" to **637 of 660** pairs. It is not a parsing artefact — the raw replies
are bare "No"s, and the misses are exactly the cases record linkage exists to handle:

```
Record 1: 'i wyatt 1746-08-03 weeford b75 5sp architect'
Record 2: 'j wyatt 1749-08-03 weeford b75 5sp architect'          -> "No"   (same person)

Record 1: 'oliverio goldsmith 1728-11-10 elphin'
Record 2: 'olivier goldsmith elphin m24 4ng'                       -> "No"   (same person)
```

Note the reversal. On product pairs the same model, same prompt, same temperature says "Yes"
to everything (precision 42–57%); on person pairs it says "No" to everything (recall 1.6%). A
general-purpose LLM has no calibrated prior about how much a name, a date of birth or an
address is *allowed* to differ between two records for the same customer — and that prior is
the entire content of the task. A random forest over string-similarity features, trained on a
few thousand pairs from other people entirely, gets 76.5 F1 on the same pairs.

**The frontier models show the same reversal, only less extreme.** GPT-4o says "Yes" to 93–99% of
matching products but to only 51% of matching people, and when it does say "Yes" to a person
pair it is right 99.2% of the time. Its accuracy (81.2) beats the forest's (77.0) because 62% of
these pairs are non-matches and it gets nearly all of those right. F1, which is about finding
the matches, is 67.2 against 76.5. GPT-4o-mini is further along the same line: 38% recall.
Three of GPT-4o's 660 replies were hedges on true matches ("The records provide limited
information…"), which the paper's parser counts as "No"; that is under 1.5 points of recall
and does not change the picture.

GPT-4o's gap is not scale alone. Zero-shot, it brings a prior about what "the same person"
means that is far stricter than these records need: a changed year of birth or a dropped first
name is, to GPT-4o, usually a different person. **GPT-6 Luna is the first model here to beat
the forest (84.7 vs 76.5)**, at 95% precision, and its short visible explanations suggest why —
it reasons about which differences are plausible clerical drift.

One caution on that number. historical_50k is built from real historical figures, and 7 of
Luna's 50 visible explanations cite outside knowledge ("…Henry de Vere, 18th Earl of Oxford,
whose birth date was 1593-02-24…"). That knowledge cuts both ways — it rejected two true matches
because "the historical Oliver Goldsmith died in 1774" — and its reasoning is hidden, so the net
effect cannot be measured from this task. A real customer file offers no such help, which is
what the next test controls for.

### Customer-style person records, where no model can know the people

`scripts/customer_person_llm_task.py` builds the same kind of task from the synthetic
two-source customer file (`customer_data.py`): 660 pairs, 250 positive, same prompt. Positives
are the same simulated person with the name written differently; negatives are different people
sharing a surname with name similarity > 0.75 — mostly household members. The records carry what
a customer file carries: name, DOB where the source has it, address, email and phone, including
the junk contacts (`xxx@xxx.com 1111111111`) and role names in name fields ("lady of house"). The
random forest is trained on hard pairs drawn from a disjoint half of the simulants.

| Method on the same 660 synthetic customer pairs | P | R | F1 | Accuracy |
|---|---|---|---|---|
| GPT-4o-mini, zero-shot ($0.010) | 98.1 | 80.4 | 88.4 | 92.0 |
| GPT-4o (2024-08-06), zero-shot ($0.17) | 93.8 | 96.8 | 95.3 | 96.4 |
| Jev 1.13 decision model, P > 0.5 ($0.010) | 95.9 | 94.0 | 95.0 | 96.2 |
| Jev Router, zero-shot ($0.23) | **99.6** | 93.6 | 96.5 | 97.4 |
| Random forest on similarity features, disjoint simulants (free) | 96.1 | **97.6** | 96.8 | 97.6 |
| GPT-6 Luna, zero-shot ($0.027) | 99.2 | 95.2 | **97.1** | **97.9** |

The picture changes completely. With contact fields to go on, GPT-4o is 28 points better than on
historical_50k and within 1.5 of the forest; Luna and the forest are level (97.1 vs 96.8, well
inside the ±1.5 a 250-positive test can resolve). (Luna's score on people it cannot know does
not show that its historical_50k lead is free of recognition - that would need historical_50k
with the names swapped - but the forest-level result here is at least not memorisation.)

**For customer matching, then:** the earlier conclusion of this section — that a zero-shot LLM
is not a matcher — was drawn from genealogy-style records alone and does not survive this test.
What holds is narrower:

1. **On records with good contact fields, a trained forest is as good as the best LLM and
   free.** An LLM adds nothing there except the cost of a call per pair.
2. **On contact-poor, noisy records, a current model beats the forest** — Luna 84.7, and Jev
   87.1 with written criteria, structured fields and a cut-off tuned on a few hundred labelled
   pairs, against 76.5 (a forest with no tuned cut-off and unlike-the-task training pairs; see
   the caveat in "In short") — and older LLMs do not. The obvious slot is the arbiter for the
   uncertain middle band that [chapter 2](02-whole-customers.md) gives Ditto, and [chapter 2](02-whole-customers.md) tested exactly that: Jev
   there scored 82.97 held out, below the forest alone (84.68). Being best on hard pairs in
   isolation did not make it best in the pipeline.
3. **Measure on your own records before choosing.** The same model moved 28 points between two
   person tasks built the same way. No public benchmark stands in for a specific customer file.

The other two LLM uses the 2025–26 literature supports are labelling training data (Steiner &
Bizer 2026 show Ditto trained on GPT-5.2 labels ≈ Ditto trained on human labels), which [chapter 4](04-labels.md)
tests, and selecting among blocked candidates (ComEM, COLING 2025), which remains untested here.

---

[← Against other tools](09-against-other-tools.md) · [Contents](README.md) · [What the tool adopted, and what it dropped →](11-adopted-and-dropped.md)
