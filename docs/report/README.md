# custmatch: the research

What works for matching customer records, measured. This report is the evidence behind
custmatch: each chapter answers one question - which matcher, how to set the threshold, what to
do about fake and shared contacts, how far it scales, how it compares with other tools - and
gives the experiments, their numbers and their caveats. How to *use* the tool is in
[guide.md](../guide.md); this is why it works the way it does.

**How to read it.** The findings come first, each pointing to the chapter that holds the
evidence. Chapters [2](02-whole-customers.md)-[9](09-against-other-tools.md) are the questions that shaped the tool; [chapter 10](10-reproductions.md) reproduces
published results the rest builds on; [chapter 11](11-adopted-and-dropped.md) lists what the tool adopted and what it tried
and dropped; [chapter 12](12-the-service.md) covers the service; [chapter 13](13-open-questions.md) what is still open. Every chapter is a file of its own and names the scripts that reproduce it
([reproducing.md](../reproducing.md) has the commands), and the test suite checks every number
here against `results/*.json`.

**Where it ran.** An Apple M1 Pro laptop (16 GB) for most of it, and cloud machines on Modal for
the runs of millions of records. Hosted language models (GPT-4o, GPT-4o-mini, GPT-6 Luna, the
Jev router and decision model) ran through OpenRouter in September 2026 for $7.26 in total;
GPT-4 is cited, not rerun (about $21). Hosted models are as served then (Jev: snapshot
`typesafe/jev-1.13-20260917`) and may have changed since. Most experiments ran between
September and October 2026; corrections are marked where a later check changed a result.

**Scoring.** Matching is scored by *pairs*: precision is the share of pairs put together that
are one person, recall the share of true pairs put together, F1 their balance. **Clustered**
scores count every pair of records inside the customers produced - after links are joined up -
and are the figure that matters for customer data; **pairwise** scores judge each compared pair
alone. Unless a table says otherwise, numbers are clustered F1 on a 0-100 scale.

## Chapters

1. [Data, and the work it builds on](01-data.md)
2. [Match whole customers, not pairs](02-whole-customers.md)
3. [Which matcher](03-which-matcher.md)
4. [Labels](04-labels.md)
5. [Customer data: fake contacts, households and sparse records](05-customer-data.md)
6. [Blocking and scale](06-blocking-and-scale.md)
7. [Golden records](07-golden-records.md)
8. [Matching for marketing: against a CDP](08-against-a-cdp.md)
9. [Against other tools](09-against-other-tools.md)
10. [Reproducing published results](10-reproductions.md)
11. [What the tool adopted, and what it dropped](11-adopted-and-dropped.md)
12. [The service](12-the-service.md)
13. [What is still open](13-open-questions.md)

## The findings

### In brief

1. **Judge the threshold on whole customers, not pairs.** Joining accepted pairs into customers
   turns a few false links into giant customers: a random forest at its best *pairwise*
   threshold merged 8,200 of 12,880 records into one customer. The threshold has to be chosen on
   the customers it produces, and precision is worth more than recall. The effect is known
   (Hassanzadeh et al., VLDB 2009); the method here - the threshold chosen from cross-fitted
   labels by closure-aware F1 - is custmatch's central design choice ([chapter 2](02-whole-customers.md)).
2. **A random forest is enough for customer data.** On records with addresses, emails and phones
   a transformer (Ditto) adds under a point; on genealogy-style records with no contact details
   it adds eleven. Averaging the forest with CatBoost adds 0.4 on average, up to 1.8 on hard files (chapters [2](02-whole-customers.md)-[3](03-which-matcher.md)).
3. **Fake and shared contacts can be told apart without labels,** by who shares them: a
   placeholder is shared by people who agree on nothing else, a family email by people with one
   surname or address ([chapter 5](05-customer-data.md)).
4. **Labels decide the result.** About 1,000 are needed; active learning matters most on sparse
   records; a language model can label pairs only where its labels are near-perfect ([chapter 4](04-labels.md)).
5. **Blocking sets the ceiling.** A pair it never proposes is never matched, and at millions of
   records it is the main limit again ([chapter 6](06-blocking-and-scale.md)).
6. **Matching is half of a golden record.** Merging recovers about half the recoverable errors in
   a field; past about 98 clustered F1 the source data, not the matcher, is the limit ([chapter 7](07-golden-records.md)).
7. **Against exact-identifier matching (a CDP),** custmatch finds far more of each customer -
   recall 86.2 against 64.3 - at slightly lower precision, and with a precision target sends
   a tenth as many messages to the wrong person ([chapter 8](08-against-a-cdp.md)).
8. **Against other tools,** it is the most accurate on customer data (97.99 against Splink's
   78.88 at a million records); Splink is faster once its large blocks are skipped, and dedupe
   is as accurate but about 90 times slower ([chapter 9](09-against-other-tools.md)).
9. **Published results mostly reproduce,** with one trap: Ditto can collapse silently to "every
   pair matches" when trained without warmup ([chapter 10](10-reproductions.md)).
10. **A zero-shot language model is a good judge of hard pairs and a poor matcher in the
    pipeline:** its errors chain through customers like any other false link (chapters [2](02-whole-customers.md), [10](10-reproductions.md)).

### In detail

**The field's published comparisons need care.** The classical baseline in the most-cited
deep-learning comparison is understated - a straightforward similarity-feature classifier
matches or beats it on all 12 datasets, by up to 24 F1 ([chapter 10](10-reproductions.md)) - and FEBRL, the person
dataset most people reach for, is saturated: rules score 97.7 F1 and three matchers ≈99.9
([chapter 3](03-which-matcher.md)). Use historical_50k or NCVR to rank methods on people.

**Blocking → a supervised pairwise matcher → a clustering pass, with the threshold tuned on
the clustering metric, not on pairwise F1.**

That last clause is the one that bites. On held-out person records a random forest at its
best pairwise threshold is 93.4% precise and scores 84.8 pairwise F1 — and transitive closure
still merges **8,200 of 12,880 records into a single cluster**, for a clustered F1 of **0.45** (on
the 0-100 scale: under half a percent).
Usable clusters need a far stricter threshold (0.95), which costs 20 points of pairwise F1.
Precision is worth much more than recall once you cluster: closure propagates a false edge
super-linearly, while a missed edge costs you once.

**On person records a transformer is worth it.** Same pairs, same splits:

| | Random forest | Ditto (RoBERTa) |
|---|---|---|
| Pairwise F1 | 84.8 | **88.2** |
| Clustered F1, threshold tuned on one half, scored on the other | 84.68 | **95.42** |
| Clustered F1, best threshold picked on the test set | 83.5 | **94.7** |
| Clustered F1 at threshold 0.5 | 0.92 | **71.5** |
| Scoring speed | instant | ~25 pairs/s |

(The picked-on-test row used a coarser threshold grid near the forest's optimum than near
Ditto's; the held-out row does not depend on it. Adjusted Rand equals clustered F1 / 100 on
these runs, so it is not a second piece of evidence.)

Ditto wins on precision, which is what closure rewards: at its best clustering threshold
(0.998) its accepted edges are 99.8% precise. It also degrades gracefully as the threshold
moves while the forest falls off a cliff. Its scores saturate near 1, so sweep thresholds up to
0.9995 — a grid that stops at 0.95 reports 88.1 instead of 94.7, which this repo did until it
was checked.

**Run it as a cascade with the forest as a veto only.** Ditto takes ~100 minutes to score 152k
pairs on a laptop. Letting the forest reject pairs it scores low and sending the rest to Ditto
scores **96.34 and 95.47** clustered F1 with the veto tuned on one half of the test people and
scored on the other - level with Ditto alone (96.38, 94.46) - while skipping 39% and 19% of
Ditto's work. (With everything picked on the full test set it skipped 56% and scored 95.4.)
Letting the forest also *accept* its confident pairs looks cheaper but loses 1–17 points held
out: each false edge it accepts goes straight into closure.

**Ditto mostly reproduces, but two of eleven runs collapsed silently** to "everything is a
match" - F1 sitting exactly on 2p/(1+p) - and Ditto's output gives no hint. It trains with zero
warmup steps, a known trigger; 10% warmup took Amazon-Google from 35.3 to 74.0 against the
paper's 74.67 ([chapter 10](10-reproductions.md)).

**Graph structure helps less than it first appeared.** Requiring *bridges* — edges whose removal
splits a component — to clear a higher bar adds 0.9 to the forest but only 0.1 to a well-tuned
Ditto. An earlier +3.9 was an artefact of the threshold grid above. At 0.665 false edges per
record nothing rescues the graph: precision first, structure second.

**Clustering recovers recall that blocking discarded.** Ditto's clustered recall is 93.3%
against an 82.1% blocking ceiling — closure infers pairs no blocking key ever proposed. That
suggests a precise matcher plus closure can make up for tight blocking; it was not tested
against wider blocking ([chapter 3](03-which-matcher.md) found wider blocking helped every pairwise method).

**The full process is in `scripts/customer_pipeline.py`** — contact profiling → blocking with
hub suppression → frequency-weighted features → classifier → clustering → household layer. On a
two-source pseudopeople file (20k records, 10,735 people, 4,305 households) it gets **98.53
clustered F1** held out (97.8 with the threshold picked on the test set) and **90.2 F1** on
households, with 22× fewer candidate pairs than a naive version that had no block-size cap.
Its accuracy gain over that naive version is about nil when both are tuned the same way (97.81
against 97.79): the contact handling buys fewer pairs and no junk chaining, not F1. It finds
the junk contacts perfectly, but that is true by construction - the file's junk values were
written by the same author as the junk patterns ([chapter 5](05-customer-data.md)).

**An LLM can replace the hand labels — on customer-style records.** Jev labelled all 15,444
training pairs of the customer pipeline for $0.41, 98.9% correctly; the forest trained on those
labels scores 97.99 clustered F1 against 98.53 with true labels. On historical_50k its labels are
91% right — precise but missing a quarter of the matches — and the forest falls from
84.68 to 74.09; loosening the labels with 300 hand-labelled pairs makes it worse, because false
match labels teach the forest to merge people. Check the LLM's label accuracy on a few hundred
of your own pairs first.

**On customer-style records the forest is nearly enough.** Ditto trained on the same pipeline
pairs lifts clustered F1 from 98.5 to 99.1 (held out) — under a point, against 11 on
historical_50k — and that recovers half of what perfect linkage could add to the merged
records: 90.4% → 90.9% all-fields-correct against a 91.5% ceiling (test customers only).

**Linkage is only half the job.** A golden-record test set (built by regenerating pseudopeople
with noise off) shows merging recovers about half the recoverable name errors — first_name
96.15% → 97.29% against a 98.57% ceiling — and that perfect linkage would add only about a
point of all-fields-correct records (1.1 on test customers only; the 0.9 in [chapter 7](07-golden-records.md) scores
clusters that include the forest's training records). Past ~98 clustered F1 the source data,
not the matcher, is the bottleneck.

**With no labels**, use Splink (Fellegi–Sunter). With its documented settings it scores 82.79
clustered F1 on historical_50k ([chapter 9](09-against-other-tools.md)); the 98.9 precision at 43.6 recall of [chapter 3](03-which-matcher.md)
is this report's own untuned Splink configuration and understates it. Seeding a matcher from
Splink's confident matches did not work ([chapter 9](09-against-other-tools.md): they are only the easy extremes).

**A zero-shot LLM can now match people — on the right records, with the right model.** On
genealogy-style records (historical_50k: drifting birth years, initials, no contact fields),
GPT-4o scores 67.2 F1 and GPT-4o-mini 53.8 against a random forest's 76.5 — they read a
changed birth year as a different person — and Llama-3.1-8B manages 3.1. GPT-6 Luna, a 2026
low-cost reasoning model, is the first to beat the forest: **84.7**, at $0.0001 a pair. On
customer-style records with addresses, emails and phones (synthetic, so no model can know the
people), GPT-4o reaches 95.3, Luna 97.1 and the forest 96.8. So a trained forest is as good
and free where records are rich, and a current reasoning model is the stronger zero-shot judge
of hard, contact-poor pairs - but in the pipeline itself that did not carry over (below). The
same model moved 28 points between two person tasks built the same way — measure on your own
records before choosing.

**Jev: use the decision model, not the router — and give it criteria.** Jev is TypeSafe's
decision model: typed questions in, probabilities out. Called directly as the matcher it ran all
seven test sets for $0.11 and sits at GPT-4o's level on products. Its options matter on people:
written criteria for what counts as the same person lift its F1 at the default cut-off by 6.1
points on historical_50k (95% interval +3.4 to +9.1), and criteria plus structured fields,
with a cut-off tuned on 330 of the 660 test pairs, reach **87.1** - the highest score on that
task here (Luna 84.7, the forest 76.5), for $0.016. The comparison favours Jev: 87.1 is the best
of four variants scored on the same pairs, and the forest had no tuned cut-off and training
pairs unlike the task's; trained in-distribution ([chapter 2](02-whole-customers.md)) the forest ranks the hard band
about as well (AUC 0.930 against 0.934). Extra questions in the same call (same household? placeholder
contact?) come free and do not hurt the match answer. Jev Router, the product that uses Jev to
pick another model per request, tied Luna on quality (±2 F1) at 4–10× the cost.

**But a zero-shot model does not belong in the clustering path.** Put Jev in the cascade instead
of Ditto — the forest vetoes, Jev judges the 67,630-pair uncertain band ($1.65) — and clustered
F1 falls to 83.0, below the forest alone (84.7), against 95.9 with Ditto (veto tuned held out). Its ranking across the
whole band (AUC 0.934 vs Ditto's 0.994) is not precise enough once transitive closure chains
records through every false edge. Best on 660 hard pairs, worst in the pipeline: benchmark the
matcher where it will be used.

### How custmatch compares, and when it does not

On a synthetic two-source customer file it reaches **97.3 clustered F1 from 1,000 labelled pairs**
(Splink's best: 84.4), **98.6 with active learning**, and households at **90.9 F1**:

| Compared on the same data | custmatch | Alternative |
| --- | --- | --- |
| Synthetic two-source customer file, 20k records, 1,000 labels: clustered F1 | **97.3** | Splink's best: 84.4 |
| Australian customers + web events, many sign-ups: precision / recall | 93.5 / **86.2** | CDP-style exact matching, tuned for precision: **95.7** / 64.3 |
| The same, few sign-ups | 91.3 / **82.4** | **92.8** / 56.3 |
| For contacting people, `"min_precision": 0.97`: records sent to someone else's email, many / few sign-ups | **0.05% / 0.08%** | CDP tuned for precision: 0.58% / 0.94% - while finding about 6 to 10 points more matches |
| Couples who separate: the leaver's new records merged with the ex | **0.4%** | CDP-style, tuned: 3.8% |
| North Carolina voters (names and places only), 200,000 people, 3 draws: clustered F1 | **95.6 (94.7-96.2)** | Splink, given the same labels: 93.9 |
| The synthetic file copied to 1,006,700 records, one laptop: clustered F1 / seconds / peak memory | **97.99** / 71.0 / **3.64 GB** | Splink 5, large blocks skipped: 78.88 / **21.6** / 5.96 GB |
| The synthetic file, 20,134 records: clustered F1 / seconds / peak memory | 97.38 / **5.4** / **0.35 GB** | dedupe: **97.75** / 499 / 8.5 GB |

The advantage is largest where customer-data problems exist; on clean, name-only data such as
the voter file the two are close, and Splink there learned from labels custmatch's own model
chose. Splink is faster once its large blocks are skipped; dedupe matches custmatch's accuracy
on the customer file but is about 90 times slower. On the other tools' own datasets custmatch
leads on noisy person data; on dedupe's organisation records dedupe wins one, and Splink the
other unless custmatch's `max_block` is raised as its warning advises ([chapter 9](09-against-other-tools.md)). The Australian rows are three
independently generated populations, against a CDP tuned on the truth - generous to it
([chapter 8](08-against-a-cdp.md), "Reassessed for marketing"). Three conditions:

- **custmatch needs good labels.** With 300 random labels and no active learning, or with 5% of
  labels wrong, its precision swung between datasets from the 70s to the high 90s - often worse
  than the CDP, which needs no labels. `summary.json` warns when labels are weak.
- **Its default threshold is tuned for overall accuracy,** and makes more wrong merges between
  strangers than the CDP (similar names, sparse records), though far fewer within households. For
  contacting people use `"min_precision"` and `"contact_confidence"`: safer than the CDP, at the
  cost of finding fewer matches.
- **Its threshold is judged on the randomly drawn labels.** With only a dozen matches among them
  the estimate is coarse; since 0.1.0 custmatch takes the middle of the thresholds those labels
  cannot tell apart (this fixed a voter draw that had dropped to 85.4), and `summary.json` warns
  below 30 random matches.

On that customer file it reaches **97.3 clustered F1 from 1,000 labelled pairs** (Splink's best:
84.4), **98.6 with active learning**, and households at **90.9 F1**. The synthetic and
Australian files were generated for this project, so their errors are ones the tool was designed
around - see the caveats in each chapter. On five benchmarks (four public, plus that synthetic file),
averaged over three label draws, it beats Splink's unsupervised result on four and is 0.3 behind
on febrl3 (measured before 0.1.0's threshold rule). A million records take about 1.7 minutes on a laptop (94 s with `"blocking":
"fixed"`), and ten million 37 to 57 minutes and 34 GB on a 16-core cloud machine - measured
on copies of a 20,000-record file, so blocks grew less than they would in a real file of that size.
