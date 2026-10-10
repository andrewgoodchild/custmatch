# 2. Match whole customers, not pairs

The central finding. A matcher scores pairs of records; a customer file needs groups. Joining
every accepted pair into customers (transitive closure) turns a few false links into giant
customers, so the threshold has to be chosen - and judged - on the customers it produces.

## RF vs Ditto on person records, and what clustering does to both

`scripts/person_em_build.py`, `scripts/person_ditto.py`, `scripts/person_head_to_head.py`.

The person baselines ([chapter 3](03-which-matcher.md)) and the reproductions ([chapter 10](10-reproductions.md)) left two gaps: no transformer had been run on person data, and every number was
pairwise, while a customer file needs *clusters* (golden records). This closes both.

**Setup.** historical_50k, split by person so no individual appears in two splits: 60% of
clusters for training, 15% validation, 25% test. Records are serialised in Ditto's own
`COL … VAL …` format over first name, surname, DOB, birth place, postcode, occupation and
gender. Train 12,000 pairs, validation 3,000, and the test split keeps **every** blocked
candidate pair among its 12,880 records — 152,772 pairs covering 77,704 true pairs across
1,289 people, an 82.1% blocking ceiling. Both models see identical pairs. Ditto is the
authors' `DittoModel`/`DittoDataset` with 10% warmup added, after what [chapter 10](10-reproductions.md) found.

### The finding: your pairwise threshold will destroy your clusters

The clustering pass takes every edge above a threshold, forms connected components, and scores
the resulting partition. For the random forest:

| Threshold | Pairwise P | Pairwise R | Pairwise F1 | Clustered F1 | Clusters | Largest | ARI |
|---|---|---|---|---|---|---|---|
| 0.30 (best pairwise) | 93.4 | 77.7 | **84.8** | **0.45** | 742 | **8,200** | 0.003 |
| 0.60 | 97.2 | 72.4 | 83.0 | 1.97 | 1,259 | 3,818 | 0.018 |
| 0.75 | 98.6 | 66.6 | 79.5 | 39.8 | 1,695 | 421 | 0.397 |
| 0.85 | 99.1 | 60.4 | 75.1 | 68.3 | 2,103 | 195 | 0.683 |
| 0.90 | 99.3 | 55.8 | 71.5 | 78.8 | 2,405 | 127 | 0.788 |
| 0.95 | 99.6 | 48.0 | 64.8 | **83.5** | 3,033 | 36 | **0.835** |

At the threshold that maximises pairwise F1, the matcher is **93.4% precise** — and transitive
closure still collapses **8,200 of 12,880 records into a single blob**. Clustered F1 is 0.45.
The model is not bad; the pipeline is. At 6.6% false-positive edges, a graph this dense is
past its percolation point, and a handful of wrong links is enough to chain unrelated people
together through shared surnames and birth years.

Getting usable clusters costs 20 points of pairwise F1: threshold 0.95, where pairwise F1 falls
to 64.8 but clustered F1 reaches 83.5 and the largest cluster is a plausible 36 records.

Three things follow, and they are the most directly useful results in this report:

1. **Tune the threshold on the clustering metric, not the pairwise one.** They disagree by
   0.65 of threshold here, and the pairwise-optimal choice is catastrophically wrong.
2. **Precision is worth far more than recall once you cluster.** Recall lost to a high
   threshold costs you linearly; precision lost costs you super-linearly through closure.
3. **Connected components is a bad clusterer at any threshold you would choose for matching.**
   It has no notion of cluster cohesion, so one bad edge merges two entities permanently. This
   is why production systems use size caps, correlation clustering, or hierarchical merging
   (custmatch uses a size cap and the clustered threshold; it tried correlation clustering and
   dropped it) —
   and why "we got 95% precision" is not by itself evidence that a customer file will come out
   clean.

### RF vs Ditto, head to head

Same pairs, same splits, same clustering pass.

| | RF pairwise | Ditto pairwise | RF clustered (best) | Ditto clustered (best) |
|---|---|---|---|---|
| Precision | 93.4 | **98.2** | 96.4 | 96.1 |
| Recall | 77.7 | **80.1** | 73.7 | **93.3** |
| F1 | 84.8 | **88.2** | 83.5 | **94.7** |
| Adjusted Rand | — | — | 0.835 | **0.947** |
| Largest cluster | — | — | 36 | 50 |

**The transformer wins on person records, and by far more than the pairwise numbers suggest.**
+3.4 F1 pairwise, **+11.1 clustered**, +0.112 ARI. It earns that through precision at the top of
its score range: at threshold 0.998 Ditto's accepted edges are 99.84% precise, and on a graph
that gets closed transitively, that difference compounds.

**Correction.** An earlier version of this section reported Ditto's best clustered F1 as 88.1.
That was the top of a threshold grid that stopped at 0.95. Ditto's scores saturate near 1 — 40%
of all pairs score above 0.95 — so the useful thresholds are 0.995–0.9995, and the grid never
reached them. With the tail included the optimum is **94.66 at 0.998**. The forest's optimum
(0.95) was inside the old grid and is unchanged; above it the forest only loses recall.

The threshold sensitivity is the bigger practical gap:

| Threshold | RF clustered F1 | Ditto clustered F1 |
|---|---|---|
| 0.05 | 0.24 | 26.7 |
| 0.30 | 0.45 | 62.9 |
| 0.50 | 0.92 | 71.5 |
| 0.70 | 20.0 | 77.6 |
| 0.90 | 78.8 | 84.3 |
| 0.95 | **83.5** | 88.1 |
| 0.99 | 70.2 | 92.7 |
| 0.998 | 58.1 | **94.7** |
| 0.9995 | 57.4 | 94.1 |

The forest is unusable as a clusterer below 0.7 and works only in a narrow band around 0.95.
Ditto degrades gracefully and is already at 62.9 where the forest is at 0.45. If you cannot tune the threshold
carefully — and in production, on data whose error profile drifts, you often cannot — that
robustness is worth more than the headline F1.

### Clustering recovers recall that blocking threw away

Look at Ditto's clustered recall: **93.3%** at its best threshold (95.0% at 0.95), against a
blocking ceiling of **82.1%**. That is
not an error. Transitive closure infers pairs that blocking never proposed — if A–B and B–C are
both candidates and both match, A–C is recovered even though no blocking key ever put A and C
together.

So the [chapter 3](03-which-matcher.md) conclusion — that blocking recall is a hard ceiling — is only true pairwise.
With a clustering pass, a precise matcher climbs back over it: 11 points of recall here that
blocking had discarded. That inverts the usual advice. Rather than widening blocking (more
candidate pairs, quadratic cost, more chances to make a false link), it can pay to keep
blocking tight, run a high-precision matcher, and let closure do the reaching.

### What this costs

The forest trains in seconds on CPU and scores 152,772 pairs instantly. Ditto took ~30 minutes
to train and **100 minutes to score the same pairs** on this laptop — about 25 pairs/second.
For a real customer file with millions of candidate pairs that is not viable without a GPU.

### The cascade, measured

`scripts/cascade.py`. The obvious production shape is a cascade: score every pair with the
forest, and spend Ditto only where the forest is unsure. Every pair with forest score `rf`
is **rejected** if `rf <= lo`, **accepted** if `rf > hi`, and otherwise sent to Ditto; the
resulting edges are clustered exactly as above. It needs no new training or inference — both
models' scores on the identical 152,772 test pairs are already on disk.

The forest is a good rejector and a poor acceptor. Scores of 0.01 or less cover 39% of pairs
and hold only 628 of 63,824 true candidate pairs. But even above 0.99 it accepts 50 false edges,
and closure turns each one into a merge. Picked on the full test set:

| Pairs sent to Ditto | lo | hi | Clustered F1 | ARI |
|---|---|---|---|---|
| 0% (forest alone) | — | — | 83.53 | 0.835 |
| 19.8% | 0.2 | 0.95 | 89.47 | 0.895 |
| 24.6% | 0.2 | 0.98 | 92.15 | 0.921 |
| 29.7% | 0.2 | 0.995 | 93.52 | 0.935 |
| **44.3%** | **0.2** | **never** | **95.40** | **0.954** |
| 100% (Ditto alone) | — | — | 94.66 | 0.947 |

Those are picked on test, like the rest of this section, so they are optimistic. The honest
check splits the test people into two halves, tunes the cascade *and* Ditto-alone's threshold on
one half, and scores both on the other:

| Cascade, tuned on the other half | Sent to Ditto | Cascade F1 | Ditto alone F1 |
|---|---|---|---|
| Forest only rejects (A→B) | 61.3% | 96.34 | 96.38 |
| Forest only rejects (B→A) | 81.1% | 95.47 | 94.46 |
| ≤ 30% budget (A→B) | 28.2% | 95.14 | 96.38 |
| ≤ 30% budget (B→A) | 27.8% | 91.92 | 94.46 |
| ≤ 20% budget (A→B) | 19.9% | 90.52 | 96.38 |
| ≤ 20% budget (B→A) | 18.9% | 77.32 | 94.46 |

Three conclusions:

1. **Use the forest as a veto, never as an acceptor.** Letting it only reject matches Ditto
   alone out of sample (−0.04 and +1.0) while skipping 19–39% of Ditto's work held out (56%
   picked on the full test set, where the veto is `rf <= 0.2`): 39% on half B with the same veto; 19% on
   half A, where tuning picked `lo = 0` (reject only pairs the forest scores exactly 0). The small gain
   on the full set is an ensemble effect: the veto removes false edges Ditto would have kept.
2. **Letting the forest accept costs points, and the cost is unstable.** At a 30% budget the
   held-out loss is 1.2–2.5 points; at 20% it is 6–17, depending on which half you tune on.
   Every auto-accepted false edge goes straight into closure with no second opinion.
3. **The saving is real but bounded:** about 45 minutes of Ditto instead of 100 on this
   laptop. The cascade makes Ditto cheaper; it does not make it cheap.

### Can a zero-shot model replace Ditto in the band? No

`scripts/jev_cascade.py`. [Chapter 10](10-reproductions.md) found Jev — TypeSafe's decision model, with written
criteria and structured fields — at 87.1 F1 on 660 hard historical_50k pairs, ahead of Luna and
the forest. The obvious next step is a pipeline with no deep-learning training at all: the forest
vetoes as above, and Jev judges the 67,630-pair band instead of Ditto. Jev scored the whole band
for $1.65 (~$0.000024 a pair, ~17 minutes at 16 parallel calls). Same pairs, same clustering;
thresholds are tuned on one half of the test people and scored on the other:

| Clustered F1, 152,772 pairs | Picked on test | Held out (A→B / B→A) | Mean held out |
|---|---|---|---|
| Forest veto, then Ditto (above) | **95.40** | 96.34 / 95.84* | **96.09*** |
| Forest veto, then mean of Jev and Ditto | 94.53 | 96.50 / 94.86 | 95.68 |
| Ditto alone | 94.66 | 96.38 / 94.46 | 95.42 |
| Forest veto, then mean of forest and Jev | 82.26 | 90.71 / 82.86 | 86.78 |
| Forest alone | 83.53 | 88.30 / 81.06 | 84.68 |

| Forest veto, then Jev | 74.89 | 88.42 / 77.51 | 82.97 |

\* This script fixes the veto at `rf <= 0.2`, the value picked on the full test set, so its
"held out" veto rows include one test-picked parameter. With the veto also tuned on the other
half (the table above) the B→A cell is 95.47, not 95.84.

**It does not work.** On its own in the band Jev is *worse* than the forest it replaces; averaged
with the forest it gains two points with a wide swing between halves; averaged with Ditto it adds
nothing. The ranking explains it: across the band Jev's AUC is **0.934**, the forest's 0.930 and
Ditto's **0.994**. That gap looks modest pairwise, and it is exactly the gap this section's first
table shows closure punishing — every false edge Jev passes can chain two people together, and
over 67,630 pairs a 0.93-AUC scorer passes thousands.

So the two results do not conflict. The 660-pair task measures pairwise F1 on a balanced set of
deliberately hard pairs; this measures what the pairs do to a customer file. **A zero-shot model
that is best on a hard-pair benchmark can still be the worst choice inside the pipeline.** For
clustering, the matcher has to be precise across the whole band, and here only the model trained
on this data is. That is the section's opening lesson again — precision first — measured on an LLM.

## The match graph in Kuzu, and structure-aware clustering

`scripts/kuzu_graph.py`. The person-record graph — 12,880 records, 152,772 candidate edges —
loaded into an embedded Kuzu database, each edge carrying its RF and Ditto probability, all 12
similarity features, and the ground-truth label for audit.

**The port is faithful.** Kuzu's `algo` extension runs weakly-connected-components over a
filtered projection, so the match threshold becomes a `WHERE` clause. It reproduces the scipy
numbers exactly: 0.45 clustered F1 at 0.30, 83.53 at 0.95.

```
CALL PROJECT_GRAPH('G', {'Record': 'n.rec_id >= 0'}, {'CANDIDATE': 'r.rf_prob > 0.95'})
CALL weakly_connected_components('G') RETURN node.rec_id, group_id
```

### What the graph store bought that scipy did not

**A diagnosis.** One query over accepted edges, grouped by surname, found the cause of the
giant component:

| Surname | Accepted edges |
|---|---|
| *(blank)* | 3,801 |
| baronet | 1,320 |
| moore | 425 |
| jones | 350 |

A blank surname and the title "baronet" — an occupation leaking into a name field — generate
more accepted edges than the four commonest real surnames combined. That is the same
hub/junk-value pattern as §8, found here by a two-line query instead of a re-run.

**Provenance on any merge.** Every wrong merge can be inspected with its features attached:

| Record A | Record B | prob | surname_sim | dob_exact |
|---|---|---|---|---|
| charlie barkley 1759-01-01 | rachel barkley 1743-01-01 | 0.89 | 1.00 | 0 |
| henry young 1842-02-01 | george young *(no dob)* | 0.86 | 1.00 | 0 |

Same surname, different first name, conflicting or missing DOB — siblings and spouses, exactly
the household-adjacent confusions from §8. In production this is what makes a targeted unmerge
possible: you can find the single edge responsible rather than rebuilding the cluster.

### Bridges: clustering that uses graph structure

A **bridge** is an edge whose removal splits a component — a single point of failure, where one
wrong link merges two whole groups. That is precisely the percolation mechanism. So: build the
graph at a permissive threshold, but require any *bridge* to clear a much higher bar, and
iterate (removing bridges creates new ones).

Both rules tuned on the same fine grid (`scripts/cascade.py`):

| Model | Best plain connected components | Best bridge-aware | Δ |
|---|---|---|---|
| Random forest | 83.53 (th 0.95) | **84.39** (base 0.90, bridges ≥ 0.995) | +0.9 |
| Ditto | 94.66 (th 0.998) | 94.80 (base 0.998, bridges ≥ 0.999) | +0.1 |
| Cascade, forest veto | 95.40 (th 0.995) | 95.51 (base 0.98, bridges ≥ 0.999) | +0.1 |

**Correction.** This section first reported bridges lifting Ditto from 88.10 to 92.04. That
compared a bridge rule allowed to demand 0.99 against a plain threshold capped at 0.95, so most
of the "gain" was just a stricter threshold. Tuned fairly, bridges are worth about 0.1 on Ditto
and 0.9 on the forest. The idea is sound — the edges it cuts really are the ones chaining
people together — but a well-calibrated matcher with a well-chosen threshold has already
dropped most of them. Bridge-aware clustering helps a weaker matcher a little; it is not a
substitute for tuning the threshold on the clustering metric.

### The limit: structure cannot rescue a low-precision matcher

At the pairwise-optimal threshold 0.30, where the RF has 0.665 false edges per record, nothing
worked:

| Fix | Clustered F1 | Largest cluster |
|---|---|---|
| Plain connected components | 0.45 | 8,200 |
| Bridge-aware | 0.73 | 6,353 |
| Hub suppression (blank/baronet) | 1.10 | 4,732 |
| Both together | 1.44 | 4,138 |

Only 324 of the accepted edges were bridges. Above the percolation threshold the false edges
form a dense web, not a chain, so there is no small set of cuts that separates it. The same
fix applied to Ditto at the same base threshold gives 79.8, because Ditto's false-edge density
is far lower.

**So the ordering is: precision first, structure second.** Get the false-edge density below the
percolation point — which is a matcher and blocking problem — and only then does structure-aware
clustering add anything — and on a well-tuned matcher that is under a point. Applied the other
way round it buys nothing.

---

[← Data, and the work it builds on](01-data.md) · [Contents](README.md) · [Which matcher →](03-which-matcher.md)
