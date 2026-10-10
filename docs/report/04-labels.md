# 4. Labels

Labels are the most important input. Language models can label pairs for a fraction of the
cost - but only where their labels are near-perfect, which is why custmatch gates them on a
check against your own labels.

## Can an LLM replace the hand labels? Only where its labels are near-perfect

`scripts/jev_labels.py`. The use of LLMs the recent literature supports (Steiner & Bizer 2026) is
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
removed; the code lives on in `scripts/jev_labelling.py` for the scripts that produced these
numbers.

---

[← Which matcher](03-which-matcher.md) · [Contents](README.md) · [Customer data: fake contacts, households and sparse records →](05-customer-data.md)
