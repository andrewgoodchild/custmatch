# 7. Golden records

Matching is only half the job: the golden record must say the right thing. How much a merged
record recovers, and why the source data, not the matcher, becomes the limit.

## Golden records: does the merged record say the right thing?

`research/scripts/golden_record.py`.

Every benchmark in chapters [2](02-whole-customers.md)-[10](10-reproductions.md) — and every benchmark in the literature surveyed in [chapter 1](01-data.md)
— scores only **linkage**: which records belong to the same customer. None of them scores
**survivorship**: given a cluster of noisy records, what should the surviving record contain?
That is half of what a customer master actually does, and it had gone untested here.

pseudopeople closes the gap, because it can regenerate the identical population with noise
disabled (`config=psp.NO_NOISE`), giving the true value of every field for every person.

### How much is actually recoverable

| Field | Raw records correct | Best survivorship rule | Ceiling (true value present in the cluster) |
|---|---|---|---|
| first_name | 96.15 | **97.29** | 98.57 |
| last_name | 97.05 | **98.13** | 98.91 |

With perfect linkage handed in, merging recovers about **half** of the available error on
names — first_name 96.15 → 97.29 against a ceiling of 98.57 — and the remaining ~1.4% is
unrecoverable because *no record in the cluster holds the true value*. No survivorship rule
can invent information that is not there.

### Why modal voting underperforms here

| Strategy | All fields correct (oracle clusters) | first_name | last_name |
|---|---|---|---|
| most_complete (modal, then trusted source) | 90.38 | **97.29** | **98.13** |
| most_common (modal) | 90.34 | 97.26 | 98.10 |
| longest | 89.73 | 97.17 | 97.82 |
| first_seen / source_priority | **91.38** | 96.66 | 97.79 |

This file has **1.9 records per person**. With two records and a disagreement there is no
majority to take — modal voting degenerates to a coin flip, which is why it beats
first-record-wins on names by only ~0.6 points. Survivorship rules earn their keep when
entities carry many records; on a two-source file they barely matter.

Two honest caveats on that table:

- `first_seen` and `source_priority` are identical here because the CRM rows are concatenated
  first, so the "first" record is always the CRM one.
- They lead on *all fields correct* only because of how I defined truth: the truth table takes
  the census (residential) address, while billing holds a mailing address that legitimately
  differs. Any rule preferring CRM is therefore favoured on the two address fields. **The name
  fields are the clean comparison**, and there the modal rules win.

### Linkage error vs survivorship error

| | All fields correct |
|---|---|
| Predicted clusters (our pipeline, 97.8 clustered F1; includes the forest's training records) | 90.46 |
| Oracle clusters (perfect linkage) | 91.38 |

Perfect linkage would buy **0.9 points** of golden-record accuracy (1.1 on test customers
only, [chapter 5](05-customer-data.md)'s Ditto run: 90.41 against 91.52). The other ~8.6 points of
error are noise in the source data that survivorship cannot undo. That is worth knowing before
spending another month on the matcher: at this level of linkage quality, **the matcher is no
longer the bottleneck — the source data is.**

That was a bound; [chapter 5](05-customer-data.md)'s Ditto run lets it be measured with a real, better matcher. On the
pipeline's test customers (`customer_ditto.py golden`, the same survivorship code):

| Clusters from | All fields correct | first_name | last_name |
|---|---|---|---|
| Random forest (98.53 held-out clustered F1) | 90.41 | 96.75 | 97.65 |
| Forest veto, then Ditto (99.14) | 90.93 | 97.05 | 97.90 |
| Perfect linkage | 91.52 | 97.37 | 98.08 |

Ditto recovers **half of what perfect linkage could add** — +0.5 of a possible +1.1 points —
for a fine-tuned transformer and ~45 minutes of laptop compute. The bound holds with a real
matcher: past ~98 clustered F1, a better matcher moves the merged record by tenths of a point,
and the ~8.5 points of error left are in the source data.

---

[← Blocking and scale](06-blocking-and-scale.md) · [Contents](README.md) · [Matching for marketing: against a CDP →](08-against-a-cdp.md)
