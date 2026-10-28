# 13. What is still open

Updated October 2026. Of the next steps this section first listed, the NCVR benchmark has since been
run on the Leipzig files up to all 5 million records, LLM labelling was tested ([chapter 4](04-labels.md)), and
blocking became a research thread of its own ([chapter 6](06-blocking-and-scale.md), "Blocking, and what Splink, dedupe and Zingg
taught" and "Larger files"). What remains:

1. **Validate on a real customer file.** Every result here and in the guide is on public or
   generated data, and the generated files' errors were written by the same author as the fixes
   that handle them. One real file, with a few hundred labelled pairs, would test more than any
   further benchmark.
2. **Blocking at millions of records.** On all 5 million NCVR records the keys chosen on a
   200,000-person sample stopped working - blocks outgrow the size limit and true pairs are never
   compared - and clustered F1 fell from 95.6 to 84.5; splitting oversized blocks and more random
   labels won back 2.9 points ([chapter 6](06-blocking-and-scale.md), "Larger files"). Splitting
   on several fields at once (`"blocking": "adaptive"`) now lifts the full file's blocking
   ceiling to 99.05 ("Where the rest of the gap is"), so blocking is no longer the main limit:
   the full five-million run with it scored 88.45, its recall 90.2. It is now the default where
   the default keys would leave over 1% of the records out, and random labels are drawn stratified
   (since 0.1.3 on every file), which lifted a million people with 300 random labels from 80.84
   to 92.88. Open: the ten points between the full run and its ceiling, mostly precision; why the
   full run with stratified labels (86.24 with nothing set, 300 random) found only 10 random
   matches, no more than plain draws of 300 (11 and 13); and why stratified labels cost historical_50k 1.41
   points with active learning (three draws) while helping with random labels alone and on
   every synthetic address book.
3. **Real drift: NCVR from the raw snapshots.** The Leipzig benchmark has real names but synthetic
   errors in four fields. Pairing two of the State Board of Elections' snapshots by NCID would give
   real moves, name changes and typos - the contact-poor case where [chapter 2](02-whole-customers.md) found a transformer
   worth 11 points. It is real personal data: not downloaded, pending a decision to.
4. **Test whether tight blocking plus closure beats wide blocking** ([chapter 2](02-whole-customers.md)'s suggestion, from
   clustered recall exceeding the blocking ceiling), and whether chaining risk grows with records
   per customer ([chapter 5](05-customer-data.md)'s hypothesis, from two files that differ in many ways at once).
5. **LLM candidate selection** (ComEM, COLING 2025): choosing the match among a blocked set rather
   than judging pairs one at a time. Untested.
6. **Close the GPT-4 row, if the reproduction table needs it.** GPT-4o and GPT-4o-mini reproduce and
   Luna beats the published GPT-4 row, so this only completes the table. `gpt-4-0613` retires
   2026-10-23 and costs ~$21 for the five product sets; `llm_zeroshot_openrouter.py --model
   openai/gpt-4 --budget 25` runs it. Not run: it spends money for no new finding.
7. **Real-time matching at scale.** Built, and replayed against batch runs on the demo ([chapter 12](12-the-service.md));
   a replay at millions of records is next.

For the service:

- **The compact index at ten million records:** its memory and rebuild time, and lookup time with
  blocks near `max_block`, need measuring before committing to a latency target.
- **Warehouse sync and export** are not built, and the format is open: plain Parquet (simple,
  easy to erase) or Iceberg (warehouses read it in place, but needs a catalog and the three-step
  erasure in [chapter 12](12-the-service.md)).
- **How far the online path should go:** scoring and joining only, with guards and re-cuts left to
  batch, is simplest.
- **Run retention:** how many previous runs to keep for rollback, against the copies erasure must
  reach.

---

[← The service](12-the-service.md) · [Contents](README.md)
