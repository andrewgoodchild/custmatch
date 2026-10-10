# 12. The service

custmatch-server runs the library as a self-hosted service: one database, the batch run on a
schedule, real-time matching over HTTP and a steward dashboard (server/README.md says how it
works). This chapter is what building it found: why one database, what real-time matching costs and
where it differs from batch, what stewards' decisions are worth, and how the dashboard compares with
Splink's charts and the MDM products. Scripts: `server/scale_test.py`, `server/replay_test.py`,
`server/learning_test.py`, `tests/test_server_rules.py`, and the proofs of concept listed at the end.

**At a million records** (`server/scale_test.py`) the real-time matcher starts in 6 s using 0.46 GB
more, matches a record in 24 ms (median) and looks one up in 7 ms. **Stewards' decisions train the
next run's model** (`server/learning_test.py`): 80.0 -> 88.1 clustered F1 over three review rounds on
the demo, against 84.6 when they only override their own pairs.

## One database, not DuckDB

An earlier version read results through DuckDB, in memory in each process, over Parquet files on S3
published by a manifest. A critique found it the weakest part:

- **The serving workload is point lookups** ("which customer is record X in?", "show customer Y"),
  which a column store over remote Parquet answers slowly and with S3 requests per query, and which
  a database with indexes answers directly.
- **Each process loads its own copy**, so memory and reads multiply with workers; and readers must
  notice a new manifest and switch, which is custom code a database does for free.
- **A DuckDB database file is locked by its writer** - a second process could not open it even
  read-only (`research/poc/storage_concurrency.py`) - so it can never be the shared state.
- **The manifest has no lock.** Two batch runs could race to publish; a transaction in the catalog
  table cannot.
- **A database was needed anyway** for stewards' decisions.

DuckDB stays useful outside the service: an analyst can query the exported Parquet or Iceberg files
with it. Nothing in the service depends on it.

## Real-time matching

**Measured** (`research/poc/realtime.py`, the sparse Australian demo, 176 records arriving one at a time on
1,590): 21 ms median, 34 ms at the 95th percentile, 55 ms worst, on a laptop. `custmatch add` for one
record took 0.18 s and grows with the file - fine for batches, not for real time.

What it has to handle, and what was measured:

1. **One writer, enforced.** Without it, two simultaneous records of one new person each saw no
   candidates and each started a customer; with it, one customer. The matcher takes a **lease** - a
   Postgres advisory lock, or a lock row in SQLite - so exactly one process writes. Lookups
   (`write=false`) read the database and keep working if the matcher is down.
2. **A compact, rebuildable index.** Built: blocking keys are 64-bit hashes in a sorted numpy
   array beside their record numbers, the records one columnar DataFrame. At a million records the
   first (dict) version took 23.5 s and 1.37 GB to start; this one 6.0 s and 0.46 GB, and a match
   24 ms median (`server/scale_test.py`). The rebuild time is the matcher's recovery time.
3. **One definition of blocking keys** for batch and real time. Built: `core.blocking_families` is
   used by both, and a test checks the hashed keys give exactly the batch blocker's candidate pairs
   (the server's first copy of the keys had already drifted: it missed the LSH keys).
4. **Features depend on file-wide statistics.** Scoring on a small frame changed name rarity by up to
   6.8 until `rich_features` took the file's record count (`n_records`). What remains is inherent:
   contact weights (up to 0.65) and name rarity (up to 0.45) are scored with the statistics as they
   stand when a record arrives, and batch with the statistics after every record has arrived.
5. **Real-time answers are provisional.** After all 176 arrivals, 95.6% of records were in the
   customer a batch run gives (`custmatch add`, one batch: 96.4%). The batch run is the truth; when it
   moves a record, the change reaches users as a merge or split in the history table, with stable IDs,
   and the API marks real-time decisions as provisional until a batch run confirms them.
6. **The batch run's rules, online.** Built: stewards' overrides (`POST /overrides`, stored in the
   database and read by the batch run too): a "same" merges the two customers at once, a
   "different" stops any later merge across it, and a customer already holding both waits for the
   batch run to split it - the API says so. Trusted IDs join, the twin guard caps a twin's score,
   and a record never joins or merges a customer holding another record of a duplicate-free
   source. A "different", a duplicate-free source and a merge's own conflicts win over a "same", as
   in batch. Merges are stored (`realtime_merges`), replayed when the matcher restarts, and applied
   to every customer ID the API returns until a batch run reconciles them; `tests/test_server_rules.py`
   checks each rule now and after the next batch run. Left to the batch run: cookie stitching and
   `max_cluster` re-cuts. (Building this found `trusted_ids` silently ignored by the library unless the
   column was also an extra field; fixed.)
7. **Merging on one record's say-so over-merges.** `server/replay_test.py` runs a batch over half
   the demo, sends the other half through the matcher one record at a time, then runs a batch over
   everything (`server/replay_test.py --all`, `research/results/server_replay.json`, five seeds; pairwise
   F1 against the true people, the batch run 81.91). Merging every customer a record links to -
   one record then bridges two people - gave 80.46; merging only at 0.9 or more ("strong", the
   default) 81.67, with 86.78 agreement with the batch run; never merging 81.15. The test suite
   runs the replay with floors on both.
8. **Joining on one link chains people.** A record joined the customer of its best link, so one
   strong link to one record of a customer - often a sparse one, a "joy santos" with no birth
   date - pulled it into a customer whose other records were other people. Now the record is
   scored against the customer's records (up to 25) and joins only if it matches at least half
   of them; otherwise the next customer, or a new one. With "strong" merging, real-time precision
   went from 86.60 to 89.98 for recall 75.69 to 74.80, F1 80.75 to 81.67 - better on four of
   five seeds (0.11 lower on one), and the gap to the batch run from 1.16 to 0.24. A two-record
   customer still joins on one of its two: requiring more than half lifted precision to 92.81
   but cut recall to 69.45 (F1 79.44). Also rejected in a trial: a mean-score rule and a veto on any clear non-match.
   About 18 ms a record either way.

## Erasure from exports

A warehouse export (Parquet or Iceberg) is not built; this is what one would have to handle.
For Iceberg exports, a delete is not enough: after deleting a person's rows, time travel to the
earlier snapshot still returned them, and after expiring snapshots the original data file - still
holding the rows - stayed on disk (`research/poc/storage_concurrency.py`). Erasing from an export means
deleting the rows, expiring the snapshots **and removing data files no longer referenced**; pyiceberg
0.12 does the first two, the service the third. Plain Parquet exports are simpler: rewrite them, or
keep only the latest.

## Memory and the Arrow-strings question

pandas 3 already stores text as Arrow strings: the loaded table is about 200 MB per million records,
and converting it explicitly saved nothing (`research/poc/arrow_strings.py`). `pyproject.toml` does not pin
pandas, so a user on pandas 2 gets Python-object strings and more memory; the service image should
pin pandas 3. The batch run's memory is in its working tables (blocking, features, model scores,
outputs) - see the report, "Larger files".

## The dashboard against Splink's charts and the MDM products

Against Splink's charts and dashboards (its gallery's fourteen; what each is for, and ours):

| Splink | Here |
| --- | --- |
| Waterfall chart (each comparison's share of a pair's match weight) | A waterfall in the review queue, on every link of a customer, and on sampled links: the forest's score decomposed along each tree's path (base rate plus each feature's share, which add up to the score; `GET /explain`) |
| Cluster studio (a cluster's network; sample clusters by size or structure) | A customer's records and links drawn as a network (score as width and colour, bridge links dashed, click a link for its waterfall); **Customers** samples the largest, those with the weakest link, the most bridges, the sparsest links, or at random |
| Comparison viewer (example pairs across the score range) | **Links**: a random sample of accepted links in any score band; the review queue holds the uncertain ones |
| Match weight histogram | Candidate pairs by score with the threshold (**Model & data**) |
| Accuracy charts, threshold selection | Estimated precision, recall and F1 against the threshold (cross-fitted, closure-aware: the run now saves the whole sweep) |
| Match weights, m and u parameters | Feature importance (a forest has no m and u; the waterfall is the per-pair view) |
| Parameter estimate comparisons | **Runs**: each run's threshold, estimated precision and recall, labels, and its model's feature importances beside the previous run's |
| Term-frequency adjustment | What agreeing on a name is worth: the rarity feature, ln(records / records with the name), for the commonest, a middling and the rarest names; contact weights (junk counts for nothing, shared for little) |
| Unlinkables | A sample of records each scored against an exact copy of itself: the share that cannot reach each threshold even then (1.8% of the demo at its threshold) |
| Completeness, profile columns | Completeness by source and field; per field the distinct values, the top value's share and the share of values on one record; the most common values (placeholders stand out) |
| Cumulative comparisons from blocking rules | Each blocking key's candidate pairs and how many no earlier key proposed (exact up to 50 million pairs; on the demo, email blocking adds 1 pair the other keys missed); blocks skipped as too large; records never compared |

Splink's charts are files a data scientist opens after a run; these are views in the running
service, next to the decisions they inform.

Against the stewardship features of MDM and matching products, from their own documentation
(Informatica MDM, Reltio, Semarchy xDM, Profisee, Tamr, Senzing, IBM Match 360, Ataccama ONE;
Informatica, IBM and Profisee from page excerpts and marketing pages where the manuals were not
reachable), the most common, and ours:

| Feature (products offering it) | Here |
| --- | --- |
| Potential-match queue, pair review that trains the model (all; TAM, IBM train) | Review; decisions train the next run (`server/learning_test.py`) |
| Manual merge of any two entities (7) | Merge with customer: previewed, applied now, kept by later runs |
| Unmerge / split a record (7) | Split out: the record becomes its own customer now; "different" decisions keep it apart later |
| Merge preview (INF, ATA, IBM) | Every merge, split and decision previewed before it is made |
| Side-by-side comparison with confidence (5+) | Review, Links, Compare |
| Attribute lineage; pin / override a golden value (4-5) | Lineage per golden field; Pin, which beats survivorship in every later run |
| Why matched; why not matched (SEN, REL, INF, ATA) | Waterfall; Compare says whether blocking would ever compare two records |
| Remembered not-a-match, with undo (REL, TAM, ATA, IBM) | Decisions persist; revoke from the Decisions tab |
| Task claiming (INF, REL, TAM, ATA) | Review pairs shown to a steward are held from others for 15 minutes |
| Approval / four-eyes with comments (REL, INF) | `CUSTMATCH_FOUR_EYES=on`; notes on every decision, split, merge and pin |
| Queue filters and bulk actions (REL, TAM, ATA) | Source, model's call and score filters; list mode with bulk decisions |
| Search; find possible matches for a typed-in person (SEN, REL) | Search; Find possible matches (`POST /possible`, saves nothing) |
| Relationship / graph view (REL, INF, IBM, SEM) | The customer's network of links; its household |
| History and audit (REL, INF, SEM) | The customer's run-by-run history and every steward action on its records |
| Steward activity (REL) | Per steward: decisions, overturn rate, approvals, revocations, splits, pins; decisions per day; CSV export |

Not built: priorities and due dates on tasks, notifications to stewards, editing a source record's
values in place (corrections belong at the source), saved searches.

## The proofs of concept

| Script | Shows |
| --- | --- |
| `research/poc/storage_concurrency.py` | DuckDB file locking; partial reads by glob against a manifest; concurrent Iceberg appends; Iceberg erasure leaving data files |
| `research/poc/realtime.py` | Online matching latency; candidate and feature equivalence with batch; consistency after arrivals; the duplicate-customer race and the one-writer fix |
| `research/poc/arrow_strings.py` | Table memory with object and Arrow strings; behaviour of string operations |

---

[← What the tool adopted, and what it dropped](11-adopted-and-dropped.md) · [Contents](README.md) · [What is still open →](13-open-questions.md)
