# custmatch-server: operator guide

custmatch-server runs custmatch as a service next to your data: it keeps your customer records
and their loads in one database, matches them in batch on a schedule, answers "who is this?" in
real time, and gives data stewards a dashboard to review and correct matches. One image, one
database (Postgres in production, SQLite for a trial). How it works is below; what it was
measured at is in the report's chapter on the service
([docs/report/12-the-service.md](https://github.com/andrewgoodchild/custmatch/blob/main/docs/report/12-the-service.md)).

| Process | Command | What it does |
| --- | --- | --- |
| API and dashboard | `custmatch-server serve` | Lookups, real-time matching, overrides, the dashboard at `/dashboard` |
| Scheduler | `custmatch-server schedule ...` | Loads files dropped in an inbox, runs the batch at set times, checks health, alerts |
| One-off jobs | `load`, `batch`, `key`, `check` | The same work by hand |

All processes share the database (`CUSTMATCH_DB_URL`) and a data directory for run files
(`CUSTMATCH_DATA_DIR`). Run one API process and one scheduler; the API scales to more copies for
lookups, but only one holds the write lease for real-time matching at a time.

## How it works

```
                 ┌─────────────────────────── custmatch-server ──────────────────────────┐
  files ──► inbox│ scheduler ── load ──────────────┐                                     │
                 │     │                           ▼                                     │
                 │     └─ batch run ──publish──► ┌──────────── the database ───────────┐ │
                 │       (the custmatch          │ source records and every version,   │ │
                 │        library)               │ loads, runs (which one is current), │ │
  your apps ────►│ API: lookups, POST /match ◄──►│ customers, links, golden records,   │ │
                 │   matcher: one writer,        │ history, steward decisions,         │ │
                 │   index in memory             │ audit log, API keys                 │ │
  stewards ─────►│ dashboard ◄──────────────────►└─────────────────────────────────────┘ │
                 └───────────────────────────────────────────────────────────────────────┘
```

- **One database holds everything shared:** the source records and their loads, the runs and
  which one is current, the results, stewards' decisions and the matcher's lock. Postgres for
  production (many readers and writers, advisory locks, managed everywhere); SQLite on a
  persistent disk for a trial or one small team. The code targets both, so moving from trial to
  production is configuration.
- **Loads keep history.** A load is full (records missing from it are marked deleted, not
  removed) or incremental. The current version of every record is kept, and every earlier
  version, so "what did this record look like at load N" has an answer. Only the mapped fields
  are stored.
- **The batch run is the truth.** It is `custmatch run` over every live record - loaded from
  files or arrived through the API - with labels and stewards' decisions read from the database,
  and customer IDs carried over from the current run. It publishes in **one transaction**:
  readers see the old run or the new one, never a mix. The current run and one before it are kept.
- **Real-time matching is provisional.** `POST /match` scores a record against the current run's
  model in memory - a compact index of hashed blocking keys, built from the batch blocker's own
  key definition - and joins it to the customer it fits: it must match at least half of that
  customer's records, not just one. It merges a second customer only on a link of 0.9 or more.
  Exactly one process writes (a lease); lookups keep working without it. Stewards' decisions,
  trusted IDs, the twin guard and duplicate-free sources apply at once; the next batch run
  reconciles everything, and changes reach users as merges and splits in the history.
- **The tables** (`custmatch_server/schema.py`, created by versioned migrations):

| Table | Holds |
| --- | --- |
| `loads` | Every load: source, origin (a file or `api:<day>`), full or incremental, and what it inserted, updated, deleted and left unchanged |
| `source_records`, `source_record_versions` | The current version of every record, with a content hash and `deleted_at`; and every earlier version |
| `runs`, `current_run` | The catalog of runs, and the one readers see |
| `customers`, `record_customer`, `links`, `customer_events` | Per run: golden records, which customer each record is in, the accepted links with evidence, and merges and splits (partitioned by run on Postgres) |
| `realtime_records`, `realtime_merges`, `realtime_moves` | The API's matching decisions, replayed when the matcher restarts, until a batch run reconciles them |
| `overrides`, `steward_log`, `golden_pins`, `review_claims` | Stewards' decisions and their approval, the audit log (with each action's note), pinned golden values, and review pairs held by the steward working on them |
| `api_keys`, `leases` | Hashed API keys with roles, and the matcher's write lease |

## Start it

A trial on your machine, with Postgres and the demo data:

```sh
docker compose -f server/docker-compose.yml up -d --build db api
docker compose -f server/docker-compose.yml run --rm api key create me --role admin   # shown once: keep it
```

Load the demo (or your own files, below) and run a first batch - one scheduler pass, done by hand:

```sh
for s in web app store; do cp examples/demo/$s.csv server/inbox/$s.full.csv; done
docker compose -f server/docker-compose.yml run --rm scheduler \
  schedule /demo/config.json --labels /demo/labels.csv --inbox /inbox --once
```

Then start the scheduler, which from now on loads what arrives in the inbox and runs the batch
at 02:00 UTC (only one scheduler works at a time, so a `--once` pass while it runs does nothing
and says so):

```sh
docker compose -f server/docker-compose.yml up -d scheduler
```

Then open http://localhost:8765/dashboard and paste a key with the `steward` (or `admin`) role.

The image holds the library's default model only. For the opt-in `"matcher_model":
"forest_catboost"` (docs/report/03-which-matcher.md "Which classifier"), build it with CatBoost: `CUSTMATCH_EXTRAS=catboost
docker compose -f server/docker-compose.yml build`, or `docker build --build-arg CUSTMATCH_EXTRAS=catboost
...`. A run trained with it cannot be served by an image without it; the API says so.

For production, run the same image three ways (API, scheduler, and Postgres you already manage)
on whatever runs containers for you, with your config and labels mounted where the commands
point. Without Docker: `pip install ./ ./server` from the repository.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `CUSTMATCH_DB_URL` | `sqlite:///custmatch.db` | The database. Postgres: `postgresql+psycopg://user:password@host/db` |
| `CUSTMATCH_DATA_DIR` | `data` (`/data` in the image) | Run files: each run's inputs, outputs and model. Shared by every process |
| `CUSTMATCH_AUTH` | on | `off` answers anyone who can reach the API: a local trial only |
| `CUSTMATCH_ALERT_WEBHOOK` | none | Where the scheduler posts alerts: any URL taking a JSON POST (`{"text": ..., "checks": [...]}`; Slack incoming webhooks read `text`) |
| `CUSTMATCH_MAX_RUN_AGE_HOURS` | 36 | Warn when the current run is older |
| `CUSTMATCH_MAX_CUSTOMER_CHANGE` | 5 | Warn when the customer count moves by more than this % between runs |
| `CUSTMATCH_MAX_REVIEW` | 1000 | Warn when the review queue is longer |
| `CUSTMATCH_FOUR_EYES` | off | `on`: a steward's decisions, merges, splits and revocations wait until a different steward approves them (Decisions tab). Needs authentication on |
| `CUSTMATCH_DB_PASSWORD` | `custmatch` | The Compose file's Postgres password: set it for anything but a local trial |

The matching itself is configured by the custmatch config file (sources, columns, rules) and the
labels file: see the [guide](https://github.com/andrewgoodchild/custmatch/blob/main/docs/guide.md). The database creates and upgrades its own tables
when any process starts.

## API keys

```sh
custmatch-server key create NAME --role ROLE   # prints the key once; only its hash is stored
custmatch-server key list
custmatch-server key revoke NAME               # stops working within 30 seconds
```

| Role | Can |
| --- | --- |
| `read` | Look up records and customers, read runs, loads, decisions and health |
| `match` | Also match records in real time (`POST /match`) |
| `steward` | Also record decisions (`POST /overrides`): the dashboard's role |
| `admin` | Everything |

Give each person and each system its own key: a steward's decisions are recorded under their
key's name. `server/sql/roles.sql` sketches least-privilege database roles for Postgres.

## Getting records in

**The inbox** (with the scheduler running): drop a file named `<source>.<mode>.<csv|parquet>` -
`web.full.csv`, `store.incremental.2026-10-05.parquet` - where `source` is a source in your config
and `mode` is:

- `full`: the source's complete contents. Records missing from it are marked deleted (kept, with
  their history, but left out of matching).
- `incremental`: new and changed records only.

The scheduler waits until a file's size has stopped changing, loads it, and moves it to
`inbox/done/` (or `inbox/failed/`, with an alert). The image runs as user 10001: on Linux, make the
inbox folder writable by it (`chown 10001 inbox`). Each load is recorded: what it inserted,
updated, deleted and left unchanged, and every version of every record.

**By hand:** `custmatch-server load CONFIG SOURCE FILE [--full]`.

**Through the API:** records matched with `POST /match` and `"write": true` are stored as that
day's API load and included in the next batch run.

## Batch runs and the schedule

`custmatch-server schedule CONFIG --labels LABELS --inbox DIR --batch-at 02:00` loads the inbox as
files arrive and runs the batch at 02:00 UTC daily (repeat `--batch-at` for more). A batch run
matches every live record, keeps customer IDs stable from the previous run, publishes in one
transaction (readers see the old run or the new one, never a mix), keeps the previous run for
rollback, and runs the health checks. Only one scheduler works at a time, even if two are started.

With your own scheduler (cron, a Kubernetes CronJob, Airflow), run
`custmatch-server schedule ... --once` instead: one pass - load the inbox, run a batch, check -
exiting 0 (ok), 1 (warnings) or 2 (failed).

**Stewards' decisions train the model.** Each "same" or "different" a steward records overrides
that pair at once, and the next batch run also trains on it. Measured on the demo
(`server/learning_test.py`, `results/server_learning.json`): starting from 200 random labels,
three rounds of answering the review queue lifted clustered F1 from 79.5 to 88.8 when decisions
train the model, against 84.1 when they only override their own pairs.

## The dashboard

`/dashboard`, with a `steward` key:

- **Overview**: health, the current run, loads, recent decisions.
- **Model & data**: the threshold and accuracy estimates, score distribution, what the model
  relies on, completeness and common values by source, records too sparse to link, blocking,
  runs compared.
- **Customers**: search for a person; browse customers worth a second look (largest, weakest
  link, ...); sample links by score. A customer shows its golden record, its records as a network
  of links, and why each link scored what it did.
- **Review**: the uncertain pairs, most uncertain first: same person, different people, or skip
  (keys S, D, K). Filter by source, the model's call or score; a list mode decides many at once;
  a note goes with each decision. Pairs shown to one steward are held from the others for 15
  minutes.
- **Decisions**: every decision with its author and note; revoke one, approve or reject proposals
  (with `CUSTMATCH_FOUR_EYES=on`), download them as CSV; steward activity, including how often
  each steward overturned the model.

In a customer: golden-record lineage (which records hold each value) and pinning a value, which
beats survivorship in this and every later run; merging with another customer and splitting a
record out, both previewed before they happen and applied at once; the household; every steward
action on its records. Under Customers, **Compare** puts any two records side by side and says
whether blocking would ever compare them (why they did not match); **Find possible matches**
scores a person you type in, without saving anything. Every panel has an (i) explaining what it
shows, how it is calculated and what good looks like.

## Monitoring

`GET /health` (no key) says the process is up. `GET /status` and `custmatch-server check` run the
health checks; the scheduler runs them after every batch and posts anything not ok to
`CUSTMATCH_ALERT_WEBHOOK`.

| Check | Warns or fails when |
| --- | --- |
| run | there is no published run, or the latest run failed |
| run_age | the current run is older than `CUSTMATCH_MAX_RUN_AGE_HOURS` |
| customers | the customer count moved more than `CUSTMATCH_MAX_CUSTOMER_CHANGE` % between runs: look at the run's merges and splits before trusting it |
| review | the review queue is over `CUSTMATCH_MAX_REVIEW`, or growing fast |
| loads | a load failed in the last day, or has run for over two hours |
| realtime | a real-time record has waited longer than the run age limit for a batch run |

## Backup and restore

Back up two things together: the database and the data directory.

```sh
pg_dump -Fc "$DATABASE" > custmatch-$(date +%F).dump        # Postgres
tar czf custmatch-runs-$(date +%F).tgz -C "$CUSTMATCH_DATA_DIR" runs
```

With SQLite, stop the processes and copy the `.db` file (or use `sqlite3 custmatch.db ".backup copy.db"`).
Restore both from the same time: the database names the run directories it uses. Only the
current run and the one before it are kept, so older run directories can be deleted or archived.

## Upgrading

1. Back up (above).
2. Stop the scheduler, so no batch run is in progress.
3. Start the new image's processes: the database upgrades itself at start (migrations).
4. Run `custmatch-server check`.

Migrations do not run backwards automatically; to go back, restore the backup and the old image.

## Sizing

Measured with Postgres on a laptop (`server/scale_test.py`, `results/server_scale.json`), per
million records: loads about 1 minute, a batch run about 2.5 minutes (2-3 GB of memory), the
real-time matcher starts in 6 seconds and holds about 0.5 GB, a real-time match takes 24 ms
(median) and a lookup 7 ms; the database grows by about 1.8 GB. The batch library has run 10
million records on one 16-core machine (docs/report/06-blocking-and-scale.md, "Larger files").

## Troubleshooting

| Symptom | Look at |
| --- | --- |
| `401` from every call | The key: `custmatch-server key list` (revoked?); the header is `Authorization: Bearer <key>` |
| `409 another matcher holds the write lease` | Another API process is the writer; send writes to one process, or wait 60 s after stopping the old one |
| A file stays in the inbox | Its name: `<source>.<full|incremental>.<csv|parquet>`, with a source from the config |
| A file lands in `inbox/failed/` | The scheduler's log and the alert: usually a missing column or an unknown source |
| "customers" warns after a run | Customers' events in the dashboard (Customers, then a customer's history); a new source or a data problem can merge many customers at once |
| The review queue keeps growing | More labels help most: answering the queue trains the next run |
