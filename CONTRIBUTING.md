# Contributing

Thanks for looking. custmatch is small and measured: every option in it was benchmarked, and
`tests/test_reported_results.py` checks that the numbers in the docs still match `research/results/*.json`.

## Set up

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest -q                 # about two minutes; data-dependent tests skip without data/
custmatch run examples/demo/config.json --labels examples/demo/labels.csv --out demo_out/
```

The service (`server/`) has its own package and tests:

```bash
pip install -e ./server httpx
pytest -q tests/test_server_*.py          # SQLite; CUSTMATCH_*_DB_URL point some at Postgres
python server/smoke_test.py               # the end-to-end smoke test, printed
```

The research scripts need more (see `docs/reproducing.md`).

## Changes

- **Behaviour changes need a test that fails without them.** A review of the code found that
  the suite caught only 4 of 27 deliberately introduced bugs; the tests added since each fail
  on the code they guard against - please keep it that way.
- **Changes that move a measured number need the measurement rerun.** If your change affects
  results, rerun the script that produced them (the table in `docs/reproducing.md` says which),
  commit the updated `research/results/*.json`, and update the docs; the claims tests will tell you
  which sentences are stale.
- **Say what a result does not show.** Single label draws, datasets the design was tuned on,
  and synthetic data written by the same author as the fix are all worth a sentence.
- Keep third-party data out of the repository unless its licence allows redistribution, and
  credit it in `NOTICE`.

## Reporting problems

Open an issue with the config (column names are enough, not the data), the command, and
`summary.json`. For security or privacy problems see docs/guide.md, "Security".
