# Changelog

## Unreleased

**Fixed:** dates of birth are read in more forms: with a space between the parts ("1953 11 09",
"11 09 1953"), with the month in words ("09 nov 1953", "November 9, 1953", "jul 18sat 1953"), and
with no separators ("19531109"). They stayed as written before and disagreed with the same date
written "1953-11-09". Partial dates ("may 2008") are still kept as written. Found on BPID, AWS's
public benchmark of personal identities: pairwise F1 on its 10,000 labelled pairs rose from 67.48
to 74.45 with 8,000 labels, and its blocking recall from 70.18% to 71.89%. No date in any other
dataset in this repository reads differently (docs/report/09-against-other-tools.md "BPID").

**Fixed:** me.com and mac.com addresses read as icloud.com, as Gmail's dots already were: they
are one Apple account, so jo@me.com and jo@icloud.com now agree. Reported from deduplicating an
address book, where one person's me.com and mac.com addresses did not agree. BPID's 13 me.com
addresses change nothing there (74.45 with 8,000 labels, rerun).

**Added:** `"group_rows": true` on a source reads long-format data - one row per contact point,
rows sharing a `record_id` one record. The record's other surnames, emails and phones become
aliases, so a person with several addresses is one record, not a record per address whose
pairs take up the labels. `add` reads such sources the same way.

**Fixed:** with a country set, a title in the first-name field ("Dr John", "Prof. Jane") is
dropped. It stopped "Dr John" agreeing with "John", and became the name's nickname root, so
every "Dr" in a file shared a blocking key. `"normalise": {"titles": false}` keeps titles.
No benchmark sets a country, so none moves.

**Fixed:** no-reply, notification and bounce addresses (`drive-shares-noreply@google.com`,
`calendar-notification@...`, `notifications@...`, `mailer-daemon@...`) are junk on sight, not
only a bare `noreply@`; the user name's word must be joined by `-` or `_`, as services write
them, so BPID's `gertie.noreply@` style personal addresses are not caught. A run warns when such
an address carries a person's name: in an address book, a file-sharing relay filed under one
person's name was linked to ten of that person's addresses, every false link of the best run.
No benchmark file has one besides `noreply@example.com`, already junk.

**Added:** a warning when a two-field blocking key blocked on one field alone for a quarter of
the candidate pairs or more (first name + street is first name alone where there is no street),
and a per-key count of those pairs in `blocking_report.json` (`pairs_with_a_blank_part`). On an
address book with no streets, first name alone gave 660 of 1,057 pairs, mostly strangers, and
nothing said so. Of the benchmarks it fires on historical_50k and fake_1000, which have no
streets; the keys themselves are unchanged.

**Added:** `"blocking": "adaptive"`. A block too big for `max_block` is split on up to three
refinements on different fields, the file's extra fields included, and every piece is kept, so a
typo in one refining field no longer loses the pair. On the 5M-record NCVR benchmark it lifts the
most F1 blocking allows from 97.13 (`"fixed"`) to 99.05, with 192M candidate pairs against 126M,
and the full run's clustered F1 from 87.19 to 88.45 (recall 86.7 to 90.2).

**Changed:** with no `"blocking"` set, blocking switches to `"adaptive"` by itself when the
default keys would leave more than 1% of the records only in blocks too big to compare, and
`summary.json` says so; the large-block check still runs, and `add` and runs reusing the model
block the same way. `"blocking": "core"` keeps the old behaviour. None of the five small
benchmarks or BPID switches (0.45% at most); NCVR does from 200,000 people, and so do dedupe's
two organisation files, which gain (childcare sites 86.95 to 89.08, patents 75.88 to 77.37).
On all 5,000,000 NCVR records with nothing set, both changes together score 86.24 clustered F1
against 84.33 before, with the same 300 random + 700 active labels.

**Changed:** `custmatch sample` draws half the random labels spread over the pairs' agreement
patterns and writes each one's `weight`; `run` weighs the threshold's estimates by it, so they
stay unbiased. Plain draws held a handful of matches: on 1,000,000 NCVR people with 300 random +
700 active labels clustered F1 rose from 80.84 (2 random matches) to 92.88, and on synthetic
address books with 300 random labels from 71.58 to 77.21 over 30 draws. It is the default at
every size; `--no-stratified` draws plainly; labels without weights count equally, as before.
Results recorded before, and comparisons giving every tool the same labels, still draw plainly
(docs/report/06-blocking-and-scale.md "Random labels that still hold matches"). A stratified draw now
returns as many pairs as asked: a pair both halves picked was kept once, so on a small file 300
asked could give 290 or so (30 asked gave 29). Each overlap is redrawn from its stratum; where
none overlaps, as on NCVR, the same pairs are drawn.

**Added:** `research/scripts/address_book_data.py`, synthetic personal address books built from
people (one record per email address; employer changes, Apple and Gmail variants, nicknames,
titles, maiden names) with the traps a real one has - relay addresses under a person's name, a
switchboard phone, shared household phones, same-name strangers, brand names - each tagged in
the truth; and `research/scripts/address_book_benchmark.py`, which scores four labelling budgets
on dirty and clean books with false links broken down by trap.

**Faster:** oversized blocks are split with one sort, not a scan apiece, and pairs are
deduplicated with an integer sort: `"fixed"` blocking at 200,000 people takes 6.3 s, from 9.7 s.
The candidate pairs are the same.

## 0.1.2 (2026-10-16)

Six ideas from three other matching projects - zer (ZAL-Analytics/zer), jlink (keltokhy/jlink)
and sriracha (ccuetoh/sriracha) - each checked against custmatch first.

**Fixed:** text is read in one Unicode form (NFKC, invisible format characters removed,
typographic apostrophes and hyphens made ASCII). A name typed on a Mac arrives decomposed -
"Müller" as u plus a combining diaeresis - and differed from the same name typed elsewhere in
every exact key and agreement feature; so did a name with a zero-width space pasted from a web
page, full-width letters, and O’Brien against O'Brien. Accents are kept. Of the benchmark
datasets it changes 30 values in historical_50k (mostly d’Urban-style apostrophes) and 11 in the
synthetic file: with 1,000 random labels clustered F1 is unchanged on both (85.53, 97.28); with
300 random and 700 active labels historical_50k went from 85.72 to 86.64 on one draw - active
learning follows another path once a few values change - and the synthetic file is unchanged
(98.61) (research/scripts/unicode_form.py). Found by fuzzing: cleaning twice now
always gives what cleaning once gave.

**Added:**
- `"aliases"`: other values of a field - a maiden or previous surname, a second email, a work
  phone - blocked like the field's own and compared through the best-agreeing values
  (docs/guide.md "Previous names and second contacts").
- Placeholder dates of birth (with `"placeholders": true`): 1900-01-01, 1899-12-31, any 9999 date,
  and a date on many records that share no names (a form default). Where 1 January means "year
  only" (historical_50k) it is kept, and so are dates in the future, which are typos of real ones.
  placeholders.csv and summary.json report them.
- `manifest.json` in every output: the SHA-256 of the config, each source file and the labels, the
  seed, the library versions and every output file. `custmatch verify DIR [--against DIR2]`
  checks a folder against it, or two runs against each other.
- `custmatch audit sample` / `custmatch audit estimate`: label a sample drawn from every score
  band and get the run's precision, recall and F1 on your own records with 95% intervals. Each
  run saves the pool it draws from (state/audit_pool.parquet).
- CI: Dependabot keeps the pinned actions, the image's base and the Python dependencies current;
  pip-audit checks the dependencies for known vulnerabilities, weekly and on every change.
- The service image (ghcr.io/andrewgoodchild/custmatch-server) is built on Python 3.14.

## 0.1.1 (2026-10-12)

**Fixed:** the same input, labels and seed now give the same result every time. The forest's
probabilities on large batches came from scikit-learn's predict_proba, which adds the trees'
votes from several threads in the order they finish, so the last bit could differ between runs.
That was enough to reorder tied pairs in active learning, so `custmatch sample --active` and the
dashboard could ask for different pairs, and a run could score differently. The trees are now
added in a fixed order, at the same speed.

## 0.1.0 (2026-10-10)

First public release: the library, and **custmatch-server**, the self-hosted service
(server/README.md; its measurements in docs/report/12-the-service.md, "The service"). The service keeps source records and their loads in
one database (Postgres or SQLite), runs the batch on a schedule from an inbox folder with health
checks and webhook alerts, matches and looks up records in real time over HTTP (24 ms median at
a million records), and gives data stewards a dashboard: the review queue with score
breakdowns, customers as a network of links, merge and split with previews, golden-record
lineage and pinned values, decisions that train the next run (80.0 -> 88.1 clustered F1 over
three review rounds on the demo), four-eyes approval, an audit log, and the model's accuracy and
the data's quality - every panel explained. API keys with roles. Reviewed before release for correctness, simplicity and security: real-time work arriving during
a batch run is reconciled by the next run, not lost; real-time customer IDs survive batch runs
and are never given to anyone else; a matcher that another process wrote past rebuilds itself;
four-eyes covers merges, splits and revocations, and a proposal no longer replaces the applied
decision it would change; record ids may hold any printable character; list endpoints are capped,
the dashboard has a content security policy, the image runs as a non-root user. Library changes
made for it:
the standard features are vectorised (2.8x faster, identical values), blocking reports pairs per
key, runs record their threshold sweep, `"trusted_ids"` columns are kept by the loader (they
were silently ignored unless also extra fields), and evidence no longer calls two blank streets
"same street".

Removed, because measurement showed they did not earn their place (docs/report/11-adopted-and-dropped.md "Removed in
0.1.0"): the Ditto transformer stage (`--ditto`, `"ditto_model"`, the `ditto` extra), correlation
clustering (`"clustering"`, `"cc_max_size"`, `"cc_missing"`), the ambiguity rule
(`"ambiguous_margin"`, `"ambiguous_different_below"`), `"missing_as_unknown"`, and the Wikidata
name list (`scripts/build_name_variants.py`). Configs naming them get a warning. Also removed:
unused helpers and constants. `torch` is now declared under the `models` extra (TabPFN).

The Splink feature (`"splink_feature"`, the `splink` extra) is replaced by `"match_weight_feature"`:
custmatch's own Fellegi-Sunter weight (term-frequency adjusted, fitted by EM without labels;
changeable fields such as address and phone count only when they agree), no Splink needed.
Opt-in: over three label draws +3.83 on the sparse Australian file and +1.06 on historical_50k
with random labels, at most 0.02 lower anywhere, but it fails the web-sign-up scenario (a
first name and an email only). Splink is still used by the benchmark scripts, as the tool
compared against.

Removed after a review on six datasets (`research/scripts/modal_models.py`): the `"matcher_model"` options
hgb, CatBoost, TabPFN and `"auto"`, the `models` extra and with it torch. None beat the forest by
enough to justify its install; TabPFN, the best, gained 0.5 on average for a GPU and 7 to 12
times the run time. Configs naming them are refused or warned.

Added after that review: `"matcher_model": "forest_catboost"`, opt-in with the `catboost` extra
(`pip install "custmatch[catboost]"`), the forest's and CatBoost's probabilities averaged. Over
the same six datasets and three label draws it gained +0.41 clustered F1 on average and was
never more than 0.03 behind on a dataset's mean (historical_50k +1.3; `research/results/ensemble.json`),
for runs about 30% longer on files of up to 50,000 records. The forest stays the default. The
server image takes it with `--build-arg CUSTMATCH_EXTRAS=catboost`; a saved model that needs
CatBoost where it is not installed now says so instead of failing on the import.

Also added: oversized blocks that are one real entity are compared after all. A block skipped as
a hub is scored by the trained model on 50 sampled pairs; one whose pairs mostly match (an
organisation with hundreds of records) joins the candidates, within a pair budget; `add` and
`explain` repeat the check. It kept nothing on the six person benchmarks and lifted dedupe's
patent applicants from 6.93 to 75.88 clustered F1 (`research/scripts/large_block_check.py`);
`"check_large_blocks": false` turns it off. And contact profiling now reads the other contact:
an email on 8+ records with mostly different mobiles, or a mobile with mostly different emails,
and no shared surname or address, is junk - the made-up mobile typed on nameless sign-ups (a new
scenario, "Nameless sign-ups ... all typing one made-up mobile", failed every draw before it).
Numbers that only look made up - runs, repeated pairs and triples, 867-5309 - are no longer
junk on sight, since people buy them as "gold" numbers: they are junk when 3 or more records
that are different people share one. A bought number on a sparse sign-up (a new scenario) went
from never linked to always; junk on sight is kept for values no one can own (0000000000, the 555
area code, fictional ranges). No benchmark score changed.

Also added: `suspect_records.csv`, records that look entered in bad faith or by a bot, for
review - profanity in a name or email (the LDNOOBW list, downloaded on first use, without the 36
words that are real names: Hooker, Dick, Felch...), a disposable email domain, or a random-looking
email user name that shares nothing with the record's names. It flagged none of 5 million North
Carolina voters, nor any record in the held-out public and generated files, and changes no
match. 2.4 s at a million records; `"suspect_records": false` turns it off.

The real-time matcher no longer chains people through one sparse record: a record joins a
customer only if it matches at least half of the customer's records (up to 25 scored), not just
its best link. Over five replay seeds real-time precision rose from 86.60 to 89.98 and F1 from
80.75 to 81.67, better on four of five seeds (`server/replay_test.py --all`, which now writes
`research/results/server_replay.json`).

Also removed: LLM labelling (`custmatch label`, and the `"llm_fields"` key). It was tried on
synthetic data to see what the hype was about, and dropped: it added little over active learning
with hand labels, and sending customer records to a third party is a security risk. custmatch now
sends no customer data anywhere, and no longer depends on httpx. The research is in the report
(chapter 4). SECURITY.md is now the guide's "Security" section.

Also removed, as dead code: the loop that chose among several models (one is fitted now), the
`"matcher"` field of `summary.json` (always "forest"), refusals of saved runs that used Ditto or
the Splink feature (no released version made them), and unused variables and imports in the
research scripts.

Dependencies: duckdb, jellyfish and kuzu are gone. The opt-in `"blocking": "fixed"/"learned"`
modes pair records in numpy instead of a DuckDB self-join (identical pairs, about 2 s slower at a
million records) and use custmatch's own Soundex (identical to jellyfish's on every Latin-script
name in the test data; names in other scripts get no code instead of a meaningless one). Kuzu
was only in the test extra, for the research graph script. Results with a country profile were
rerun for the sparse Australian file after the Wikidata name list was removed (80.7 -> 81.7).

The historical_50k benchmark config adds similarity blocking on the names within a birth place:
exact keys left a perfect matcher at 93.3 clustered F1, similarity blocking lifts that to 96.8,
and the forest gains about a point over three draws (docs/report/06-blocking-and-scale.md "Dense files").

The threshold rule: thresholds the random labels cannot tell apart (within half a held-out
match's worth of the best estimate) count as tied and the middle one is chosen, instead of the
strictest. Chosen on 15 captured cases and checked on 24 held-out public-benchmark cases
(`research/scripts/threshold_capture.py`, `research/scripts/threshold_rules.py`): mean F1 lost against the best
threshold 0.84 -> 0.14 and 0.60 -> 0.27, worst case 10.2 -> 1.1 and 4.7 -> 1.8.
`"threshold_rule": "strictest"` keeps the old rule. Every published result from a script that
runs custmatch was rerun under it, except the classifier bake-off, the Ditto run on
historical_50k and the Splink 5 column, which say so. On NCVR at 200,000 people custmatch now
leads Splink on every draw (95.7 against 94.1). Also fixed: the cookie shared-device limit
counted one value typed two ways as two identities.

A second review (code, experiments, release) and its fixes. Claims corrected: the stricter-
threshold result had used a threshold picked against the truth - `min_precision` as a user runs
it is safer than a precision-tuned CDP but finds fewer matches; the CDP comparison is now on
three generated populations; on NCVR at 200,000 people (three draws) Splink given the same
labels beat custmatch on average; record dates helped in all six runs, but not clearly through
moves; the scenario file is a development set. Bugs fixed: an add config's overrides are kept
for later adds; a single "most_recent" survivorship rule sees dates and addresses are never
assembled from different records; blocking reports hold no key values under minimal_outputs or
hashing; the name skeleton is hashed with names; hashed names turn the twin guard off and limit
the cookie name check to exact names; lsh_blocking with another blocking mode is an error;
contact safety never falls back to junk; event phones meet records of any configured country; a
must-link survives every later step; placeholders are found before hashing; Ditto honours
min_precision. Unknown config keys and a missing name list are reported; the first-run download
uses certifi; the security notes (now docs/guide.md "Security") cover every pickle load and the
download; CI adds Python 3.13.

For contacting people: `"min_precision"` (the most complete threshold meeting a precision
target), `"contact_confidence"` (contact details only through certain links; `contacts.csv` per
record, in `run` and `add`), and warnings when labels are random-only or hold too few matches.

`"unique_sources"` (one record per person in a duplicate-free source; run and add) and
`"lsh_blocking"` (MinHash similarity blocking), both from anonlink, measured on NCVR. Splink
compared on NCVR (`--splink`). Deterministic CDP matching compared with custmatch at high and
low sign-up rates (`research/scripts/cdp_comparison.py`). Hashed first names no longer leave their
nickname roots in plain text. The guide's privacy section now says what hashing does and does
not protect, and points to anonlink for linkage between organisations.

The Leipzig NCVR benchmark (`research/scripts/ncvr_leipzig.py`, report "A public benchmark"):
`extra_blocking` keys take blocking transforms (`"first_name:first1"`, `"last_name:soundex"`),
which lift recall where postcode blocks are too big. Third-party data is no longer committed:
the US nickname list downloads on first use (pinned, checksum-checked) and the Wikidata list is
built locally, both into `~/.cache/custmatch`.

Cookie and device-ID stitching (`"identity_events"`, `custmatch/identity.py`, `cookies.csv`)
with a value limit, excluded channels and staff cookies, and a first-name check against
call-centre agents (+2.6 F1 on generated events, `research/scripts/cookie_research.py`).

A scenario file (`examples/scenarios/`, `research/scripts/scenarios.py`): 24 hard cases with known
answers, scored in docs/report/05-customer-data.md "Scenarios". `"twin_guard"` keeps twins apart. Accented first
names now find their nickname roots. `add`'s config may set decision and output options
(`max_cluster`, `ambiguous_margin`, survivorship, ...) and is refused if it changes a setting
the trained model depends on. Test coverage 88% to 95%.

`"record_dates": true`: a years-apart feature from `updated_at` (+1.2 F1 on a dated build of the
Australian file, `research/scripts/record_dates_research.py`); `years_apart` as an extra-field comparison.

Monitoring: `monitor.json` per run and `custmatch monitor` (match-rate, fill-rate, score and
threshold drift; `--fail-on-drift`). Survivorship rules are checked when the config loads, and
`{"prefer": [...], "then": rule}` chooses the fallback. Fixes from a code review: `add` on an
older state no longer breaks customers apart on the next `add`; a retired customer ID is never
reissued; `summary.json` counts customers correctly with carried IDs; must-links survive the
ambiguity rule, correlation clustering and the size re-cut; `add` applies must-links between
existing records; an updated record's old links are dropped and it keeps its customer's ID;
`explain` no longer says "matched" for a link that was cut. The "stable matches" research is now
`research/scripts/order_dependence.py`.

Running it over time: `review.csv` (pairs near the threshold); `"overrides"` (must-link /
cannot-link, honoured by `run` and `add`; a cannot-link unmerges); `"trusted_ids"`;
`run --previous` keeps customer IDs across reruns and `history.csv` logs merges, splits, updates
and deletes; `run --model` rescores with a saved model; `add` handles updated keys and
`--delete`; `custmatch explain` says why two records were or were not matched;
`"survivorship"` per field (most recent via `updated_at`, preferred sources, addresses whole).
Privacy: `"hash_fields"` (keyed hashes), `"minimal_outputs"`, `custmatch forget`.

Country profiles: `"country"` takes one code or several (`["AU", "NZ"]`, with an optional
per-record `country` column) and switches on that country's date order, postcode format, phone
plan, national identifier (US SSNs that are never issued or are placeholders are blanked) and
nickname lists (a new Australian / British list: Bec, Shazza, Davo, Con). Profiles for US, AU,
NZ, GB and CA in `custmatch/countries.py`; other ISO codes get sensible defaults.

Fixes from a review of the code, research claims, release readiness and tests:

- A clean `pip install .` now works: networkx, duckdb and jellyfish are declared, and a test
  checks every import against `pyproject.toml`.
- Labels written back by spreadsheets ("1.0", "True") are read; unknown values are refused
  instead of silently becoming non-matches.
- `custmatch add` reads new records with the saved run's settings (country normalisation,
  name features), blanks placeholders, and refuses links that would grow a customer past
  `max_cluster` (`refused_links.csv`).
- The email-domain typo fix no longer rewrites real domains (ge.com, cloud.com) and is the same
  for every file.
- Blocking: splitting an oversized block keeps records without the refining key (short first
  names); `"fixed"` and `"learned"` blocking keep the country keys; junk emails are never
  email-user keys; `custmatch blocking` describes the blocker `run` uses.
- Households: streets that differ in direction or type (N / S Main St, Park Rd / Dr) are no
  longer merged; contact profiling is one grouped pass (56 s -> fast at 20,000 emails).
- Typed Parquet columns: a numeric phone column no longer gains a trailing digit; dates with
  gaps no longer crash. Australian unit addresses (3/12, Unit 4, U4) are split.
- Clear errors instead of crashes: no candidate pairs, one label class in an active batch or a
  fold, missing files.
- Docs: claims reworded where the evidence was weaker than stated (see `docs/report/`), a
  bundled demo, CI, CONTRIBUTING and SECURITY.

## 0.1.0

First version: blocking, forest matcher with the threshold chosen after clustering,
households, golden records, incremental matching, LLM labelling behind a gate, optional
Splink feature and Ditto stage, country normalisation.
