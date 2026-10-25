# custmatch — matching your customer records

`custmatch` finds which records are the same customer across your files, groups customers into
households, flags junk and shared contact values, and writes one golden record per customer.
This guide is how to use it: the ideas behind each step, then your files, labels, a run, and
running it over time. How well it works, and the experiments behind every choice, are in
[report](report/README.md).

## How it works

### The problem

A business holds records of the same people in several places: a CRM, a billing system, web and
app sign-ups, a store's point of sale. The same customer appears many times, written
differently each time - "Bill Smith" and "William Smith", a phone with and without its country
code, a typo in the surname, a new address after a move. Some records hold only a first name and
an email. Other records look alike but belong to different people: a husband and wife sharing a
phone, a father and son with one name, two strangers called John Smith.

**Matching** (also called entity resolution, record linkage or deduplication) decides which
records are the same person. custmatch then groups people into **households** and writes one
**golden record** per customer: the best name, address, email and phone across their records.

### The pipeline at a glance

```
records ─► clean & normalise ─► profile contacts ─► blocking ─► compare pairs ─► random forest
                                                                                      │
golden records ◄─ households ◄─ customers ◄─ threshold chosen on whole customers ◄────┘
```

Each step is explained below, in that order; the options that control it are in the numbered sections after.

### Records, sources and the config

A **source** is one file or table: `crm.csv`, `billing.parquet`. A **record** is one row of a
source, identified by the source's name and its `record_id` (`crm:12345`). The config file
maps each source's columns onto custmatch's own fields - `first_name`, `last_name`,
`date_of_birth`, `email`, `phone`, `address` and so on - so files with different column names
can be matched together. Fields custmatch does not know can be compared too, as
**extra fields** (a loyalty number, a postcode, a town), each with a comparison type: `exact`,
`similarity` (fuzzy text) or `edit` (typos in codes and dates).

### Cleaning and country profiles

Before anything is compared, values are put in one form: lower case, spaces trimmed, an address
split into house number, street and unit (`unit 4, 12 main st` and `4/12 main st` are the same).
Text is put in one Unicode form too: a name typed on a Mac arrives with its accents as separate
characters and differed from the same name typed elsewhere; invisible characters pasted from web
pages, full-width letters and curly apostrophes (O’Brien) differed the same way. Accents are
kept - José and Jose are compared by similarity. Of the benchmark datasets, only historical_50k
(30 values, mostly d’Urban-style apostrophes) and the synthetic file (11) change, and
clustered F1 with 1,000 random labels did not move on either (85.53, 97.28;
`research/scripts/unicode_form.py`).
A **country profile** (`"country": "AU"`, also US, NZ, GB, CA) adds the local rules:

- **Phones** parsed with the country's numbering plan, so `0412 345 678` and `+61 412 345 678`
  become one number. Without it they are two different numbers.
- **Dates** read day-first or month-first as the country writes them.
- **Emails** cleaned: Gmail's dots and `+tags` ignored, me.com and mac.com read as icloud.com
  (one Apple account), common domain typos fixed.
- **Postcodes** read off the end of an address.
- **Titles** dropped from first names: "Dr John" and "Prof. Jane" are john and jane. A record
  named only "Dr" keeps it.
- **Nicknames**: Bill is William, Peggy is Margaret. Each first name gets its **roots** - the
  full names it can be short for - and two first names are compatible when their roots meet.
- **Transliteration**: names written in Chinese characters or Arabic script are written in
  Latin letters, so 王伟 can meet "Wang Wei".

### Contact profiling: personal, household and junk values

An email or phone that two records share is strong evidence they are one person - usually. Two
kinds of shared value break that rule, and custmatch looks for both before matching:

- **Junk values**: placeholders typed to get past a mandatory field - `test@test.com`,
  `noemail@noemail.com`, `0400 000 000`, `123 Main St`. Some are recognised by pattern; others by
  behaviour: a value carried by many records whose names and addresses agree on nothing is
  junk. Junk values are ignored, as if blank. Service addresses are junk on sight too: no-reply,
  notification and bounce mailboxes (`drive-shares-noreply@google.com`,
  `calendar-notification@...`, `mailer-daemon@...`). Mail clients file them under the name of
  the person the service wrote for, so an address book holds them with a real name, and the
  run warns about those: the name can still link them to that person.
- **Household values**: a family email or a home phone, shared by people with one surname or one
  address. Agreeing on it means "same household", not "same person", so it counts for less.

Everything else is **personal**. Each value gets a class and a weight, and the weight - how much
an agreement on it is worth - falls as the value gets more common. Placeholder names and
addresses ("John Doe", "1 Test St") can be blanked too (`"placeholders": true`). See the guide,
"Households, fake contacts and placeholders".

### Blocking: which pairs to compare

Comparing every record with every other is impossible at scale: a million records make half a
trillion pairs. **Blocking** compares only records that share a **blocking key** - a value, or a
combination of values, that two records of one person usually have in common. The pairs it
proposes are the **candidate pairs**. custmatch's default keys include:

- surname + the first three letters of the first name (`smith|wil`)
- surname + street; house number + street; first name + street
- email; phone (when not junk)
- a nickname root + birth date, from the country profile

and you can add keys of your own (`"extra_blocking"`), using **transforms**: `first1` (first
letter), `first3`, `year`, `soundex`.

**Soundex** is a 1918 phonetic code that gives names which sound alike the same key. It keeps
the first letter and turns the next consonants into digits by sound group - b, f, p, v are 1;
c, g, j, k, q, s, x, z are 2; d, t are 3; l is 4; m, n are 5; r is 6 - drops vowels, merges
repeats, and pads to four characters. Robert and Rupert are both R163; Smith and Smyth are both
S530. As a blocking key it lets a misspelt surname still meet the right one.

**Hubs.** A key shared by too many records - a common surname in a big city, a postcode - would
produce a huge block whose pairs are mostly strangers. Blocks over `"max_block"` records (60 by
default) are skipped; a summary warning names common values that were, so you can raise the
limit if they are legitimate.

**Half-blank keys.** A two-field key with one field blank blocks on the other alone: first name
+ street is first name alone where there is no street. That keeps recall where addresses are
patchy, and historical_50k, which has no streets, relies on it; but on a file with no
addresses, such as an address book, those pairs are mostly strangers who share a
first name. `blocking_report.json` counts them per key (`pairs_with_a_blank_part`), and
`summary.json` warns when one key's are a quarter of the candidate pairs or more.

**Big files: oversized blocks are split, not skipped.** Blocks grow with the file: keys that make
blocks of a few dozen at 200,000 people make blocks of thousands at millions, and skipping them
leaves records compared with no one. When the default keys would leave more than 1% of the
records only in skipped blocks, custmatch instead splits each oversized block on up to three
other fields at once, keeping the pieces of all of them (`"blocking": "adaptive"`), and
`summary.json` says it did. On the NCVR voter file that lifted clustered F1 from 90.29 to 92.88 at
a million people (300 random + 700 active labels, the random ones stratified as below), and on
all five million records from 87.19 to 88.45 against splitting on one field (`"fixed"`). The five small benchmarks
and BPID stay under 1% (0.45% at most), so they block as before; dedupe's two organisation
files switch, and score 89.08 and 77.37 against 86.95 and 75.88. `add` and runs reusing the model
block the way the run did. Set `"blocking": "core"` to always skip instead; `"adaptive"` to always
split.

**Similarity blocking** (`"lsh_blocking"`) finds pairs no exact key would: it hashes the names'
letter pairs (MinHash, locality-sensitive hashing) so that names with typos in both records still
land in a shared bucket.

**The blocking ceiling.** A pair blocking never proposes can never be matched, whatever the
model does. The best result possible with a set of keys - every true candidate pair accepted -
is the ceiling, and on hard files it, not the model, is the limit. See "Blocking, and
what Splink, dedupe and Zingg taught" in [report/06-blocking-and-scale.md](report/06-blocking-and-scale.md) and "Dense files: why historical_50k scores lower".

### Comparing a pair: features

For each candidate pair custmatch computes **features** - numbers describing how the two records
compare:

- **Name similarity** by **Jaro-Winkler**, a 0-to-1 score for how alike two strings are that
  forgives transposed letters and rewards a shared beginning (Jonathan / Jonathon: 0.95).
- **Exact agreements**: house number, unit, middle initial, date of birth, SSN.
- **Contact agreements**, with the weight from contact profiling: sharing a personal email counts
  for much more than sharing a household phone.
- **Rich features** (on by default): how rare an agreeing name is (two Zebedees are stronger
  evidence than two Smiths), first and last name swapped, matching initials, and dates graded -
  days apart, same year, day and month swapped, one typo - because a mistyped birth date is
  common and should not separate one person.
- **Edit distance** (Levenshtein): how many single-character changes turn one value into another,
  for codes and dates.
- **Missing values.** For dates, IDs and the rich features a blank is its own state (−1), not
  a disagreement. For names it is not: a blank surname against a filled one counts as
  disagreement, which is why a sign-up with only a first name and an email is the hardest case.

### Labels

custmatch learns what a match looks like from **labelled pairs**: candidate pairs a person has
marked 1 (same person) or 0 (different people). `custmatch sample` writes pairs side by side to a
CSV for labelling. About 1,000 labels is the working target; at least 200 with both matches and
non-matches are required.

- **Random labels** are drawn at random from the candidate pairs. They represent the file
  honestly, so custmatch uses only these to judge its threshold (step 8). But when matches are
  rare, a random batch holds few of them.
- **Active learning** (`sample --active`) picks the pairs the current model is least sure of -
  those near the threshold - 100 at a time. These teach the model far more per label, especially
  on sparse records. Actively chosen pairs train the model but never judge the threshold,
  because they are deliberately unrepresentative. The recommended mix: 300 random, then active
  batches up to about 1,000.
- **No language model.** custmatch does not use a language model to label: we tried one on synthetic data to see what the
  hype was about, and dropped it - it added little over active learning with hand labels, and
  sending customer records to a third party is a security risk (report/04-labels.md).

Labels are the most important input: with few or wrong labels custmatch's precision becomes
unreliable, and `summary.json` warns when the labels look weak.

### The random forest

The **model** turns a pair's features into a probability that the two records are one person.
custmatch uses a **random forest**: 300 decision trees, each trained on a random sample of the
labelled pairs and a random subset of features, and the forest's probability is the average of
the trees' votes. One tree learns rules like "same email and first names compatible → match";
averaging many varied trees gives smooth, robust probabilities and resists overfitting a few
hundred labels. Matches are rare, so the classes are **balanced**: a matching pair counts for
more in training.

Gradient boosting (hgb, CatBoost, LightGBM, XGBoost) and tabular foundation models (TabPFN,
TabICL) were measured against it and did not earn their place; see "Which
classifier" in [report/03-which-matcher.md](report/03-which-matcher.md).

**The match weight** (`"match_weight_feature": true`, optional) adds a classic
**Fellegi-Sunter** score as one more feature. For each field it learns, without labels, two
probabilities: **m**, how often two records of one person agree on it, and **u**, how often two
random records do. Agreement adds log₂(m/u) to the score; an agreement on a rare value adds more
(**term-frequency adjustment**). m is fitted by **expectation maximisation** (EM) over all the
candidate pairs, so the weight carries what half a million unlabelled pairs say. Fields people
change - address, phone, email, surname - count only when they agree. See "A match
weight of custmatch's own" in [report/06-blocking-and-scale.md](report/06-blocking-and-scale.md), under "Blocking, and what Splink, dedupe and Zingg taught".

### From pairs to customers: the threshold and clustering

The model scores pairs; customers are groups. A pair scoring above the **threshold** is a
**link**, and linked records form customers by **connected components**: if A links to B and B
to C, A, B and C are one customer, even though A and C were never compared.

**Chaining** is the danger in that: a few borderline links can join hundreds of records -
whole families, or strangers - into one giant customer. A threshold that looks good pair by pair
can be badly wrong once links are joined up. This is well known in entity-resolution research
(Hassanzadeh et al., VLDB 2009, compared clustering methods for duplicate detection on exactly
this), and tools guard against it in different ways: dedupe clusters hierarchically and re-cuts
large clusters. custmatch's answer, its central design choice, is how the threshold is chosen:

- **Judged on whole customers.** Each candidate threshold is scored by **closure-aware F1**: a
  held-out labelled pair counts as matched only if its two records end up in the same customer
  after clustering, not merely if the pair scored above the line.
- **Cross-fitted**: the labels are split into 4 folds; each fold is scored by a model that never
  saw it, so every random label helps choose the threshold without being judged by a model
  trained on it.
- **An in-cluster precision check**: pairs sampled inside the customers each threshold produces -
  most never compared directly - are scored by the model, to catch chaining the labels cannot see.
- **Ties to the middle**: with few random matches, many thresholds score the same; custmatch
  takes the middle of those it cannot tell apart, not the strictest. See "How it
  decides" below.
- **`"min_precision"`** considers only thresholds whose estimated precision meets a target (0.97,
  say) and picks the best of those, for uses where a wrong merge costs more than a missed one
  (step 12). If none meets it, the most precise is used and `summary.json` warns.
- **`"max_cluster"`**: a customer larger than this is re-cut at stricter thresholds.

### Rules that override the model

- **Steward overrides** (`"overrides"`): pairs a person has ruled `same` or `different` win over
  the model, and are kept across runs. `different` splits a customer at its weakest link.
- **Trusted IDs** (`"trusted_ids"`): records sharing a loyalty or customer number are one person.
- **One record per source** (`"unique_sources"`): for a source that holds each person once (a
  voter roll), a customer with two of its records is cut.
- **Twin guard** (`"twin_guard"`): same surname and birth date but clearly different first names
  (not nicknames) are kept apart - twins share everything else a model counts as evidence.

### Households

After people are matched, **households** group customers who live together. Customers sharing a
standardised address are one household; streets are matched fuzzily within one house number (a
typo in "Jacaranda" does not split a family), and address households are joined when their
customers share a household-class contact and a surname. A customer with no address joins a
household only through such a shared contact. See "Households, fake contacts and
placeholders" in [report/05-customer-data.md](report/05-customer-data.md).

### Golden records and survivorship

A **golden record** is one record per customer, built field by field from the customer's
records. **Survivorship** is the rule that picks each field's surviving value:

| Rule | Picks |
| --- | --- |
| `most_common` | the value most of the customer's records hold |
| `most_complete` | the fullest value (William over Bill, a full date over a year) |
| `most_recent` | the value from the newest record (needs record dates) |
| `first_seen` | the value from the earliest source in the config |
| `longest` | the longest value |
| `source_priority` / `{"prefer": [...], "then": ...}` | a trusted source's value first, then another rule |

The address survives as one unit - number, street and unit from the same record - so a golden
address never mixes two. No rule can recover a value none of the records holds.

### Contacting people safely

Matching for analysis and matching for contact have different costs: a wrong merge that sends
one person's offer to someone else - often an ex-partner - is worse than a missed match.
`"min_precision"` trades recall for precision (step 8), and `"contact_confidence"` chooses a
customer's email and phone only among records joined by strong links, preferring personal
values over household ones. `contacts.csv` lists them. See "Contacting people safely" below.

### Over time: new records, stable IDs, drift

- **`custmatch add`** matches new, changed or deleted records against a saved run without
  retraining, keeping existing customers.
- **Stable customer IDs**: a rerun with `--previous` carries customer IDs forward, so a customer
  keeps its ID while its records stay together; merges and splits are logged in `history.csv`.
- **Hysteresis** (`"stability_margin"`, 0.05 by default with `--previous`): a pair's previous
  decision stands unless its score moves by more than the margin, so records do not flip
  between two people as data arrives. See "Stable matches as records arrive" below.
- **Monitoring** (`custmatch monitor`): match rates, field fill, score distributions and the
  largest customer compared across runs, with alerts when they drift.
- **Record dates** (`"record_dates"`): when records are dated, evidence ages - an old address
  disagreeing with a new one is expected, and the newest details can win survivorship.

### Cookies and device IDs

Web and app events carry an anonymous **cookie** or device ID, and sometimes an email or phone
when the visitor logs in. **Identity stitching** (`"identity_events"`) treats records that share
identifiers seen with one cookie as one person, with guards against shared devices (a family
laptop, a call-centre screen) that would otherwise join many people. See "Cookies and
device IDs" below.

### Explaining and reviewing

- **`links.csv`** lists every link with its evidence (which fields agreed), weakest first.
- **`custmatch explain`** says why two records were, or were not, matched: the score, the
  threshold, the evidence, overrides, and whether blocking ever compared them.
- **`review.csv`** lists pairs just either side of the threshold (`"review_band"`), the ones a
  person should check first.

### Privacy

- **`"hash_fields"`** replaces emails, phones or IDs with keyed hashes (HMAC-SHA256) after
  cleaning, so they are matched for equality but never stored in plain text.
- **`"minimal_outputs"`** keeps names and contacts out of the outputs.
- **`custmatch forget`** erases a person's records from a saved run, for erasure requests.

See "Privacy" below.

### Measuring quality

- **Precision**: of the pairs custmatch put together, the share that are truly one person.
- **Recall**: of the pairs that are truly one person, the share custmatch put together.
- **F1**: the balance of the two (their harmonic mean).
- **Pairwise** scores judge each candidate pair alone; **clustered** scores judge the final
  customers, counting every pair of records inside them. Clustered F1 is what matters - it is
  where chaining shows - and the figure the docs quote.
- **The ceiling**: the clustered F1 a perfect model would reach given the blocking (step 4).

**Auditing a run on your own records.** The benchmark figures in these docs were measured on
other people's data. To know how a run did on yours, label a sample and let custmatch estimate:

```bash
custmatch audit sample --state results/ --n 300 --out audit.csv     # fill label: 1 same person, 0 not
custmatch audit estimate --state results/ audit.csv
```

The sample is drawn about equally from twelve strata - six score bands, each split into the
pairs the run linked (put in one customer) and the pairs it did not - because most candidate
pairs score near 0 and would fill a plain random sample. The estimate starts from the run's own
scores summed over every candidate pair and corrects them by how far the labels differ from the
scores in each stratum (the difference estimator of survey sampling), so it is unbiased however
well the scores are calibrated. You get precision, recall and F1 of the run's links, each with a
95% interval. Recall is among candidate pairs: a true match blocking never proposed is not
counted (`custmatch blocking` reports what blocking misses).

Checked against the truth on the synthetic file (`research/scripts/audit_check.py`): one run,
true precision 98.87, recall 96.31, F1 97.57; over 100 audit samples of 300
pairs the estimates averaged 98.83, 96.09 and 97.44, with 95% intervals
1.47, 3.06 and 1.91 points wide that held the truth 91, 91 and 95 times in 100 - a little
narrow for 95%, so read them as approximate. Two simpler intervals did worse and are not used:
resampling the labels alone held it 85, 85 and 92 times (a stratum whose labels all
agree looks certain), and a Beta interval per stratum 87, 85 and 71 times.
The run saves what the sample is drawn from in `state/audit_pool.parquet`; `custmatch add`
does not, so audit the batch run.

Every number in the docs is checked against `research/results/*.json` by the test suite.

## Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
```

One opt-in model needs an extra: `uv pip install --python .venv/bin/python -e ".[catboost]"`
for `"matcher_model": "forest_catboost"` (report/03-which-matcher.md "Which classifier").

custmatch ships no third-party data. The first run downloads the US nickname list
(carltonnorthern/nicknames, Apache 2.0; 78 KB, pinned and checksum-checked) into
`~/.cache/custmatch`, or the folder `CUSTMATCH_DATA` names. Offline, copy `nicknames.csv` there
yourself, or turn nicknames off with `"normalise": {"nicknames": false}`.

Config keys custmatch does not read (a typo such as `"twin_gaurd"`) are listed in
`summary.json`'s warnings with the nearest known key.

## 1. Describe your files

A JSON config names each source and maps its columns onto the pipeline's. Only `record_id`,
`first_name` and `last_name` are required; map whatever else you have.

```json
{
  "sources": [
    {"name": "crm", "path": "crm.csv",
     "columns": {"record_id": "id", "first_name": "fname", "last_name": "lname",
                 "date_of_birth": "dob", "address": "street_address",
                 "email": "email", "phone": "phone"}},
    {"name": "billing", "path": "billing.parquet",
     "columns": {"record_id": "acct", "first_name": "first", "last_name": "last",
                 "addr_number": "house_no", "addr_street": "street", "ssn": "ssn",
                 "email": "mail", "phone": "tel"}}
  ],
  "source_priority": ["crm", "billing"]
}
```

| Pipeline column | Notes |
| --- | --- |
| `record_id` | unique within its source; outputs name records `source:record_id` |
| `first_name`, `middle_initial`, `last_name` | middle name is cut to its initial |
| `date_of_birth`, `ssn` | compared exactly; leave out if you do not hold them |
| `address` or `addr_number` + `addr_street` + `addr_unit` | one string is split into house number, street and unit |
| `email`, `phone` | phone punctuation is stripped; junk values are detected, not trusted |
| `country` | optional: the record's country (code or name) when a file spans several - see "Country profiles" |

Only `record_id` and `first_name` need values: a blank surname, date of birth or address is
treated as unknown.

**Country profiles.** Say which countries your customers are in, and custmatch reads their
records the way those countries write them:

```json
  "country": "US"
  "country": ["AU", "NZ"]        (several: the first is the default, a `country` column decides per record)
```

| | Dates | Postcode read off addresses | National ID | Nicknames |
| --- | --- | --- | --- | --- |
| `US` | month first | ZIP (98101, 98101-1234) | SSN, checked | English list |
| `AU`, `NZ` | day first | 4 digits | none | English + Australian / British list |
| `GB` | day first | UK postcode (SW1A 2AA) | none | English + Australian / British list |
| `CA` | day first (`"date_order": "mdy"` to change) | Canadian postcode (K1M 1M4) | none | English list |
| any other ISO code | day first | none | none | English list |

Every country also gets its phone numbering plan (`phonenumbers` knows them all), email
cleaning, names across languages and in other scripts, and blocking keys for records with no
surname. **SSN, checked:** a number the US never issues (area 000 or 666, group 00, serial 0000)
or a well-known placeholder (123-45-6789, 078-05-1120, one digit repeated) is blanked rather
than trusted; ITINs (9xx) are kept. Other countries get no national identifier - Australia's
TFN, for one, may not lawfully be used to match customers. The Australian / British nickname
list (`custmatch/data/nicknames_au_uk.csv`, 90 pairs written for this project: Bec, Shazza,
Davo, Lachie, and the Greek- and Italian-Australian Con, Jim, Joe, Sam) adds forms the US list
lacks. To add a country, add a profile to `custmatch/countries.py`.

On the public benchmarks, whose records are already in one clean format, a profile changes
little - mean of three label draws, random / active labels (`research/scripts/matcher_bakeoff.py` (measured with the threshold rule before 0.1.0, which took the strictest of tied thresholds; not yet rerun)):

| Dataset | Profile | Without | With |
| --- | --- | --- | --- |
| synthetic | `US` | 97.54 / 98.79 | 97.49 / 98.82 |
| febrl3 | `AU` | 99.50 / 99.98 | 99.46 / 99.92 |
| febrl4 | `AU` | 99.83 / 99.96 | 99.84 / 99.97 |
| historical_50k | `GB` | 86.29 / 87.28 | 86.77 / 86.45 |

All within the spread between draws. Its value is on messy files - phones written several
ways, mixed date orders, nicknames, placeholders - as in [report/05-customer-data.md](report/05-customer-data.md) "Sparse records".

In more detail, for files where names are partial and an email or mobile carries the match (see
[report/05-customer-data.md](report/05-customer-data.md) "Sparse records"), the country profile puts phones in one
international format (0412 345 678, +61 412 345 678 and 412345678 agree),
reads dates day-first outside the US, cleans emails (Gmail dots and +tags; me.com and mac.com as icloud.com; a big mail
provider's name mistyped by one letter, gmial.com or hotmail.con - real domains such as ge.com
are left alone), matches nicknames (Mick, Mike -> Michael) and names across languages
(Giuseppe -> Joseph, Kostas -> Konstantinos), transliterates names written
in Chinese or Arabic script, reads the postcode off addresses, and adds blocking keys for
records with no surname.

CSV and Parquet both work. Paths are relative to the config. `source_priority` breaks ties in
the golden record (most common value first, then the higher-priority source).
`examples/synthetic/config.json` is a working example.

**Data that is not shaped like a customer file.** Add your own fields and blocking keys:

```json
  "extra_fields":   [{"name": "birth_place", "compare": "exact"},
                     {"name": "postcode", "compare": "similarity"},
                     {"name": "date_of_birth", "compare": "edit"}],
  "extra_blocking": [["last_name", "date_of_birth"], ["postcode"]],
  "max_block": 200
```

`compare` is `exact`, `similarity` (Jaro-Winkler, for names and places) or `edit` (normalised
edit distance, for dates and codes). Naming a standard column, as `date_of_birth` above, adds a
fuzzy comparison of it - worth it when typos are common: in Splink's `fake_1000` data 58% of
true matches differ in date of birth. Each extra field must also be mapped in `columns`.
`max_block` (default 60) is the size above which a block is skipped as a hub; raise it when a
large block is a legitimately common value such as a city, not a junk contact. A skipped block
whose records the model judges to be one entity is compared anyway ("How it decides").

**One row per contact point (`"group_rows"`).** Many exports hold a person once per email or
phone - an address book, a CRM's contact-methods table. Set `"group_rows": true` on the source
and map `record_id` to the person's or contact's ID: rows sharing it become one record. Each field
takes the record's first filled value, and its other surnames, emails and phones become
aliases (below; named `last_name_2`, `email_2`, `email_3`, ...), so they are blocked and compared
without a second record per address:

```json
{"name": "contacts", "path": "contacts.csv", "group_rows": true,
 "columns": {"record_id": "contact_id", "first_name": "first", "last_name": "last", "email": "email", "phone": "phone"}}
```

Only surnames, emails and phones can carry several values; a second first name or address is
dropped (a record's nickname is matched through the nickname list anyway). As with aliases, it
works with the default blocking only. Splitting such a file into one record per address instead
makes pairs of a person's own addresses, which take up the labels: 61 of them on the address book
that prompted this option.

**Previous names and second contacts (`aliases`).** A record may carry other values of a
field - a maiden or previous surname, a second email, a work phone. Map each as a column of its
own and list it under its field:

```json
"aliases": {"last_name": ["maiden_name", "previous_surname"], "email": ["email_2"], "phone": ["mobile", "work_phone"]}
```

Surnames, emails and phones can have aliases. Each alias value is blocked like the field's own
value - a woman's maiden name meets the surname on her records from before she married - and a
pair is compared through its best-agreeing values: surname by similarity, email and phone
exactly, each field on its own, so a record's maiden name and its second email can both count
for one pair. Alias values are normalised as their field is, hashed with it under
`"hash_fields"`, profiled with it (a junk second email is still junk), and shown to labellers
("last_name (maiden_name)"). Outputs stay one row per record. Works with the default blocking
only (not `"blocking": "fixed"` or `"learned"`). The service's real-time matcher compares alias
values but finds a new record's candidates by its own values only; the batch run blocks on both.
On a test file where a quarter of one source's people had married and moved (new surname,
address, email and phone; the old surname kept as `previous_surname`), clustered recall rose
from 75.83 to 96.67 and F1 from 86.26 to 91.34. Precision fell from 100 to 86.57: every false
merge joined two different people with the same first name and surname, whom nothing else told
apart (`tests/test_custmatch.py`, test_previous_surnames_find_people_who_changed_their_name;
library versions move these by a point or two - Python 3.10's gave precision 87.88).

**Three matcher options.**

- `"rich_features"` (on by default): nine more comparisons for names and dates that drift -
  how rare an agreeing name is, first and last name swapped, matching initials, and graded dates
  (days apart, same year, day and month swapped, a one-character typo). They lifted
  historical_50k from 82.1 to 87.0 clustered F1 and fake_1000 from 88.2 to 92.5, and cost at
  most 0.3 on the febrl sets. `"rich_features": false` turns them off.
- `"match_weight_feature": true` (opt-in): a Fellegi-Sunter match weight, fitted without labels
  on all the candidate pairs, as one more input for the forest (see [report/06-blocking-and-scale.md](report/06-blocking-and-scale.md) "A match weight of
  custmatch's own"). It helps most where labels are thin, but costs the web-sign-up
  scenario (a first name and an email only). It replaces the Splink feature, removed in 0.1.0.
- `"matcher_model": "forest_catboost"` (opt-in; `pip install "custmatch[catboost]"`): the
  forest's and CatBoost's match probabilities averaged. +0.41 clustered F1 on average over six
  datasets, never more than 0.03 behind on a dataset's mean; the gains are on the hard files -
  historical_50k +1.3, a sparse file +1.8 with random labels - for runs about 30% longer (on
  files of up to 50,000 records) and a 100 MB dependency. The
  service's score breakdown (the waterfall) is for the forest alone, so it is not shown with this
  model; feature importances are. The server image includes it only when built with
  `CUSTMATCH_EXTRAS=catboost` (server/README.md). See [report/03-which-matcher.md](report/03-which-matcher.md) "Which classifier".

**Placeholder names, addresses and dates of birth.** `"placeholders": true` blanks values typed
to get past a mandatory field - "john doe", "test test", "unknown", "123 main st", "1 test st",
house number 0, 1900-01-01, 1899-12-31 (day zero in spreadsheets), any 9999 date - and any full
name, address or date of birth carried by many records that agree on nothing else (a form's
default date on 45 different adults). Where 1 January stands for "year only" (a tenth or more of
the file's dates) 1 January dates are kept: the year is evidence. Dates in the future are kept
too: they are typos of real dates. Only the flagged part is blanked ("maria unknown" keeps
"maria"), and `placeholders.csv` lists every record touched. Phones in the reserved fictional
range (xxx-555-0100 to 0199) are blanked with them. Off by default (see [report/05-customer-data.md](report/05-customer-data.md) "Households, fake contacts and placeholders").

**Suspect records.** `suspect_records.csv` lists records that look entered in bad faith or by a
bot, for a person to review; matching is not changed. Three reasons:

- *profanity in name / email*: a first name or surname that is, as a whole, a word on the
  LDNOOBW English list (CC BY 4.0, downloaded on first use, pinned and checksummed), or a
  whole word of the email's user part. A word that is a real name is never flagged: 36 of the
  list's 279 single words are the first name or surname of at least one of 5 million North
  Carolina voters - Hooker, Dick, Cumming, Felch, Negro, Bastardo - and are excluded. With that,
  no record was flagged in the voter files, nor, held out, in historical_50k, febrl3, fake_1000
  or the generated customer files (`research/scripts/suspect_research.py`). Whole fields and whole words
  only: "beaner-young", "na'shota", Scunthorpe and cocktail@ are not flagged.
- *disposable email*: a throwaway-mail domain (mailinator, guerrillamail, 10minutemail...).
- *random-looking email*: a part of the user name of 8 or more characters mixing letters and
  digits, alternating them (k7f2z9) or with under 15% vowels, that shares nothing with the
  record's names (`xk7fq2zr9p@`, not `jsmith84@`). A bot's random phone numbers cannot be told
  from real ones one at a time; a bot's default number is shared, and caught as junk.

It costs 2.4 seconds at a million records. A list that cannot be downloaded skips the
profanity check with a warning in `summary.json`. With `"hash_fields"`, the check runs on the
plain text before hashing. `"suspect_records": false` turns it off.

## 2. Label some pairs

The matcher learns from pairs labelled "same person" (1) or "different" (0). **Label at least
500**: on the synthetic customer file 500 labels gave 97.37 clustered F1 and 1,000 gave 97.28;
3,000 gave 97.80. Dense files want more (see below).

```bash
custmatch sample config.json --n 1000 --out to_label.csv
# fill the `label` column with 1 or 0 - the two records are shown side by side
```

**Better: let the forest choose.** Label 300 random pairs, then ask for the pairs the forest is
least sure about, 100 at a time, appending each batch to your labels:

```bash
custmatch sample config.json --n 300 --out labels.csv                   # label these
custmatch sample config.json --active labels.csv --n 100 --out next.csv  # label, append, repeat
```

With 1,000 labels in total this scored **98.61** clustered F1 against 97.28 for 1,000 random ones,
and it won on every one of three further label draws (98.85 / 98.81 / 98.90 against 97.58 /
97.74 / 97.78) - more than 3,000 random labels achieved (97.80). Stop at 500 and it does not
help yet (96.45 against 97.37). **On dense files it is inconsistent:** on historical_50k, with
about ten records per person, it scored 86.64 against 85.53 for random labels, but 86.20 against
87.02 on the code before 0.1.2, and drawing half of each batch at random (`--mix 0.5`) did not
settle it (87.13 against 87.55, with the Splink feature on). Where one person has many records,
compare both on your own data. Actively chosen pairs are marked `by = active`: `run` trains on
them but chooses its threshold only on the random ones, because a threshold judged on hard
pairs alone misjudges the clusters.

**Random pairs are drawn where the matches are.** Matches are rare among candidate pairs - a
fraction of a percent with millions of them - and 300 plain random pairs can hold two, too few
to judge the threshold on. `custmatch sample` draws half of them uniformly and half spread over
the pairs' agreement patterns (which fields agree exactly), and writes a `weight` column: the
candidate pairs each label stands for. Keep that column - `run` weighs each random label by it
when it judges the threshold, so the estimate stays unbiased. On 1,000,000 NCVR people with 300
random + 700 active labels it lifted clustered F1 from 80.84 to 92.88 (7 random matches instead
of 2); on synthetic address books with 300 random labels, from 71.58 to 77.21 over 30 draws,
with about twice the matches. `--no-stratified` draws plainly, as before 0.1.3; labels without
weights count equally. The losses measured: historical_50k with 300 random + 700 active
labels, 87.36 to 85.95 over three draws, and fake_1000 with 1,000 random, 93.36 to 92.44.

## 3. Run

```bash
custmatch run config.json --labels labels.csv --out results/
```

| Output | Contents |
| --- | --- |
| `clusters.csv` | `source, record_id, customer_id, household_id, confidence` for every record |
| `golden_records.csv` | one row per customer: the surviving value of each field |
| `shared_and_junk_contacts.csv` | emails and phones classified `junk` (never used as evidence) or `household` (shared by different customers) |
| `links.csv` | every accepted link: score, the evidence in words, and whether it is a bridge - weakest first |
| `suspect_records.csv` | records that look entered in bad faith or by a bot - profanity, a disposable or random-looking email - for review; matching ignores it |
| `summary.json` | counts, the chosen threshold, its cross-fitted quality, and warnings |
| `manifest.json` | what produced the run: the SHA-256 of the config's settings, every source file and the labels, the seed, the versions of custmatch, Python and the libraries that decide the numbers, and the SHA-256 of every output file (below) |
| `state/` | the records, trained forest and threshold, for `custmatch add` |

**Review `links.csv` from the top.** In one check on the synthetic file (not saved to
`results/`), the weakest 10% of links held most of the false merges (160 of 187), so review
there first. Ordering by bridges instead found 44: with about two records per customer most
links are bridges.

**Showing a run is what it claims.** The same records, labels, config and seed give the same
customers, and `manifest.json` records each of them by SHA-256. `custmatch verify results/`
checks that no output file - and no input file still at its path - has changed since the run;
`custmatch verify results/ --against rerun/` compares two runs and lists any difference in
inputs, settings, library versions or outputs (exit code 3 if there is one). For an auditor:
rerun from the inputs the manifest names, then verify the two folders against each other. Run
times are recorded but not compared (`summary.json` is fingerprinted without its stage timings,
`history.csv` without its event times). `custmatch add` and `forget` write a manifest too, naming
the manifest of the state they started from, so a chain of batches can be followed back. No
secret is written: the `hash_fields` key stays in its environment variable.

## Adding new records: `custmatch add`

```bash
custmatch add new_records.json --state results/ --out results_2/
```

Scores only the pairs that involve a new record, against the saved forest and threshold. A new
record joins the customer it links to, starts a new one, or - if it links two existing
customers - merges them, and `merges.csv` lists every such merge for review. Existing customer
ids never change except by a logged merge. With 10% of the synthetic file added this way,
clustered F1 was 96.82 against 97.13 for a full rerun, in 3.4 seconds instead of 5.2. Part of
that gap is the name-rarity feature: rarity is counted over the records present, so a forest
saved from a run over 90% of the file sees slightly shifted values (without rich features the
gap was 0.04). Contact profiles are recomputed as values spread, but past decisions are not
revisited: rerun `run` periodically.

New records are read with the saved run's settings (its `"country"` normalisation, extra fields
and name features), whatever the new config says beyond its sources, and placeholders are
blanked if the run blanked them. With `"max_cluster"`, a link that would grow a customer past it
is refused and listed in `refused_links.csv`. `add` does not re-cluster existing customers
(the `max_cluster` re-cut applies to `run` only). It loads the
saved model with pickle: only point `--state` at output folders you created ("Security" below).

## Running it over time

**Review queue.** Every run writes `review.csv`: the pairs whose score is within
`"review_band"` (default 0.1) of the threshold, most uncertain first, with the evidence. Fill in
`decision` with `same` or `different` and feed the rows back as overrides.

**Overrides and unmerge.** `"overrides": "overrides.csv"` names a file of `a, b, decision`
(record keys `source:record_id`; `same` or `different`). `run` and `add` both honour it: `same`
joins the two records whatever the model says; `different` keeps them apart even when other
records would chain them together - the customer is split by cutting the weakest link on the
strongest path between them, which is how you unmerge. `summary.json` counts what was applied.
It may be a list of files. An `add` config's own overrides file is kept with the state, so a
decision made with one batch holds for every later `add`.

**Trusted identifiers.** `"trusted_ids": ["loyalty_id"]` links records that share a value of
that column outright (a customer number, a verified account). A value on more than
`"trusted_max"` records (default 5) is ignored as a placeholder, and a `different` override
still wins.

**Stable IDs and history.** `custmatch run ... --previous results_last_week/` keeps customer IDs
where records overlap: a customer keeps the previous ID it holds most records of. Every run and
`add` writes `history.csv` - customers created, merged or split, records updated or deleted -
extending the previous run's. `--model results_last_week/` reuses that run's trained model and
threshold without labels, for scheduled reruns.

**Updates and deletes.** `custmatch add` treats a record whose key already exists as an update:
its old version is replaced and it is matched again, so a changed address can move it. `--delete
deleted.csv` (columns `source, record_id`) removes records; a customer that loses a record is
re-split on its saved links, so removing the record that joined two groups divides them again.

**Why, and why not.** `custmatch explain --state results/ crm:123 billing:456` says whether
blocking ever compared the two records (and which shared keys were skipped as oversized blocks),
the evidence and score against the threshold, any override, and whether they ended up together
directly or through a chain of other records.

**Which value survives.** Each customer's golden record takes one value per field from its
records. `"survivorship"` sets the rule - one name for every field, or per field with `default`
for the rest:

```json
  "survivorship": {"default": "most_complete", "email": "most_recent", "address": "most_recent",
                   "phone": {"prefer": ["app", "store"], "then": "most_recent"}}
```

| Rule | Picks |
| --- | --- |
| `most_complete` (the default) | The most common value; ties go to the source earliest in `"source_priority"`, then the longest |
| `most_common` | The most common value |
| `most_recent` | The value from the record with the latest `updated_at` (map that column); records without a date fall back to `most_complete` |
| `source_priority` | The value from the source earliest in `"source_priority"` |
| `{"prefer": [sources], "then": rule}` | The first listed source with a value, else the `then` rule (default `most_complete`) |
| `longest` | The longest value |
| `first_seen` | The first record's value, in file order |

`address` is one field for this: the house number, street and unit come together from one
record, never mixed. `"golden_fields"` chooses which fields get a golden value. A misspelt rule
or field stops the run with a message listing the valid ones. Blank values never survive.

**Record dates.** Map each record's date (last updated, or created) to `updated_at` and set
`"record_dates": true`. The matcher then sees how many years apart two records are, so it can
learn that evidence ages: a shared mobile years apart may be a reissued number, and a different
address or surname years later may be a move or a marriage. The same date drives
`most_recent` survivorship.

Measured on a dated build of the sparse Australian file (`research/scripts/record_dates_research.py`):
1,850 people move, 836 change surname and 1,459 mobiles are reissued to a stranger;
mean of three label draws, 1,000 labels each:

| Measure | Without dates | With `"record_dates"` |
| --- | --- | --- |
| Clustered F1, all records | 82.51 | 83.27 |
| Precision | 93.23 | 92.52 |
| Recall | 74.03 | 75.73 |
| One person's records across a move, kept together (%) | 69.73 | 71.05 |
| Across a surname change, kept together (%) | 62.89 | 63.16 |
| Recycled mobile: owner and stranger wrongly together (%) | 0.33 | 0.17 |

Dates lifted clustered F1 in all six runs - the three label draws above, and one draw on each
of three more dated builds (`--data-seeds`): by 0.5 to 1.2 points, 0.7 on average. Keeping people
together across a move improved in 5 of the six runs but by very different amounts (-0.1 to
+3.4 points), so which kind of change drives the gain is not settled. Reissued mobiles were
rare false merges either way (well under 1% of 4,070 owner-stranger pairs), too few to measure:
custmatch does not join two records on a phone alone when the names
disagree. `"record_dates"` is off by default because it changes the feature set, and a saved
model must match.

**Monitoring.** Every `run` and `add` writes `monitor.json`: the match rate (records in a
customer with others), overall and per source; customer sizes; how full each field is; and the
spread of match scores. `custmatch monitor` lines up outputs in time order and flags drift
between consecutive ones:

```bash
custmatch monitor results/2026-09 results/2026-10 results/2026-11 --out trend.csv --fail-on-drift
```

| Alert | When |
| --- | --- |
| Match rate | Overall or a source's rate moved by more than 5 points |
| Field fill | A field's fill rate moved by more than 10 points: a feed stopped sending a column, or a mapping broke |
| Scores | The score distribution shifted: population stability index above 0.2 |
| Largest customer | It more than doubled: check for chaining |
| Threshold | It changed: a retrained model, so expect IDs to move |

`--fail-on-drift` exits with code 3 when any alert fires, for a scheduler to stop on.

## Stable matches as records arrive

In some systems, as records arrive, a record keeps swapping between two people - usually two
partners - and back. The field calls this order dependence: the result depends on the order
records arrive in. The goal is what Senzing calls sequence neutrality (the same result whatever order records arrive
in), and Senzing holds a record that fits two people equally as "ambiguous" rather than choose.

`research/scripts/order_dependence.py` streams a slice of the sparse Australian file (1,200 households,
couples sharing surnames, addresses and family emails) into custmatch in 20 batches, and after
each batch compares every record's partners among the records already there. Mean of two
arrival orders:

| Method | Partner changes | Returns to earlier partners | Customer-ID changes | Final clustered F1 |
| --- | --- | --- | --- | --- |
| A full rerun after every batch, model retrained | 820 | 3.0 | 33,971 | 77.54 |
| Rerun with one fixed model | 382 | 2.5 | 39,878 | 77.54 |
| ... and customer IDs carried over (`--previous`) | 382 | 2.5 | 492 | 77.54 |
| ... and hysteresis 0.05 (the default with `--previous`) | 279 | 0.0 | 376 | 77.88 |
| ... and hysteresis 0.10 | 244 | 0.0 | 323 | 78.66 |
| `add` from the first model that labels allowed | 0 | 0.0 | 1,792 | 70.63 |
| `add` with the fixed model | 0 | 0.0 | 76 | 77.65 |

The "fixed model" was trained once on labels over the whole slice, including records that had
not arrived yet - more than a live system would have - and only two arrival orders were run.

- **Retraining causes about half of the partner changes.** A new model and threshold every run
  accounts for 53% of them (820 against 382 with one fixed model). Train deliberately (with new
  labels), then rerun with `--model` or match with `add`.
- **The rest is drift at the threshold:** name-rarity and shared-contact features shift as data
  arrives, nudging borderline pairs across. Hysteresis - keep the previous decision unless the
  score moves by more than a margin - removed about a quarter of them at 0.05 and over a third at
  0.10, with no loss of F1 in this run (77.88 and 78.66 against 77.54). An earlier run of the
  same experiment found 0.10 keeping old mistakes, so treat the size of the effect as uncertain:
  only two arrival orders were run. It is on by default (0.05) whenever `--previous` is given
  (`"stability_margin"` to change or switch off).
- **`add` with a good model never changes a record's partners** and scored level with a full
  rerun here (77.65 against 77.54): the steady state is `add` between occasional reruns.
- **The couple case.** In a second run, 162 records were given only a
  first initial, the couple's surname and their address, each belonging to one partner. Across
  every method they moved between partners 0 times: custmatch links record to record and
  holds a near-tie apart (with a full rerun, 143 of 162 stayed on their own, 18 joined their own
  person, 1 both partners and none the partner alone).
  Our explanation, not tested here: the swapping comes from assigning each record to its single
  best-matching customer and comparing it with that customer's combined profile, which shifts
  as either partner gains records; custmatch does neither. An ambiguity rule (hold a record between two people the model judged
  different) was tried: it cost 1 to 6 points without adding stability, and was removed.

## Cookies and device IDs

Marketing data - web and app events - carries an anonymous cookie or device ID on every event,
and an email or phone only when the visitor logs in or fills a form. Customer data platforms
(CDPs) stitch on this: a cookie seen with an identifier is that person's, so records holding
the identifiers it was seen with are one customer, and its anonymous history is theirs.

```json
  "identity_events": [{"path": "web_events.csv", "cookie": "anonymous_id",
                       "identifiers": {"email": "email", "phone": "mobile"},
                       "exclude": {"column": "channel", "values": ["call_centre"]},
                       "exclude_cookies": "staff_cookies.csv"}]
```

Event values are cleaned exactly as record values are, so they meet. Each run writes
`cookies.csv`: every cookie, the customer it was stitched to, or why not.

**The danger is over-merging** - "identity graph collapse" - where one identifier becomes a hub
that chains many people into one profile. CDPs guard against shared devices (Segment's
example is a store's shared iPad) with value limits and blocked values. custmatch has both,
and one more:

- **Value limit:** a cookie seen with more than `"cookie_max_identities"` (default 1) values of
  one identifier - two emails, two phones - is a shared device and stitches nothing.
- **Blocked traffic:** `exclude` drops events by a column (a call-centre channel) and
  `exclude_cookies` drops known internal cookies. Values profiled as a household's shared
  contact or as junk are never used, and a cookie reaching more than `"cookie_max_records"`
  (default 10) records is ignored.
- **Name check:** a call-centre agent working in their own browser for customers - one
  customer's email, the next customer's phone - passes any per-identifier limit. A cookie
  therefore stitches only when every record it reaches has a compatible first name (the same,
  a nickname, a typo, an initial); otherwise it is reported as `conflicting_names`.
  `"cookie_name_check": false` turns it off. With first names hashed only equal names and
  shared nickname roots count, so it stitches less.
- `exclude_cookies` is a CSV whose first column lists the cookies, under a header row.

Stitched links join records whatever the model says, like trusted IDs; a steward's
`different` still wins.

Measured on the sparse Australian file with generated events (`research/scripts/cookie_research.py`):
14,544 cookies - each person on one or two devices, logging in with an email or mobile
from their own records - plus 466 family laptops two people log into, 10 call-centre
agents who each served 40 customers and 100 who each served two (one by email, one by
phone). Mean of three label draws, 300 random + 700 active labels, on one generated set of
events - so the spread shows label noise, not how much another population would differ:

| Setting | Clustered F1 | Precision | Recall | Largest customer |
| --- | --- | --- | --- | --- |
| No cookies | 81.49 | 91.31 | 72.55 | 22 |
| Cookies, both guards (the default) | 83.87 | 90.83 | 76.96 | 23 |
| Cookies, no name check | 79.88 | 81.68 | 78.21 | 24 |
| Cookies, no protection | 6.98 | 3.65 | 79.89 | 714 |

- Stitching lifted clustered F1 by 2.2 to 3.0 points on every draw, almost all recall.
- The guards are what make it safe. The value limit stopped 961 cookies and the name check
  926. Without the name check, the light agents and cross-identifier shared laptops
  cost about 9 points of precision and left it below no cookies at all; with no protection the
  heavy agents collapsed hundreds of people into one customer.
- The value limit also refuses a person's own cookie when they use two emails; that is the
  price of the default 1. Raise it if your customers commonly do and your traffic is clean.

## Contacting people safely

Matching for analysis and matching for contact are different jobs. A merge that is right four
times in five is useful for counting customers; for sending an email, a wrong one reaches the
wrong person. Two settings, and a warning, are for the second job:

- **`"min_precision": 0.97`** picks the most complete threshold whose estimated precision
  (from your labels, checked inside clusters) is at least 97%, instead of the one with the best
  F1. If no threshold reaches it, the most precise one is used and `summary.json` warns. The
  estimate is only as good as the labels behind it. On the CDP scenarios a target of 0.97 gave
  about 98% precision and kept 47 to 58% recall, with a tenth or less of the wrong-person
  emails of a precision-tuned CDP - which found about 6 to 10 points more ("Reassessed for
  marketing"). A lower target keeps more.
- **`"contact_confidence": 0.95`** makes the contact details safe even where a customer is
  joined by weaker links. Emails and phones are chosen only among records joined by links at or
  above that score - a record's own, and records certainly the same person. `contacts.csv` gives
  each record the email and phone to use when acting on it, and the golden record takes its
  contact details from the customer's largest certain group. Matching itself is unchanged: a
  weaker link still joins records into one customer for counting and analysis; it just never
  decides whose inbox is used. Steward `same` decisions, trusted IDs and stitched cookies count
  as certain.
- **A warning when the labels are weak.** custmatch's precision depends on its labels: with 300
  random labels and no actively chosen ones, it was less precise than well-configured
  exact-identifier matching. `summary.json` now warns when every label was drawn at random, and
  when fewer than 30 of the random labels are matches - too few to judge the threshold on.

## Privacy

custmatch is for one organisation matching its own customers. It runs where the data already
lives, and it is that organisation's own copy of the data it reads. What it offers is
pseudonymised matching inside that organisation, not privacy-preserving linkage between
organisations (below).

**What `"hash_fields"` protects.** `"hash_fields": ["email", "phone", "ssn"]` replaces those
values with keyed hashes (HMAC-SHA256) as they are read. From then on they exist only as hashes:
in matching, in `state/`, and in every output, including the derived columns (an email's user
part; a hashed first name's nickname roots, each root hashed on its own). The key comes from the
environment variable `CUSTMATCH_SALT` (or the one `"hash_salt_env"` names), never from the
config; use the same key for runs you compare. Recognisably fake values (test@test.com,
0400 000 000) are dropped before hashing, since once hashed they could not be told apart.

**What it does not protect.**

- A hashed field is compared for exact equality only. That suits emails, phones and ID
  numbers; hash names or birth dates and typos, nicknames and swapped days stop matching (the
  nickname list still works, through hashed roots).
- Fields you do not list stay in plain text: usually names, birth dates and addresses, and
  whatever is derived from them (a postcode taken from an address).
- Whoever runs custmatch holds the key and the plain text it was given. Hashing guards
  against the outputs and saved state leaking; it does not hide the data from the operator. A
  key that leaks makes hashed emails and phones guessable by trying likely values.

**Other controls.**

- `"minimal_outputs": true` skips the files that repeat personal data: `golden_records.csv`,
  `shared_and_junk_contacts.csv`, `placeholders.csv`, `suspect_records.csv`, and identifier values
  in `cookies.csv`.
- `custmatch forget erase.csv --state results/ --out results2/` handles an erasure request: the
  records are removed from the customers, the links and the saved state, and the deletion is
  logged by record key only. Delete older output folders yourself.
- custmatch sends no customer data anywhere.
- `state/` holds the records; only point `custmatch add` at folders you created ("Security" below).

**Linking between organisations.** When two organisations must match records without either
seeing the other's - hospitals and registries, agencies, research linkage - that is
privacy-preserving record linkage (PPRL), and custmatch does not do it. The standard technique
encodes names, dates and addresses into Bloom filters that still allow fuzzy comparison;
unhardened encodings have been broken by frequency attacks, so the encoding needs care. Use a
tool built for it, such as CSIRO Data61's open-source anonlink (`clkhash` to encode, `anonlink`
to match). For exact matching between companies on hashed emails and phones, as data clean
rooms do, both sides can hash with an agreed key exactly as `"hash_fields"` does.

## How it decides

- **Contact profiling**: a value carried by many unrelated records (`test@test.com`,
  `1111111111`) is junk and never links anyone; one shared by different first names at one
  address is a household contact and counts for little. "Unrelated" is read from the records'
  surnames and addresses, and also from their *other* contact: people reuse the same made-up
  mobile (867-5309) and it collects their real emails, or the same made-up email and it collects
  their mobiles. An email on 8 or more records with mostly different mobiles - or a mobile with
  mostly different emails - and no shared surname or address is junk. This catches the fakes
  typed on sign-ups with no name, where names cannot disagree (the scenario of nameless
  sign-ups all typing one made-up mobile, in [report/05-customer-data.md](report/05-customer-data.md) "Scenarios"). Only values no one can own are junk on sight
  (0000000000, the 555 area code, the regulators' fictional ranges, test@ addresses). Numbers
  that merely look made up - runs (0411 111111, ...2345678), repeated pairs and triples
  (0412 121212), 867-5309 - are also sold as "gold" numbers, so they need confirming: junk only
  when 3 or more records carry one and they are different people (different names, or with no
  names, different other contacts) not at one address. One buyer's records share her name.
- **Blocking**: records are compared only when they share a name, street, address or trusted
  contact key; oversized blocks are skipped - unless they are one big entity. After training,
  the model scores 50 sampled pairs from each skipped block; a block whose pairs are mostly
  matches (an organisation with hundreds of records, all agreeing) is compared in full, up to
  a budget of a million pairs or the candidate count. Junk (`a@a.com`: people agreeing on
  nothing else) and many people (an apartment building) score as non-matches and stay skipped.
  `blocking_report.json` lists every block checked. On six person benchmarks it kept none and
  changed nothing; on dedupe's patent applicants it lifted clustered F1 from 6.93 to 75.88.
  `"check_large_blocks": false` turns it off.
- **Matcher**: a random forest over name, address, date-of-birth, SSN and rarity-weighted
  contact agreement.
- **Threshold**: chosen **after clustering**. Recall is the share of labelled matches whose two
  records land in the same customer, cross-fitted over 4 folds. Precision comes from the
  clusters themselves: record pairs are sampled *inside* each candidate threshold's customers
  and scored by the forest. Labelled pairs cannot supply it: they are drawn from blocking
  candidates, so they never see the false pairs that chaining creates between records that
  were never compared. On historical_50k that blind spot hid a 242-record false customer behind
  a 91.9 labelled-pair F1; the in-cluster check moved the choice from 0.85 to 0.9 and clustered
  F1 from 70.3 to 82.1. Thresholds the labels cannot tell apart - within half a held-out random
  match's worth of the best estimate - count as tied, and the middle one is chosen (below).
- **Households**: after matching, contacts are profiled again over *customers* - a value is a
  household contact only if different customers share it - and each customer goes to its most
  common standardised address (suffixes, directions and unit designators normalised, streets
  matched fuzzily within the same house number, a missing unit filled from the only unit seen
  there). Households whose customers share a household contact and a surname are merged, never
  above 7 customers. A customer with no address falls back to its household contact.

**Choosing among thresholds the labels cannot tell apart.** Only the randomly drawn labels judge
the threshold (actively chosen ones are deliberately hard and would bias it). When they hold few
matches, the recall estimate is flat across a wide range of thresholds while the precision
estimate keeps creeping up, so "the best estimate, ties to the stricter" drifted to the strict
end and lost real recall: on 200,000 NCVR voters with 8 random matches it chose 0.85 -
85.4 clustered F1 - where 0.6 gave 95.6. Since 0.1.0 every threshold within half a match's
worth of recall of the best (50 / random matches, in F1 points) counts as tied, and the middle of
them is chosen; `"threshold_rule": "strictest"` restores the old rule. `research/scripts/threshold_capture.py`
recorded custmatch's estimates and the truth at every threshold, and `research/scripts/threshold_rules.py`
scored rules against the best threshold in hindsight:

| Cases | Rule | Mean F1 lost | Worst case | Better / worse than before |
| --- | --- | --- | --- | --- |
| 15 development cases (NCVR, Australian, synthetic), used to choose the rule | before: strictest of ties | 0.84 | 10.20 | |
| | middle of near-ties | 0.14 | 1.07 | 6 / 5 |
| 24 held-out public benchmark cases (febrl3, febrl4, fake_1000, historical_50k) | before: strictest of ties | 0.60 | 4.67 | |
| | middle of near-ties | 0.27 | 1.75 | 11 / 1 |

Taking the least strict of the near-ties instead was worse than either. The rule cannot conjure
information the labels lack: with few random matches, label more random pairs - `summary.json`
warns below 30.

## Read the warnings

- **A large `largest_customer`** (over about 20 records): look at that cluster. It usually
  means one bad link is chaining different people together.
- **"blocking switched to adaptive"**: the default keys would have left over 1% of the records
  compared with no one, so their oversized blocks were split instead ("Big files" under
  Blocking). Expect more candidate pairs and a longer run; `"blocking": "core"` turns it off.
- **"... a no-reply or notification address and a person's name"**: a service's relay filed
  under someone's name ("Junk values" under Contact profiling). Its address is ignored, but its name can
  still link it to that person; remove such records if they are not people.
- **"... was blank in one part for many records"**: a default key blocked on one field alone
  ("Half-blank keys" under Blocking), so many candidate pairs share only, say, a first name.
  Random labels drawn from them hold few matches: label more, or fill in the missing field.
- **Fewer than 500 labels**: below 500 was not measured. Single draws at 500, 1,000 and 3,000
  labels scored 97.37, 97.28 and 97.80 on the synthetic file - within the spread between draws -
  and sparse files needed active learning far more than more random labels.

## Security

### Customer data

custmatch runs locally and sends no customer data anywhere.

The first run makes one network request that carries no customer data: it downloads the US
nickname list (carltonnorthern/nicknames, pinned to a commit and checked against its SHA-256)
into `~/.cache/custmatch` (and, for suspect records, the LDNOOBW word list the same way). To
avoid them, put the files there yourself ("Install" above).

The outputs (`clusters.csv`, `golden_records.csv`, `links.csv`, `contacts.csv`, `review.csv`,
`state/`) contain your records. Treat an output folder with the same care as the input files.
`"hash_fields"` and `"minimal_outputs"` reduce what they hold ("Privacy" above).

### Loading saved state

Three commands load the model saved in an output's `state/` folder with Python's pickle, which
can run arbitrary code: `custmatch add --state`, `custmatch run --model` and `custmatch explain
--state`. **Only point them at folders you created**, never at one you received from someone else.

### custmatch-server

The service stores your source records, their history and the matching results in its
database, and its data directory holds each run's files (including the pickled model - the
warning above applies to it). Protect both as you would the source systems.

- **Every API route but `/health` needs an API key** (`custmatch-server key create NAME --role
  ROLE`); keys are stored as SHA-256 hashes and shown once. A `read` key sees personal data:
  give out the least role that does the job.
- **`CUSTMATCH_AUTH=off` turns authentication off** - for a local trial only: the server then
  refuses to listen on anything but 127.0.0.1, refuses four-eyes approval (nobody is identified),
  and logs a warning.
- **The server speaks plain HTTP.** Put it behind a TLS-terminating proxy or load balancer. The
  Compose file publishes the API on 127.0.0.1 only; set `CUSTMATCH_DB_PASSWORD`.
- **The run's model is a pickle.** The API loads it only from inside `CUSTMATCH_DATA_DIR`, and
  Compose mounts that read-only into the API; protect write access to it like code.
- **Four-eyes (`CUSTMATCH_FOUR_EYES=on`)** covers every steward action that changes customers:
  decisions, merges, splits and revocations wait for a different, named steward.
- Limits on every list endpoint are capped; the dashboard is served with a content security policy
  (its script allowed by hash), `X-Frame-Options: DENY` and `nosniff`; responses are `no-store`;
  decision exports are safe to open in a spreadsheet (formula-like cells are prefixed).
- The image runs as a non-root user and needs no network at run time (name lists are fetched,
  checksum-verified, at build).
- `server/sql/roles.sql` sketches least-privilege database roles for Postgres.
- Erasure (`custmatch forget`) works on the library's files; the service does not yet erase a
  person from its database, retained runs and audit log in one operation (on the roadmap).

### Reporting

Please report security problems privately through GitHub's "Report a vulnerability" (Security
tab of the repository) rather than in a public issue.

## Glossary

| Term | Meaning |
| --- | --- |
| Active learning | Choosing the pairs to label that the model is least sure of |
| Blocking key | A value records must share to be compared |
| Candidate pair | A pair blocking proposes for comparison |
| Chaining | Borderline links joining many records into one customer |
| Closure-aware F1 | F1 counted after links are joined into customers |
| Connected components | Groups formed by following links: A–B and B–C make A, B, C one group |
| Cross-fitting | Scoring each part of the labels with a model trained on the other parts |
| EM | Expectation maximisation: fitting a model's probabilities without labels |
| Fellegi-Sunter | The classic probabilistic matching model, scoring fields by m and u probabilities |
| Golden record | One best record per customer |
| Hub | A block too large to compare, skipped |
| Hysteresis | Keeping a previous decision unless the evidence moves by a margin |
| Jaro-Winkler | A 0-1 string similarity that favours a shared beginning |
| Levenshtein | The number of single-character edits between two strings |
| LSH | Locality-sensitive hashing: similar values land in the same bucket |
| Random forest | Many varied decision trees whose votes are averaged |
| Soundex | A phonetic code for names: Robert and Rupert are both R163 |
| Survivorship | The rule picking each golden-record field |
| Term frequency | How common a value is; agreement on a rare one counts for more |
| Threshold | The score above which a pair is a link |
| Transliteration | Writing a name from another script in Latin letters |
