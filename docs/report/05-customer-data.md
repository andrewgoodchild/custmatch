# 5. Customer data: fake contacts, households and sparse records

The problems that make customer data different: placeholder and shared contact values,
households, sparse web sign-ups, and the hard cases (twins, couples, two John Smiths) collected
into a scorecard.

## Are shared and fake contact values detectable without labels?

`scripts/shared_value_detection.py`. Two failure modes that look identical at first glance —
one value carried by many records — but which need opposite treatment.

### Householding

**FEBRL cannot show this problem at all.** Of its 7,341 distinct addresses, 1,762 appear on more
than one record and **zero** are shared across different people; its SSN-like id behaves the
same way (4,561 repeated values, none spanning two entities). One person per address, by
construction.

historical_50k does show it, using postcode as an address proxy:

| Postcode | Records | Distinct people | Distinct surnames | Same-entity pair % |
|---|---|---|---|---|
| se1 7sg | 34 | 5 | 10 | 25.0 |
| sw1h 9aa | 28 | 4 | 6 | 32.0 |
| se1 8xz | 28 | 6 | 11 | 16.9 |
| pl1 3dq | 27 | 2 | 8 | 48.2 |

Only **25%** of the record pairs sharing the most common postcode are the same person. A rule
that treats address agreement as strong evidence is wrong three times in four on exactly the
values that fire most often.

### Disguised missing values

The academic name for `test@test.com` and `0000000000` is **disguised missing data** (Hua &
Pei, KDD 2007): syntactically valid, so no null check catches them, but semantically absent.

Neither public dataset has an email or phone field, so I injected them into FEBRL — 4% of
records given a placeholder, 1.5% given genuine household-style sharing. An unsupervised
detector separated them cleanly: `test@test.com` and `0000000000` flagged (400 records each),
of whose record pairs **0.01%** are the same person, while 75 household-like values (2–4
records, low surname diversity) were kept separate.

### The signature

Frequency alone cannot tell them apart; the diversity of the surrounding fields can.

| | Records per value | Co-field diversity | Treatment |
|---|---|---|---|
| Real personal identifier | 1–2 | low | full weight |
| Household share | a few | **low** (same surname or address) | household evidence, not identity |
| Junk / disguised missing | dozens+ | **high** (nothing agrees) | drop the value |

[Chapter 5](05-customer-data.md) puts this to work, where the same detector reaches 100% precision and recall on junk
values and 96.9% recall on genuinely shared household contacts.

## A customer-matching process that handles fake contacts and households

`scripts/customer_data.py`, `scripts/customer_pipeline.py`.

Built on **pseudopeople**, because it is the only public option with both `simulant_id`
(identity truth) and `household_id` (household truth) — you cannot otherwise tell an
over-merge from a household co-occurrence. Two sources with deliberately different fields,
as in real MDM:

- **CRM** = Decennial Census 2020: names, DOB, residential address, 10,231 records
- **Billing** = W-2/1099 2020: names, SSN, mailing address, employer, 9,903 records

No shared strong key: no DOB in billing, no SSN in the CRM. 10,735 people, 4,305 households,
6,712 people present in both sources. Email and phone were injected with a labelled mix —
50% personal, 25% household-shared, 15% junk placeholders (`test@test.com`, `0000000000`,
`5555555555`, …), 10% missing — plus a 10% chance a person's contact differs between sources.

### The pipeline

| Stage | What it does |
|---|---|
| 1. Profile contacts | Classify every email/phone value as personal / household / junk from value frequency, co-field diversity and syntax. No labels. |
| 2. Block | Multi-pass on name, address and contact, with hub suppression; junk values never become blocking keys. |
| 3. Features | Similarity features; contact agreement weighted by how *rare* the value is (`1/(1+log n)`), junk treated as missing, household contacts down-weighted to 0.35. |
| 4. Match | Random forest over candidate pairs, trained on a disjoint half of the people. |
| 5. Cluster | Connected components, threshold tuned on the **clustering** metric. |
| 6. Households | A separate layer: each customer placed at its modal address, households formed by address — *not* by closure. |

### Results

| | Pathology-aware | Naive |
|---|---|---|
| Candidate pairs | **48,759** | 1,064,490 |
| Blocking recall | 99.5% | 99.7% |
| Pairwise F1 | 98.16 | 98.22 |
| Clustered F1 | **97.81** | 97.39 |
| Adjusted Rand | **0.978** | 0.974 |
| False merges | **85** | 98 |
| Junk contact detection | **100% P / 100% R** | not attempted |
| Household contact detection | 57.9% P / **96.9% R** | not attempted |
| Household partition | **90.2 F1**, ARI 0.902 | not attempted |

**Junk detection is exact**: all 8 junk emails and all 7 junk phones found, nothing else
flagged. The rule that does it is `frequency ≥ 8 AND surname-diversity > 0.5 AND
address-diversity > 0.5`, plus syntactic patterns. The address-diversity clause is what
separates junk from a large household — a big family is diverse in surname but *not* in
address; `test@test.com` is diverse in both.

**Household-shared contacts**: 96.9% recall at 57.9% precision, over the 444 values actually
shared by two or more people. The discriminator that made this work is **first-name
diversity** — John and Jane share a household address, whereas one person's duplicate records
across two systems share a first name. Recall is the metric that matters here: the cost of
missing a shared contact (trusting it as identity evidence) is far higher than the cost of
being over-cautious with one.

**Households**: 90.2 F1 against `household_id`, at 88.2% recall against a ceiling of
89.5% — the share of true household pairs that agree on an exact address. So the household
layer is essentially at the limit of what the address field supports. (An earlier version
reported 89.6: each customer's household is their most common address, and ties were broken
by iterating a Python set, whose order changes with per-run string hashing. Ties now go to the
address seen first, and the result is identical across hash seeds.)

### Two things this run taught me that I did not expect

**1. The supervised matcher largely defends itself.** The naive pipeline — contacts trusted as
keys, no junk detection — still reached 97.39 clustered F1 with its threshold tuned pairwise,
and **97.79** with it tuned on clustered F1 like the aware pipeline's 97.81. A random forest
trained on data that *contains* junk contacts learns on its own that an email match with
mismatched names means little. Tuned the same way, the pathology handling is worth about
nothing in F1; the 0.4 difference in the table comes from how the naive threshold was chosen.
(Both thresholds were picked on the test people's truth; held out, the aware pipeline scores
98.53 - the Ditto comparison below.)

Where it is decisively worth it is everything else: **22× fewer candidate pairs** (1.06M → 49k,
because junk values form enormous blocks - against a naive version with no block-size cap),
a junk list you can feed back to form validation (100% precise here by construction: this
file's junk values were written by the same author as the junk patterns), a household layer
you cannot otherwise build, and weights that are auditable rather than implicit in a forest.

**2. Whether the pairwise threshold is dangerous depends on graph density.** [Chapter 2](02-whole-customers.md) found a
93.4%-precise matcher collapsing 8,200 records into one cluster on historical_50k. Here the
same mistake costs almost nothing — the largest cluster is 9 records, and pairwise-tuned
clustering lands within 0.5 F1 of cluster-tuned. The difference is records per entity:
historical_50k has ~10 with dense blocking, this file has ~2. **A hypothesis from these two
files, which differ in many ways at once:** chaining is a percolation effect, so the risk
grows with how many records per customer you hold. A file with one row per customer per system
would then be fairly safe; a file with years of interaction history would not.

### Ditto on these records

`scripts/customer_ditto.py`. Ditto had only been run on historical_50k, where it beats the forest
by 11 clustered F1. Here it is trained on the pipeline's own training people (85%, with 15% held
back to pick the epoch; 12,108 pairs, 3 of 5 epochs, ~45 minutes on the laptop) and scores the
pipeline's own 14,997 test pairs. It sees the fields the forest's features use, with contacts the
profiler flags as junk blanked, as the forest's features already treat them — so the matcher is
the only difference. Thresholds are tuned on one half of the test people and scored on the
other:

| Customer pipeline, 14,997 test pairs | Picked on test | Held out (A / B) | Mean held out |
|---|---|---|---|
| Random forest (the pipeline above) | 97.81 | 98.49 / 98.58 | 98.53 |
| Ditto alone | 98.69 | 99.00 / 99.28 | **99.14** |
| Forest veto, then Ditto | **98.78** | 99.12 / 99.16 | **99.14** |

**Ditto helps here too, but by under a point** — against 11 on historical_50k — and on both
independent halves. The forest row reproduces the pipeline's 97.81 exactly, so the comparison is
like for like. (The held-out figures sit higher than the picked-on-test ones because each half
drops the pairs that cross into the other, and with them some chances to merge wrongly.) The
gap is small because the forest has little left to miss: with exact addresses, emails and phones
on most records, the easy evidence is already decisive, which is also why the best zero-shot LLMs
and the forest tie on these records in [chapter 10](10-reproductions.md). [Chapter 7](07-golden-records.md) prices what the remaining point is
worth.

## Households, fake contacts and placeholders

What the published practice suggests, each idea scored against the synthetic file's true
households and contact classes (`scripts/household_research.py`, 1,000 random labels):

- **Contacts profiled over customers.** Before matching, a shared email or phone is called a
  household contact when records with different first names carry it - but one person with a
  nickname or typo in two sources looks like two people. Re-profiling after matching, counting
  distinct *customers*, raised household-contact precision from 57.9% to 82.1% (100% with
  perfect matching) at about 96% recall. Junk detection was already 100/100.
- **Standardised and fuzzy addresses** (as the Census Bureau's address units): suffixes,
  directions and unit designators normalised alone changed nothing on this file; matching the
  street fuzzily within one house number took household F1 from 90.15 to 91.34, recall from
  88.05 to 90.10.
- **A missing unit** filled from the only unit at that address, and **merging on a shared
  household contact plus surname** across addresses, capped at 7 customers (the rule in patents
  US6985926 and US11860909B2): +0.03 together, now the default.
- **Rejected:** discarding address groups over 7 as institutions, as US Bank's patent does. On
  this file group quarters are real households and it cut household F1 to 22.5.

Household F1 in the tool's validation is now 90.9 (90.0 before these changes; 91.3 before two fixes from a code
review - not merging streets that differ in a direction word, and not
treating two blank surnames as the same - which cost about half a point on this file).

**Fake emails and phones.** The research lists reserved fictional numbers (US xxx-555-0100 to
0199, UK and Australian drama ranges), numbering-plan validity, and open lists of disposable
and role email domains. With `"placeholders": true` numbers in the US fictional range are
blanked too; off by default, like the rest of placeholder detection. Validity rules were
not adopted: on the synthetic file 1,570 records carry numbers that break the North American
plan (the generator ignores it), and an invalid number carried by one person still identifies
them. What makes a contact useless as evidence is being shared by unrelated people, which the
profiler already measures.

**Placeholder names and addresses** (`"placeholders": true`, `scripts/placeholder_research.py`).
Pattern rules (known fake full names, placeholder words, keyboard runs, a character repeated four
or more times, house number 0, "main / test / fake / any street") plus the diversity rule used
for junk contacts: a full name on 5 or more records with mostly different dates of birth and
addresses, or an address on 15 or more records with mostly different surnames and dates of
birth. Deliberately *not* flagged, because each hit real people in the benchmarks: digits in
names (typos - 2,230 historical_50k records), single letters (initials), honorifics and
suffixes ("sir", "iii"), surnames such as Tester and Guest, and house numbers 1234 or 9999.

| | Result |
| --- | --- |
| Flags on clean data | 0 on febrl3, historical_50k and fake_1000; 23 on the synthetic file, all genuine (surnames "none", "occupant", "xxx" - the same person's other records carry a real one) |
| Detection, 3% of records given a fake name and 3% a fake address | names P 96.42 / R 100 (the misses in precision are those 23 genuine ones); addresses P 100 / R 100 |
| Matching on that file | clustered F1 93.90 -> 94.86; households 87.79 -> 90.32 |
| Matching on the clean synthetic file | clustered F1 97.28 -> 97.29; households 90.94 -> 90.94 |

It helps where placeholders exist and costs nothing measurable where they do not. It stays
opt-in only because no benchmark here has real-world placeholder data; turn it on for any file
filled in by staff or web forms with mandatory fields.

## Sparse records: first name plus email or mobile

Outside the US there is usually no SSN, and retail channels capture different fields: a web
sign-up gives a first name and an email, a loyalty app a first name and a mobile, a store a full
name and an address. No public benchmark has that shape with ground truth, so
`scripts/au_customer_data.py` generates one for Australia: 18,478 records of 9,721 people in
5,200 households across web, app and store; 37% of records have no surname. Names follow real
frequencies (US Census surnames, Social Security given names by birth year, and Chinese,
Indian, Italian, Vietnamese, Greek, Filipino and Lebanese names in roughly Census proportions),
with nicknames, anglicised names (Wei -> Kevin), family-name-first order, one-name people,
married-name changes, typos, mobiles in every format, changed and recycled numbers, shared
family emails and mobiles, Gmail dots and +tags, mistyped domains, junk contacts (0400 000 000,
ACMA's fictional 0491 570 xxx, test@test.com), unit addresses written 3/12, Unit 3 or U3, and
day-first dates. It is a stress test of our own assumptions, not a substitute for your data:
check on a few hundred labelled pairs from your own file.

`scripts/au_validate.py` runs custmatch on it as a user would, over three label draws (mean
+- spread):

| Setup | Clustered F1 | No-surname records | Households |
| --- | --- | --- | --- |
| As before, 1,000 random labels | 54.93 +- 3.75 | 49.62 | 38.41 |
| As before, 300 random + 700 active | 70.07 +- 3.73 | 64.50 | 45.75 |
| `"country": "AU"`, 1,000 random labels | 70.22 +- 1.08 | 70.06 | 46.80 |
| **`"country": "AU"`, 300 random + 700 active** | **81.69 +- 0.74** | **78.74** | **53.74** |
| Ceiling: every true candidate pair accepted | 93.27 | | 66.47 with true identities |

- **Use active learning** (`custmatch sample --active`). Only about 5% of candidate pairs are
  matches here, so 1,000 random labels hold about 50 of them; active learning finds about 400.
  With `"country"` it scored 81.69 against 70.22 with random labels. Before 0.1.0's threshold
  rule a single draw could still go wrong (without `"country"`, one active-learning draw scored
  49.08); now the worst of three is 64.80.
- **Set `"country"`.** With active learning, removing one step at a time costs: the English nickname list 5.90, phone format 5.13, email cleaning 1.83, day-first dates 1.36, the Australian / British nickname list 1.27, the email user name 0.91, the postcode read off addresses 0.77.
  Every step helped in this rerun; in earlier ones, made while a cross-language name list from
  Wikidata (since removed) was also on, email cleaning, the email user name and the postcode
  looked neutral or harmful. Three draws cannot rank the small effects: across all the reruns
  only phones and the English nicknames were clearly positive every time. Phones are the
  big one: the same mobile written 0412 345 678 and +61 412 345 678 was two different numbers.
  The English nickname list is independent of the generator (carltonnorthern/nicknames, US given
  names), so its gain is not circular; the Australian / British list shares forms with the
  generator (Bec, Mick, Con, Joe), so its gain here partly is.
- **Junk contacts on sparse records** were being misread. The profiler measured surname and
  address diversity counting blanks as a shared value, so `na@na.com`, typed by 223 unrelated
  web customers with no address, looked like one household's email. Blanks now count as unknown
  (this also changed 79 of fake_1000's 424 email classes; results below are rerun with it). And
  common fake mobiles - long runs of one digit (0400 000 000), 0412 345 678, ACMA's fictional
  ranges - are junk by pattern.
- **Most of the rest cannot be matched.** 20% of same-person record pairs share nothing any
  blocking key can see (a web record with an email against an app record with a mobile, no
  surname, no date of birth). Households are capped by addresses: 52.2% of people have one on
  any record, so even perfect matching gets 66.47 household F1. Asking for a mobile at the till,
  or a surname at sign-up, would do more than any matcher change.

Seeded runs are not bit-for-bit repeatable: the forest sums its trees in whatever order its
threads finish, and the tiny differences can change which pair active learning picks next.
Read differences under about half a point between configurations as noise.

**Names in another script.** A second build of the file (`au_customer_data.py --scripts`) writes
40% of Chinese and Lebanese customers' store records in Chinese characters or Arabic script -
256 records, everything else identical. `"country"` transliterates them (see the column notes
above). For the people concerned, clustered F1 with active learning was 68.46 without
transliteration and 77.54 with it; the same people scored 78.85 when every name was in Latin
script, so transliteration recovers 9.08 of the 10.39 points lost (87%). It is a small group (256
records); the share recovered moved with each rerun (60% while the Wikidata name list was on). The consonant-skeleton rules were written with this
file's 25 Arabic names in view, so treat the Arabic half as a demonstration; Chinese needs no
such rules (one of the 50 surnames, 曾, transliterates as "ceng" rather than "zeng").

**Households without addresses.** A customer with no address can only join a household through
a shared household contact. `"household_postcode_join": true` also lets it join the one address
household with its surname in its postcode (web sign-ups give a postcode; `"country": "AU"`
reads one off the end of store addresses). With true identities it raised household F1 from
66.47 to 67.68 but cut precision from 95.46 to 92.13 - the same surname in the same postcode is
often relatives nearby - so it is off by default. Two customers with no surname no longer count
as sharing one when merging households on a shared contact.

## Scenarios: a scorecard of hard cases

`examples/scenarios/` is a US customer file built to test, case by case, the things that go
wrong in customer matching (`scripts/scenarios.py build`). It holds 1,000 ordinary people
in households - couples sharing addresses, phones and family emails, grown children at home,
with typos, nicknames, missing fields and mistyped birth dates - and 26 scenarios, each
repeated 10 times with different people. `expected.csv` gives the right answer for every
scenario pair. custmatch is trained on labels from the ordinary records only. It is a development set, not
a held-out test: building it found two gaps that were then fixed (below), so its scores flatter
the tool. A repeat passes when every pair in it comes out right; the table
gives the share passed, mean of three label draws (`scripts/scenarios.py score`):

| Scenario | Answer | 1,500 random labels | + `"twin_guard"` | 500 random + 1,000 active labels | active + `"twin_guard"` |
| --- | --- | --- | --- | --- | --- |
| A typo in the first name (Jonathon / Jonathan) | same | 100% | 100% | 100% | 100% |
| A nickname (Bill / William) | same | 100% | 100% | 100% | 100% |
| First and last name swapped | same | 100% | 100% | 100% | 100% |
| A new surname after marriage; same date of birth and email | same | 100% | 100% | 100% | 100% |
| Moved house: a new address and phone; same name, date of birth and email | same | 100% | 100% | 100% | 100% |
| Gmail with dots and a +tag (j.smith+shop@gmail.com / jsmith@gmail.com) | same | 100% | 100% | 100% | 100% |
| One phone in different formats (+1 206 555 1234 / (206) 555-1234); no email | same | 100% | 100% | 100% | 100% |
| Day and month of birth swapped | same | 100% | 100% | 100% | 100% |
| One record has only a first name and email (a web sign-up) | same | 100% | 100% | 100% | 100% |
| First initial only, with the same email | same | 100% | 100% | 100% | 100% |
| A double-barrelled surname on one record (Garcia-Lopez / Garcia) | same | 100% | 100% | 100% | 100% |
| Case, spaces and punctuation (O'BRIEN  JOHN / obrien john) | same | 100% | 100% | 100% | 100% |
| Accents on one record (José / Jose, Zoë / Zoe) | same | 100% | 100% | 100% | 100% |
| Apartment written differently (12 Main St Apt 4 / 12 Main St #4) | same | 100% | 100% | 100% | 100% |
| Both records sparse: first name, surname and phone only | same | 100% | 100% | 100% | 100% |
| Twins: same surname, birthday, address and phone; different first names | different | 0% | 100% | 0% | 100% |
| Father and son with the same name and address (Sr / Jr); different birth years | different | 100% | 100% | 100% | 100% |
| A couple sharing surname, address, family email and phone | different | 100% | 100% | 100% | 100% |
| Two John Smiths: same name, different birthdays, addresses and phones | different | 100% | 100% | 100% | 100% |
| Strangers with the same mobile (a reissued number) | different | 100% | 100% | 100% | 100% |
| Strangers who all typed test@test.com and 555-555-5555 / 0000000000 | different | 100% | 100% | 100% | 100% |
| Unrelated people at one address | different | 100% | 100% | 100% | 100% |
| A record with one person's name and another's email and phone must not join the two | different | 100% | 100% | 100% | 100% |
| Sisters at one address, both recorded by first initial only | different | 100% | 100% | 100% | 100% |
| Nameless sign-ups, each with their own email, all typing one made-up mobile | different | 100% | 100% | 100% | 100% |
| A bought gold mobile ((212) 222-2222): a sign-up with only a first name and it | same | 100% | 100% | 100% | 100% |
| **Clustered F1 on the background records** | | **97.86** | **97.66** | **99.44** | **99.24** |

- **Everything but twins passes with ordinary labels.** Name variation, moves, marriage,
  contact formats, spouses, father and son, two John Smiths, reissued phones, junk contacts,
  roommates and a record built to chain two people together are all handled.
- **Nameless sign-ups sharing a made-up mobile** were linked as one person in every draw before
  contact profiling read the other contact (0% passed; `default_before_contact_junk` in
  scenarios.json, the code at ad62c08 on the same file and labels): with no names, nothing
  disagreed but the emails. Ten different emails on one mobile now make it junk, and every
  repeat passes.
- **A bought gold mobile** ((212) 222-2222) on a sign-up with only a first name was never linked
  while numbers that look made up were junk on sight (0%; `default_before_patterned_numbers`,
  the code at bc0301a on the same file and labels): the number was the only link, and it was
  thrown away. People buy such numbers, so a pattern now needs strangers to confirm it, and
  every repeat passes.
- **Twins are linked every time** without help: they share everything a model counts as
  strong evidence, and twins are too rare for random labels to teach it otherwise.
  `"twin_guard": true` keeps apart records with the same surname and full birth date whose
  first names are written out, clearly different and not nicknames of each other. It fixed
  every twin repeat for about 0.2 F1 here, and cost 0.07 on the sparse Australian file
  (81.69 to 81.62). A steward's `same` override still wins.
- **A sign-up with only a first name and an email** is the most fragile case: a blank surname
  counts as disagreement, and few random labels cover such a record. With random labels it
  passed 100% of repeats in this build, and 60% to 100% by draw in the previous one; with active
  learning, every repeat in both. In
  earlier runs, made while a Wikidata name list (since removed) was on, random labels missed it
  in up to 90% of repeats and active learning (`custmatch sample --active`) fixed it every time;
  the list made no measurable difference elsewhere, so read this as a fragile case either way.
  Active learning lifted the background F1 to 99.44. An option scoring blanks as unknown made it
  worse (0%) - a blank value the labels never showed is no better understood as "unknown" - and
  was removed in 0.1.0.
- Building the file found two gaps, now fixed: accented names (Ándréw) missed their nickname
  roots, and the first build, with no mistyped birth dates among the ordinary records, taught
  the model that any birth-date difference means two people (day-and-month swaps then passed
  10%). Real files have such errors, and labels that include them teach the model to forgive them.

---

[← Labels](04-labels.md) · [Contents](README.md) · [Blocking and scale →](06-blocking-and-scale.md)
