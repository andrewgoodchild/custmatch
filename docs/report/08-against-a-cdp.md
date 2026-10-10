# 8. Matching for marketing: against a CDP

Customer data platforms (CDPs) match deterministically, on exact identifiers. For marketing,
a wrong merge sends one person's message to someone else - so this chapter measures both
coverage and wrong-person contacts, against a CDP tuned on the truth.

## Deterministic CDP matching vs custmatch

This first comparison used one generated population and F1; "Reassessed for marketing" below
repeats it on three populations with the measures a marketer cares about, and supersedes it.

Customer data platforms such as Twilio Segment resolve identity deterministically: two
profiles merge when they share an exact identifier - a user ID, email, phone or cookie - with
limits on how many values of each a profile may hold. Many sites make accounts optional, so
how much a CDP can link depends on how many visitors sign up. `scripts/cdp_comparison.py`
reimplements Segment's documented rules (user ID limit 1, others 5; on a limit, the incoming
event's lowest-priority identifier is dropped and the match retried; case-insensitive values;
its recommended blocked values) and runs them and custmatch on the same records and events: the
sparse Australian file plus generated web events, at two sign-up rates. Web and app records
carry the cookie of the device they were made on, as a CDP sees them (about half of all
records; some on a shared family laptop). High: 70% of people have
an account, and 46% of records carry a user ID. Low: 15% and 10%. Both have the same
family laptops and call-centre agents as "Cookies and device IDs". "Well configured" means
cleaned identifiers, one email and one phone per profile, and junk blocked - including every
value held by more than ten records, as an admin reviewing the commonest values would.
Mean of three runs:

| Method | High sign-up: F1 (precision / recall) | Low sign-up: F1 (precision / recall) | Among records with no email, phone or user ID, low sign-up: F1 | Largest profile |
| --- | --- | --- | --- | --- |
| Segment-style, defaults, identifiers as written | 18.4 (11 / 60) | 6.7 (4 / 50) | 0.0 | 245 |
| Segment-style, defaults, identifiers cleaned | 20.8 (12 / 70) | 8.1 (4 / 62) | 0.0 | 245 |
| Segment-style, well configured | 80.2 (92 / 71) | 74.6 (89 / 64) | 0.0 | 8 |
| custmatch | 89.5 (92 / 87) | 87.2 (92 / 83) | 82.0 | 27 |

- **Out of the box, deterministic matching collapses.** Placeholder emails that store staff type
  for customers without one - noemail@noemail.com, na@na.com, the store's own address - each
  joined over a hundred people into one profile; Segment's default blocked values do not include
  them. Most of a CDP's identity work is finding and blocking such values.
- **Well configured, it is precise where identifiers exist, and blind elsewhere.** It cannot
  link two records unless an identifier joins them, so a person who used a different email in
  each system, or a record with only a name, stays apart: it scored 0 on records with no email,
  phone or user ID, where custmatch scored about 82 - by construction: those
  records are all store records, which carry no cookie, and they hold only 29 true pairs, so
  the custmatch figure is noisy.
- **Fewer sign-ups hurt deterministic matching more:** it lost about 6 points from high to low sign-ups, custmatch about 2. With few accounts there are few user IDs to join on, and the
  fuzzy evidence - names, birth dates, near-identical contacts - carries more of the weight.
- **Family emails and phones remain a cost** for deterministic matching even when well
  configured: two people sharing one email are one profile. custmatch profiles such values as
  shared and weighs them accordingly.
- **The caveat:** a CDP with a logged-in user ID on nearly every event and record does better
  than either scenario here. The comparison is generous to deterministic matching in one way
  too: its tuning blocked junk with knowledge of this file.

### Reassessed for marketing: wrong-person contacts

F1 weighs a wrong merge the same as a missed one. In marketing a wrong merge is worse - it sends
one person's offers or details to someone else, often a former partner - and that complaint is
why many marketers prefer deterministic matching. `scripts/cdp_reassess.py` reruns the
comparison with that in view, on three independently generated populations
(`scripts/au_customer_data.py --seed`), one label draw each:

- **The CDP tuned on the truth,** over value limits of 1, 2 and 5 emails and phones, blocking of
  values held by more than 3, 5, 10 or 20 records, and one or five cookies per profile. Two
  settings are shown: the one with the best F1 (one email, one phone, values on more than 5
  records blocked) and the most precise (the same, values on more than 3 records blocked, one
  cookie per profile). No CDP administrator has the truth to tune against, so this is generous.
- **custmatch as a user would run it:** its default threshold, and `"min_precision": 0.97` -
  the setting, not a threshold picked afterwards - on the same 1,000 labels (300 random, 700
  active). And with worse labels: 5% of them flipped, and 300 random labels with no active
  learning.

"Contacted at someone else's email" is the share of records whose customer's chosen email - the
commonest real one on its records - was never used by that record's person.

| Data | Method | Precision (range over datasets) | Recall | Contacted at someone else's email | Wrong pairs: same household | Wrong pairs: strangers |
| --- | --- | --- | --- | --- | --- | --- |
| Many sign-ups | CDP-style, tuned for F1 | 92.3 (92-93) | 70.4 | 1.45% | 504 | 189 |
| Many sign-ups | CDP-style, tuned for precision | 95.7 (96-96) | 64.3 | 0.58% | 262 | 79 |
| Many sign-ups | custmatch, default threshold | 93.5 (91-96) | 86.2 | 0.86% | 237 | 487 |
| Many sign-ups | custmatch, `min_precision` 0.97 | 98.3 (98-99) | 58.1 | 0.05% | 8 | 114 |
| Many sign-ups | custmatch, 5% of labels wrong | 91.6 (83-97) | 81.2 | 0.93% | 62 | 849 |
| Many sign-ups | custmatch, 300 random labels only | 81.5 (72-89) | 80.5 | 2.93% | 607 | 1713 |
| Few sign-ups | CDP-style, tuned for F1 | 88.3 (88-89) | 63.1 | 2.06% | 707 | 286 |
| Few sign-ups | CDP-style, tuned for precision | 92.8 (92-93) | 56.3 | 0.94% | 406 | 115 |
| Few sign-ups | custmatch, default threshold | 91.3 (88-94) | 82.4 | 1.65% | 166 | 776 |
| Few sign-ups | custmatch, `min_precision` 0.97 | 97.8 (97-98) | 46.7 | 0.08% | 14 | 112 |
| Few sign-ups | custmatch, 5% of labels wrong | 82.5 (69-95) | 78.7 | 1.99% | 84 | 2069 |
| Few sign-ups | custmatch, 300 random labels only | 80.6 (73-88) | 73.5 | 2.50% | 487 | 1680 |

**No method wins on every count:**

- **custmatch at its default** finds 16 to 26 points more of each person's records than the CDP
  settings, but it is 1.5 to 2 points less precise than the precision-tuned CDP and sends more
  records to someone else's email (0.86% against 0.58% with many sign-ups, 1.65% against 0.94%
  with few). It makes about three to seven times as many wrong merges between strangers (similar
  names on sparse records), and far fewer within households. Its default is for finding
  customers, not for contacting them.
- **custmatch with `min_precision` 0.97** is the safest for contacting people: about 98%
  precision and a tenth or less of the wrong-person emails of the most precise CDP setting - but it
  finds about 6 to 10 points fewer matches than that setting. Safety is bought with missed matches, as with any
  method.
- **Couples are the CDP's weak point,** as practitioners report: most of its wrong merges are
  within a household (a family email, a shared laptop's cookie).
- **custmatch needs good labels.** With 300 random labels, or 5% of labels wrong, its precision
  swung from about 70 to the high 90s between datasets and between reruns of the same one -
  unreliable, and often worse than the CDP. (These rows are one label draw per population: a
  controlled rerun moved such a setup's precision by up to 15 points with nothing changed but the
  run, so read them as ranges, not values.)
  A deterministic CDP needs no labels. `summary.json` warns when the labels are random-only or
  hold too few matches.

**What this cannot show.** The populations are generated, and their error rates and shared
contacts are our assumptions; without family emails the CDP's precision rises to 93-96
(`scripts/au_customer_data.py --no-shared`, one dataset). One label draw per dataset. A CDP with
a logged-in user ID on nearly every record would do better than either scenario here. The
academic record-linkage literature agrees in direction: probabilistic linkage gives a better
trade-off between missed and false matches when identifiers have errors, and deterministic
linkage fewer false matches at the cost of many missed ones (Tromp et al., Journal of Clinical
Epidemiology 2011). Much of "probabilistic" matching's reputation in marketing comes from device
graphs that link people by IP address and behaviour - a different technique, measured at 13 to
16% accuracy for IP-to-email links - not from matching on names and contact details.

### Couples who separate

The commonest complaint: a couple shared a home and perhaps a family email; they separate; one
moves out, and the new records are matched to the old household, so offers or personal details
reach the ex. `scripts/separation_research.py` builds that case on the many-sign-ups CDP data of
three generated populations: in each, half the couples (two-person households, and adults
sharing a family email) separate - 3,420 in all. The one who leaves has half their records made
after the split, with a new address and postcode, made on their own new device, and - habits
outlast relationships - the family email kept on half of those that had it. Every record carries
a date, the post-split ones newest; after the split the family email and the old home are the
ex's. Over the leavers' 3,265 post-split records, mean of the three populations, one label draw each:

| Method | Merged with the ex | Ex's email, brought in by matching | Ex's email, on the record itself | Post to the ex's address | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| CDP-style, tuned, commonest details | 3.77% | 1.81% | 1.78% | 1.75% | 92.4 | 70.2 |
| CDP-style, tuned, newest details (last write wins) | 3.77% | 1.38% | 1.78% | 1.50% | 92.4 | 70.2 |
| custmatch, default threshold | 0.37% | 1.30% | 1.78% | 1.32% | 88.4 | 86.3 |
| custmatch, contact safety | 0.37% | 0.58% | 1.69% | 1.32% | 88.4 | 86.3 |
| custmatch, contact safety + newest details | 0.37% | 0.59% | 1.78% | 1.29% | 88.4 | 86.3 |
| custmatch, contact safety + `min_precision` 0.97 | 0.06% | 0.58% | 1.69% | 0.83% | 98.4 | 66.0 |
| custmatch, contact safety + newest + `min_precision` 0.97 | 0.06% | 0.53% | 1.75% | 0.83% | 98.4 | 66.0 |

- **Matching the leaver to the ex** is where the methods differ: the CDP did it to 3.4, 4.4, 3.5%
  of post-split records in the three populations - a family email, or a household device used
  before the split - and custmatch to 0.8, 0.3, 0.0%; with `min_precision` almost never.
- **Most of the remaining risk is not a matching error.** It is the leaver's own details going
  stale: their earlier records - correctly matched to them - carry the family email and the old
  address, and some post-split records still carry the family email themselves. Contact safety
  (a personal email preferred over a shared one) cut the ex's email brought in by matching from
  1.3% to 0.6%; choosing the newest details (dates and `"most_recent"` survivorship, or a CDP's
  last-write-wins) added little on top. A leaver whose only address on file is the shared one will be written
  to there by any system until they give a new one.
- **The family email on a record itself** (1.8%) can only be flagged, not fixed: `contacts.csv` marks
  a household-shared email or phone (`email_shared`), so a campaign can skip or confirm it.
- **Not like-for-like:** custmatch's contact-safety rows pick a contact per record
  (`contacts.csv`); a CDP has one email per profile. And about ten post-split records make a
  percentage point in each population, so differences under half a point are within noise.

---

[← Golden records](07-golden-records.md) · [Contents](README.md) · [Against other tools →](09-against-other-tools.md)
