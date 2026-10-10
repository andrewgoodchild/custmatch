# 1. Data, and the work it builds on

Which datasets to test customer matching on, and the published work people compare against.
Person and customer records differ from the product and citation pairs most papers report on:
names drift, contact values are shared within families or faked, and the output is a set of
customers, not a set of pairs.

## The test data that matters

### Person / customer records (closest to "customer matching")

| Dataset | What it is | Size | Access |
|---|---|---|---|
| **FEBRL 3 / 4** | Synthetic Australian person records: name, address, DOB, SSN-like id, with generated typos | 5,000 + 5,000 records, 5,000 true links (F4); 5,000 records / 6,538 links (F3) | `recordlinkage` package (BSD-3) |
| **historical_50k** | Real people from Wikidata with injected errors; name, DOB, birth place, postcode, occupation | 50,578 records, 5,156 people, 303,961 true pairs | `splink` package (MIT) |
| **NCVR (Leipzig, Saeedi et al.)** | Real North Carolina voter records + GeCo-corrupted duplicates, multi-source | 5 sources × 1M records, 3,331,384 matches | `dbs.uni-leipzig.de/files/datasets/saeedi/5Party-ocp20.tar.gz` (CC BY 4.0) |
| **NCVR raw snapshots** | Real voter file; NCID gives ground truth across snapshots | 0.6–1.0 GB per zipped snapshot, 2005–2026 | `dl.ncsbe.gov/index.html?prefix=data/Snapshots/` (public record, real PII) |
| **pseudopeople** | Synthetic US census / tax / SSA records with noise and ground-truth ids | 10k sample bundled; Rhode Island (~1M) and US (~330M) on request | `pip install pseudopeople` (BSD-3) |
| **OpenSanctions Pairs** | Analyst-labelled person / company / vessel pairs, multilingual, cross-script | 755,540 labelled pairs | `data.opensanctions.org/contrib/training/` (**CC BY-NC**) |
| **Splink fake_1000** | Tiny synthetic people set with clerical labels | 1,000 rows, 250 entities | `splink` package |

Generators when you need your own: **GeCo**, **Febrl dsgen**, **Gecko** (MIT, SoftwareX 2024).
Faker/Mimesis produce clean fakes only — they need a separate corruptor to be useful here.

### The standard entity-matching benchmarks (what papers report on)

The **DeepMatcher/Magellan suite** is the lingua franca: Abt-Buy, Amazon-Google,
Walmart-Amazon, DBLP-ACM, DBLP-Scholar, Beer, iTunes-Amazon, Fodors-Zagats, Company, in
Structured / Dirty / Textual variants, split 3:1:1. Ready-to-use serialisations live in the
Ditto repo (`data/er_magellan/`). Sizes range from 450 labelled pairs (Beer) to 28,707
(DBLP-Scholar) and 112,632 (Company).

**WDC Products** (EDBT 2024) is the current product benchmark with controlled difficulty
(20/50/80% corner cases, seen/unseen products). **MatchGPT** publishes ~1,250-pair downsampled
test sets of the DeepMatcher datasets plus WDC 80% corner-case — these are what the LLM papers
score on.

Also surveyed: WDC LSPC v2, Machamp (CIKM 2021), Alaska (its site is dead; use the SIGMOD
contest mirrors), billiger.de Products (German offers, 2026), ICE-ID (Icelandic censuses,
2025/26), CorpFam (corporate-family matching, 2026).

## The papers people compare against

| Paper | Venue | Contribution | Code |
|---|---|---|---|
| Mudgal et al., **DeepMatcher** | SIGMOD 2018 | Defined the benchmark suite; "Magellan" = classical features + best classical classifier | `anhaidgroup/deepmatcher` |
| Li et al., **Ditto** | PVLDB 2021 | Fine-tuned RoBERTa on serialised pairs + augmentation, domain knowledge, summarisation. Still the supervised reference | `megagonlabs/ditto` |
| Tu et al., **Unicorn** | SIGMOD 2023 | One multi-task DeBERTa + mixture-of-experts across matching tasks | `ruc-datalab/Unicorn` |
| Peeters, Steiner & Bizer, **Entity Matching using LLMs** | EDBT 2025 | Zero-/few-shot LLM matching; prompt sensitivity study; GPT-4 beats Ditto on several sets | `wbsg-uni-mannheim/MatchGPT` |
| Steiner, Peeters & Bizer, **TailorMatch** | ICDE-W 2025 | Fine-tuning open LLMs for matching; reports Llama-3.1-8B/70B zero-shot baselines | arXiv 2409.08185 |
| Wang et al., **ComEM** ("Match, Compare, or Select?") | COLING 2025 | Selecting among candidates beats pairwise matching for LLMs | `tshu-w/ComEM` |
| Fan et al., **BatchER** | ICDE 2024 | Batched prompting cuts LLM cost 4–5× | arXiv 2312.03987 |
| Zhang et al., **Jellyfish** | EMNLP 2024 | Small open models instruction-tuned for data preprocessing/matching | HF `NECOUDBFM/Jellyfish` |
| Zhang et al., **AnyMatch** | AAAI-W 2025 | Zero-shot matching from a fine-tuned GPT-2 (124M) | `Jantory/anymatch` |
| Smith et al., **OpenSanctions Pairs** | arXiv 2603.11051 (2026) | 755k labelled sanctions/PEP pairs; rule-based 91.3 F1 vs GPT-4o 99.0 | `chansmi/OSINT_entity_resolution` |
| Steiner & Bizer, **Labeling training data with LLMs** | arXiv 2606.28823 (2026) | LLM-labelled training data ≈ human-labelled for training a matcher | `wbsg-uni-mannheim/Automatic-data-labeling` |

Splink (MoJ) is the practitioner standard for person data: Fellegi–Sunter with unsupervised
EM. Its own paper reports no accuracy numbers, which is exactly why [chapter 3](03-which-matcher.md) exists.

---

[Contents](README.md) · [Match whole customers, not pairs →](02-whole-customers.md)
