# Reproducing the research

How to rebuild the datasets, rerun every experiment and run the tests. The tool itself needs
only `pip install -e .` - see [guide.md](guide.md).

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[test]'
# the research scripts, not custmatch, need these: Splink (the comparisons), torch and
# transformers (Ditto), CatBoost and TabPFN (the classifier bake-off), Kuzu (the graph study), Modal,
# pynndescent (approximate neighbours for embedding blocking)
uv pip install --python .venv/bin/python "splink==4.0.17" torch transformers catboost tabpfn kuzu modal \
    "jellyfish<1.0" recordlinkage openai tqdm tensorboardX spacy mlx-lm pypdf openpyxl pseudopeople pynndescent
.venv/bin/python -m spacy download en_core_web_lg     # Ditto's --dk general
.venv/bin/python -c "import nltk; nltk.download('stopwords')"

# pinned: the patch applies to this commit (MatchGPT has no licence file; nothing of it is redistributed)
git clone https://github.com/megagonlabs/ditto.git external/ditto
git -C external/ditto checkout 52985564a93fb11308439516d3e17a033d43ec8f
git -C external/ditto apply ../../patches/ditto-mps-and-fixes.patch
git clone https://github.com/wbsg-uni-mannheim/MatchGPT.git external/MatchGPT
git -C external/MatchGPT checkout ba1691935c744bb0eacc9abf772f39271e318e4b
echo 'OPENROUTER_API_KEY=sk-or-...' > .env    # only for the GPT rows; .env is gitignored

# name frequencies for the Australian test file: US Census 2000 surnames and Social Security baby
# names 1880-2008 (US government data, public domain), via the FiveThirtyEight data repository
# (CC BY 4.0) and hadley/data-baby-names; used to generate data/, not redistributed
mkdir -p data/names
curl -L -o data/names/census2000_surnames.csv https://raw.githubusercontent.com/fivethirtyeight/data/master/most-common-name/surnames.csv
curl -L -o data/names/ssa_baby_names.csv https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv
```

Run one job at a time. Throughput here is bound by memory, not the GPU: with Docker Desktop
holding 10 GB of 16 GB, a Ditto epoch swung between 13 seconds and 10 minutes depending only on
what else was resident.

## Scripts

Grouped by the report chapter whose numbers they produce.

### Data (report chapter 1)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/customer_data.py` | Builds the two-source customer dataset from pseudopeople with labelled contact pathologies | ~3 min |
| `research/scripts/au_customer_data.py` | Builds the sparse Australian customer file: web / app / store channels, partial names, mobiles and emails in many formats, nicknames, anglicised names, junk and fictional numbers, known people and households; `--scripts` writes a copy with some names in Chinese or Arabic script | ~1 min |
| `research/scripts/person_em_build.py` | Builds the person-record EM dataset: cluster-disjoint splits, Ditto format + RF features on identical pairs | ~2 min |
| `research/scripts/make_demo.py` | Builds `examples/demo/` from the Australian file: 500 households, emails moved to the reserved `.example` domain, 600 labels (300 random, 300 by active learning) | ~2 min |
| `research/scripts/make_person_llm_task.py` | Builds a hard person-matching task in MatchGPT's format | seconds |
| `research/scripts/customer_person_llm_task.py` | Hard person-matching task from the synthetic customer file (no model can know the people), plus a random forest on the same pairs | ~1 min |

### Whole customers, not pairs (chapter 2)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/person_ditto.py` | Trains Ditto/RoBERTa on person records and scores the full candidate graph | ~30 min train, ~100 min inference |
| `research/scripts/person_head_to_head.py` | RF vs Ditto, pairwise and after the clustering pass, over a threshold sweep | ~3 min |
| `research/scripts/cascade.py` | RF → Ditto cascade over the saved scores: routing budget vs clustered F1, split-half tuning, bridge-aware vs plain on one grid | ~1.5 min |
| `research/scripts/kuzu_graph.py` | Loads the match graph into Kuzu with edge provenance; clustering as a graph query; bridge-aware clustering | ~4 min |
| `research/scripts/threshold_capture.py` | Saves what custmatch sees when it chooses its threshold, with the truth, so rules can be compared offline | ~1 h |
| `research/scripts/threshold_rules.py` | Compares threshold rules on those captures (the middle-of-ties rule) | seconds |

### Which matcher (chapter 3)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/febrl_person_matching.py` | FEBRL4: rules, recordlinkage ECM, Splink, supervised RF | ~5 min |
| `research/scripts/historical50k_dedupe.py` | historical_50k dedupe: rules, Splink, supervised RF, with the blocking ceiling reported | ~15 min |
| `research/scripts/person_task_baselines.py` | Scores those same pairs with a rule and an RF, for a like-for-like LLM comparison | ~2 min |
| `research/scripts/matcher_bakeoff.py` | The forest on all six datasets, same label files for every model, three draws; `--tag NAME --override JSON` runs a config variant (the other models it once compared - hgb, CatBoost, TabPFN, `auto` - were removed in 0.1.0; `research/scripts/modal_models.py` reruns them) | ~1 h |
| `research/scripts/modal_models.py` | On Modal, in parallel: the forest against hgb, ExtraTrees, LightGBM, XGBoost, CatBoost (CPU) and TabPFN v2, TabICL (L4 GPU) on all six datasets, same labels for every model, three draws; `--models forest,forest_match_weight --out research/results/match_weight.json` compares the match weight | ~1 h, ~$5-8 |

### Labels (chapter 4)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/label_curve.py` | custmatch trained on 50 to 1,000 labels, random or active, three seeds, on six datasets; and with no labels, the match weight at label-free cut-offs | ~30 min |
| `research/scripts/jev_labels.py` | Forest trained on Jev's labels instead of the truth, on both person datasets: hard, soft, and cut-off from 300 hand labels | ~10 min, ~$0.70 |
| `research/scripts/historical_labels.py` | historical_50k: the forest with 1,000 / 3,000 / 6,000 random labels, chosen and best-in-hindsight thresholds | ~5 min |
| `research/scripts/local_llm_labels.py` | Qwen3 8B or 14B through a local Ollama (`--model=`) answering 500 random and 500 hard training pairs per person dataset, against the forest out of fold; needs Ollama and the model pulled, nothing leaves the machine | ~2 h per model |

### Customer data (chapter 5)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/shared_value_detection.py` | Detects household-shared and junk/placeholder values without labels | ~2 min |
| `research/scripts/customer_pipeline.py` | **The end-to-end customer-matching process**: contact profiling → blocking → matching → clustering → households (`--both` runs the naive ablation) | ~5 min |
| `research/scripts/customer_ditto.py` | Ditto on the customer pipeline's own pairs and split: `build`, then `person_ditto.py --data-dir data/customer_em --out customer --max-len 160`, then `evaluate` and `golden` | ~45 min training, ~2 min scoring |
| `research/scripts/household_research.py` | Household and shared-contact ideas from the literature (customer-level profiling, fuzzy addresses, contact + surname merge) against true households | ~1 min |
| `research/scripts/placeholder_research.py` | Fake-name, fake-address and fake-date-of-birth detection: false positives on four clean datasets, precision / recall on injected placeholders, matching with and without | ~3 min |
| `research/scripts/unicode_form.py` | What reading text in one Unicode form changes: values changed per benchmark dataset, and custmatch before and after on historical_50k and the synthetic file | ~15 min |
| `research/scripts/audit_check.py` | `custmatch audit` against the truth: one run on the synthetic file, 100 audit samples of 300 pairs, how often each 95% interval holds the true precision, recall and F1 | ~10 min |
| `research/scripts/au_validate.py` | custmatch on that file, each normalisation step and labelling strategy over three label draws (`country_au_active` is the recommended setup); `--data au_customer_script` runs the native-script copy; `ceilings` the blocking and household limits | ~3 min per variant |
| `research/scripts/scenarios.py` | `build` writes `examples/scenarios/` (named hard cases with known answers); `score [variant ...]` trains custmatch and scores each scenario, three draws | ~5 min per variant |
| `research/scripts/cookie_research.py` | Cookie and device-ID stitching on the Australian file: the gain, and what the shared-device guards protect | ~10 min |
| `research/scripts/record_dates_research.py` | Record dates: moves, name changes and reissued mobiles over time, with and without `"record_dates"` | ~10 min |
| `research/scripts/suspect_research.py` | Which profanity-list words are real names (NCVR), and the suspect flag's false flags on held-out files; time at a million records | ~5 min |

### Blocking and scale (chapter 6)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/historical_blocking.py` | historical_50k: the blocking ceiling with exact keys, similarity blocking and the GB profile; `--run V1,V2` also runs custmatch (1,000 random, and 300 + 700 active; three draws) | ~1 min, ~6 min per variant run |
| `research/scripts/custmatch_scale.py` | Times each stage on disjoint copies of the synthetic file, up to 1M records; optional blocking mode (`fixed`, `learned`) | ~5 min |
| `research/scripts/modal_scale.py` | On Modal (16 cores, 128 GiB): the scale test at 2, 5 and 10 million records and the full 5-million-record NCVR file, with peak memory | ~1-2 h |
| `research/scripts/modal_ncvr_full.py` | On Modal: all 5 million NCVR records, labelled as a user would (300 random + 700 active), resumable; `--variants`; `python ... --local VARIANT` runs it on this machine (the adaptive variant: 2.2 h, about 51 GB, on an 18-core 64 GB laptop) | ~3 h |
| `research/scripts/embedding_blocking.py` | historical_50k and NCVR (20,000 people): the blocking ceiling with each record's 5, 10 or 20 nearest neighbours by character n-gram vectors (`--neural` for a sentence model, a ~90 MB download), against the keys and LSH | ~10 min |
| `research/scripts/embedding_pipeline.py` | The same two datasets through the whole pipeline with keys, keys + LSH and keys + 5 or 10 neighbours, exact or approximate; 100 random + 900 active labels, ten seeds, with stage times; `--timing` times each blocking alone | ~70 min for ten seeds |
| `research/scripts/embedding_scale.py` | The Leipzig NCVR file from 100,000 to 5 million records: vectors and approximate neighbours (pynndescent), against keys and LSH blocking up to a million, with peak memory | ~40 min |
| `research/scripts/memory_profile.py` | Peak memory per pipeline stage on an NCVR sample | ~5 min |
| `research/scripts/large_block_check.py` | The large-block check on and off: six person benchmarks and dedupe's patent applicants | ~10 min |

### Golden records (chapter 7)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/golden_record.py` | Golden-record test set: scores survivorship rules against noise-free truth, with predicted vs oracle clusters | ~3 min |

### Against a CDP (chapter 8)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/cdp_comparison.py` | Segment-style deterministic resolution against custmatch on the generated customer populations | ~40 min |
| `research/scripts/cdp_reassess.py` | The CDP comparison as a marketer would judge it: precision-weighted scores, wrong-person contacts, `"min_precision"`, three populations | ~1 h |
| `research/scripts/separation_research.py` | Couples who separate: how often the leaver's new records are merged with the ex | ~10 min |
| `research/scripts/order_dependence.py` | Records arriving in batches: how often they swap partners, with and without hysteresis | ~20 min |

### Against other tools (chapter 9)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/splink_vs_custmatch.py` | Splink vs custmatch on the synthetic file, scored identically; also Splink-seeded labels | ~1 min |
| `research/scripts/splink_datasets_benchmark.py` | custmatch vs Splink on Splink's bundled datasets, Splink running its official example settings | ~15 min |
| `research/scripts/beat_splink.py` | Each custmatch option (rich features, the since-removed Splink feature, active mix, Jev labels) on all five datasets against Splink; `--seed N` for another label draw | ~15 min per variant |
| `research/scripts/address_book_data.py` | Synthetic personal address books (one record per email address) with relay, switchboard, household, same-name and brand traps; `--seed N`; writes `data/address_book/sN/` | seconds |
| `research/scripts/address_book_benchmark.py` | Four labelling budgets on dirty and clean books, false links by trap; `--draws 10` compares plain and stratified 300 random labels over ten draws | ~5 min; ~10 min with `--draws 10` |
| `research/scripts/head_to_head.py` | custmatch against Splink 5 (and, with `--dedupe`, dedupe) on the synthetic file copied to a million records: time, peak memory, clustered F1 (`.venv-splink5`, `.venv-dedupe`) | ~30 min |
| `research/scripts/home_data_benchmark.py` | Each tool on every tool's own datasets: Splink's four and dedupe's two (`data/dedupe/`) | ~40 min per dedupe pass |
| `research/scripts/ncvr_leipzig.py` | custmatch and Splink on the Leipzig NCVR benchmark, by sample size and blocking keys, three draws; `--random N`, `--stratified` (random labels drawn by agreement pattern); the `current_default` variant sets nothing | ~15 min |
| `research/scripts/ncvr_blocking_scale.py` | Blocking alone on NCVR at 200k, 1M and 3.5M people: pair completeness, the most F1 it allows, and candidate pairs for the default, `fixed` and `adaptive` blockers; `--splink-datasets` checks Splink's four datasets | ~30 min |
| `research/scripts/bpid_benchmark.py` | BPID (AWS; download from Zenodo into `data/bpid/`): pairwise F1 on its 10,000 labelled pairs at 300, 1,000 and 8,000 labels, and blocking recall over its million profiles; `--variant`, `--tag` | ~3 min per variant |

### Reproductions of published results (chapter 10)

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/magellan_baseline.py` | Reproduces the Magellan column of Mudgal et al. (SIGMOD'18) — per-attribute similarity features, best of DT/RF/LR/NB/SVM on validation | ~8 min, CPU |
| `research/scripts/run_ditto_all.sh`, `research/scripts/run_ditto_remaining.sh` | Ditto fine-tuning sequences | 10 min–3 h per dataset |
| `research/scripts/llm_zeroshot_mlx.py` | Zero-shot LLM matching with MatchGPT's exact prompts and parsing, local model via MLX | ~16 min per test set |
| `research/scripts/llm_zeroshot_openrouter.py` | The same protocol against hosted models via OpenRouter, with a spend ledger, a hard `--budget` stop, and the answering model logged per pair (for routers). Use `--no-temperature --max-tokens 2000`+ for reasoning models | ~2–6 min per test set, $0.01–0.65 each |
| `research/scripts/run_llm_all.sh` | All four LLM test sets | ~1 h |
| `research/scripts/jev_decisions.py` | Jev decision model as the matcher, via OpenRouter's Decisions API: P(match) per pair, same budget ledger | ~45 s per test set, ~$0.02 each |
| `research/scripts/jev_variants.py` | Jev's API options one at a time on the two person tasks: criteria, structured fields, three-way choice, several questions per call | ~5 min, $0.12 |
| `research/scripts/jev_bootstrap.py` | Paired bootstrap intervals for the Jev variant comparisons (no API calls) | seconds |
| `research/scripts/jev_cascade.py` | Forest veto → Jev on the uncertain band of the full 152k-pair test split, clustered and scored against Ditto with split-half tuning (`--pilot N` first) | ~17 min, ~$1.60 |

### The tool and the service

| Script | What it does | Runtime |
| --- | --- | --- |
| `research/scripts/custmatch_validate.py` | Runs the `custmatch` tool end to end on the synthetic file at 500 / 1,000 / 3,000 labels; `--active` and `--active-seeds` (active learning), `--jev` (~$0.06) | ~1 to 40 min |
| `research/scripts/custmatch_incremental_validate.py` | `custmatch add` on 10% held-out records vs a full rerun | ~20 s |
| `research/scripts/robustness.py` | Feeds the CLI broken inputs (missing files, misspelt columns, empty sources, other encodings, bad labels, one giant block) and checks each ends in a clear message, never a crash or a silent run | ~3 min |
| `research/scripts/modal_rerun.py` | On Modal, one container per experiment command, results merged back into `research/results/`: the published experiments that set a country profile (CDP, couples, cookies, record dates, scenarios, order dependence, NCVR) | ~2 h on the longest job, ~$5 |
| `research/scripts/modal_name_list_ab.py` | On Modal: the code before the Wikidata name list was removed, with and without the list, on the CDP populations and the Australian file, three draws | ~30 min, ~$2 |
| `server/replay_test.py` | Real-time matching one record at a time against batch runs; `--all` writes every merge policy and the coherence check over five seeds | ~25 min with `--all` |
| `server/learning_test.py` | Steward decisions as labels: three review rounds, decisions training the model or only overriding | ~10 min |
| `server/scale_test.py` | The service at scale: loads, the batch, the real-time matcher's start and lookups | ~10 min at a million |
| `research/scripts/collect_results.py` | Gathers everything into `research/results/summary.json` | seconds |

Results land in `research/results/` as JSON plus the raw training logs.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # ~50s
```

They cover the mistakes this project actually made, not just happy paths:

| File | Guards |
|---|---|
| `test_metrics.py` | Scoring arithmetic, recall counted against *all* true pairs, and that a collapsed run scores exactly 2p/(1+p) |
| `test_contacts.py` | Junk vs household vs personal classification, including the address-diversity clause and the first-name rule |
| `test_clustering.py` | Isolated records survive as singletons; bridge-aware cuts weak links only; households never chain across addresses |
| `test_data_integrity.py` | Splits are cluster-disjoint, labels match ground truth, injected junk really is shared |
| `test_custmatch.py` | The tool end to end on generated data: config errors, the threshold chosen after clustering, the LLM labelling gate, `custmatch add`, blocking modes, the match weight (fitted without labels, reloaded by `add` and `explain`), households and placeholders |
| `test_reported_results.py` | **Every headline number in the docs still matches `research/results/*.json`**, and no section reference points at a section that does not exist |

The last one exists because a claim about a section that had never been written survived in
this repo until it was checked. Data-dependent tests skip cleanly when `data/` has not been
built.

## Datasets

| Dataset | Type | Size | Source |
|---|---|---|---|
| DeepMatcher/Magellan suite | product, bibliographic | 450–28,707 pairs | `external/ditto/data/er_magellan` |
| MatchGPT test sets + WDC Products 80% cc | product, bibliographic | ~1,250 pairs each | `external/MatchGPT/LLMForEM/tasks` |
| FEBRL 3 / 4 | synthetic person records | 5,000 + 5,000 records | `recordlinkage` |
| historical_50k | real people with injected errors | 50,578 records, 5,156 people | `splink` |
| person_em (derived) | person-record EM splits, cluster-disjoint | 12k train / 3k valid / 152,772 test pairs | `research/scripts/person_em_build.py` |

**FEBRL is saturated** — rules score 97.7 and three matchers all score ≈99.9. Use it as a
smoke test, not a benchmark. historical_50k separates methods properly (33 → 61 → 80).

NCVR (Leipzig benchmark, 5 × 1M records, CC BY 4.0): download
`https://dbs.uni-leipzig.de/files/datasets/saeedi/5Party-ocp20.tar.gz` into `data/ncvr_leipzig/`,
unpack it there, and run `research/scripts/ncvr_leipzig.py --people 20000,200000` (about 15 minutes; `--splink` adds Splink, which needs about 7 GB at 200,000 people). It holds
real names: keep it in the gitignored `data/`. The raw North Carolina snapshots, linkable by NCID,
remain the realistic version ([report](report/README.md) §13).

`research/scripts/cdp_comparison.py` (about 40 minutes) compares Segment-style deterministic resolution with custmatch at high and low sign-up rates on the Australian file; `research/scripts/scenarios.py build` then `score` regenerates and scores the scenario file; `research/scripts/cookie_research.py` and `research/scripts/record_dates_research.py` measure cookie stitching and record dates.

Surveyed but not run: pseudopeople, OpenSanctions Pairs (755k labelled pairs, CC BY-NC),
WDC Products full benchmark, Machamp, ICE-ID. [report](report/README.md) §1 has the access details.

## Reproduction traps

Found the hard way; the first three cost real time.

1. **Ditto's `--dk` cache is not validated.** It caches to `<file>.dk` and reuses anything
   non-empty, including a half-written file from an interrupted run. A truncated cache gave a
   "training set" of 203 of 6,144 pairs and a plausible-looking 28.5 F1. Patched here to check
   line counts; if you kill a run during preprocessing, delete the `.dk` files.
2. **Check every run against 2p/(1+p).** If F1 sits on that value the run collapsed and the
   number is meaningless, not merely poor.
3. **`--da` (MixDA) is suspect.** Same data, same seed, Walmart-Amazon: 17.2 with, 84.8
   without. Always run a no-`--da` control. Its `drop_col` operator can also delete the `[SEP]`
   marker and crash `dataset.py:80` — patched here to fall back to the un-augmented pair.
4. **PyTorch's MPS allocator hoards.** It grew to ~11 GB of driver memory with ~2 GB of live
   tensors. `torch.mps.empty_cache()` per epoch keeps it bounded.
5. **Ditto pads with token id 0 (`<s>` for RoBERTa) and passes no attention mask**, so anything
   that adds padding hurts accuracy — bucketing padded lengths cost ~30 points of validation F1
   on Beer.
6. **F1 is not portable across test sets.** The same 8B model scores 12–17 points "better" than
   its published numbers purely because these test sets have a denser positive rate.

`research/patches/ditto-mps-and-fixes.patch` (applied in Setup) holds every change made to the upstream code: MPS device, apex
removed, allocator cache, the `[SEP]` fallback, the `.dk` cache check, and an optional
`DITTO_WARMUP_FRAC`.

## Not done

- Clustering: connected components, a bridge-aware variant, and re-cutting customers over
  `max_cluster`. Correlation clustering was measured and removed (report, "Removed in 0.1.0").
- GPT-4 (`gpt-4-0613`) not rerun: ~$21 for the five product sets, and it retires 2026-10-23.
- Public and generated data only. The generated customer files (US and Australian) do have
  nicknames, households, shared phones and moves, but their errors were written by the same
  author as the fixes that handle them; no real customer file has been tested.
