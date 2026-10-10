# 11. What the tool adopted, and what it dropped

The findings the tool adopted, and the features it built and then removed when measurement
on its own pipeline did not support them.

## From this research to custmatch

custmatch, the tool in this repository, was built from these findings. Some it adopted, some it
built and later removed when measurement on its own pipeline did not support them:

| Finding | In custmatch |
|---|---|
| Tune the threshold on clusters, and value precision first ([chapter 2](02-whole-customers.md)) | **Adopted** - the core of the tool: the threshold is chosen by closure-aware F1, cross-fitted over folds, with an in-cluster precision check (guide, "How it decides") |
| Shared and fake contacts are separable without labels ([chapter 5](05-customer-data.md)) | **Adopted and improved** - contacts are profiled before matching, and household contacts are classified again over *customers* after matching, which raised household-contact precision from 57.9% to 84.5% on the same file |
| Households as a separate layer, by address, not by closure ([chapter 5](05-customer-data.md)) | **Adopted**, with fuzzy streets and merges on a shared household contact plus surname ([chapter 5](05-customer-data.md), "Households, fake contacts and placeholders") |
| A transformer, behind a forest veto (chapters [2](02-whole-customers.md), [5](05-customer-data.md)) | **Built and removed** - the tool's `--ditto` stage declined on customer-style records and needed 12,000 labels to add under a point on historical_50k ([chapter 11](11-adopted-and-dropped.md), "Removed in 0.1.0") |
| LLM labels instead of hand labels, where they are ~99% right ([chapter 4](04-labels.md)) | **Built and removed** - `custmatch label` labelled pairs only after the model got at least 98% of your checked pairs right; tried on synthetic data, it added little over active learning, and sending customer records to a third party is a security risk |
| A zero-shot LLM as the matcher (chapters [10](10-reproductions.md), [2](02-whole-customers.md)) | **Not used** - best on hard pairs, worst in the clustering path |
| Structure-aware clustering ([chapter 2](02-whole-customers.md)) | **Not adopted** - worth about 0.1 on a well-tuned matcher; correlation clustering was tried in the tool and removed ([chapter 11](11-adopted-and-dropped.md), "Removed in 0.1.0") |
| Blocking sets the ceiling ([chapter 3](03-which-matcher.md)) | **Measured everywhere** - blocking diagnostics before labelling, ceilings beside every benchmark, similarity blocking and split blocks as options; at millions of records blocking is again the main limit ([chapter 6](06-blocking-and-scale.md), "Larger files") |
| Survivorship is half the job, and bounded by the source data ([chapter 7](07-golden-records.md)) | **Adopted** - golden records with per-field rules, source priority, newest value with record dates, and the address surviving as one unit |
| With no labels, Fellegi-Sunter ([chapter 3](03-which-matcher.md)) | Splink is the baseline every result is compared against; custmatch has its own Fellegi-Sunter weight as an optional input to the forest ([chapter 6](06-blocking-and-scale.md), "Blocking, and what Splink, dedupe and Zingg taught") |

## Removed in 0.1.0

Five options were removed because measurement showed they did not earn their place. Configs that
still name them get a warning in `summary.json`; the research behind them stays in the report.

- **The Ditto transformer stage** (`--ditto`, `"ditto_model"`): a fine-tuned RoBERTa (or
  ModernBERT) model judging the pairs the forest could not rule out. On customer-style records
  it declined every time (forest 98.64 against Ditto 97.65 with 1,000 labels). It helped only on
  the dense historical_50k with 12,000 labels (89.43 against the forest's 88.59; 90.02 with
  ModernBERT), at 90 minutes or more and a large install. [Chapter 2](02-whole-customers.md) has
  the research.
- **Correlation clustering** (`"clustering": "correlation"`): re-checking every pair inside each
  customer and splitting where that raised agreement. It helped a little on the clean synthetic
  file and hurt badly on noisy ones (fake_1000 93.36 to 86.20, historical_50k 86.29 to 78.97):
  asked about pairs blocking never proposed, the matcher called many of them two people. The
  in-cluster precision check and `max_cluster` are the safer defences against chaining.
- **The ambiguity rule** (`"ambiguous_margin"`): holding a record between two people the model
  judged different. It cost 1 to 6 F1 points and added no stability ("Stable matches as records
  arrive"); the twin guard covers the case that mattered.
- **`"missing_as_unknown"`**: scoring a blank name or street as unknown rather than as
  agreement or disagreement. With no labelled example of a blank the model could not use it: it
  made the first-name-and-email scenario fail every time and was roughly neutral elsewhere.
- **The Wikidata cross-language name list**: 97,606 given-name equivalences (Giuseppe - Joseph).
  No measured gain, because most name variants in customer data are anglicisations (Wei ->
  Kevin) that no list holds. A controlled check - the same code with and without the list,
  three label draws (`research/scripts/modal_name_list_ab.py`, `research/results/name_list_ab.json`) - found 80.82
  against 80.85 on the Australian file and -0.05 F1 on average over eight CDP populations.

---

[← Reproducing published results](10-reproductions.md) · [Contents](README.md) · [The service →](12-the-service.md)
