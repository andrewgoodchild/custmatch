# Research

The experiments behind custmatch. None of this is needed to use the library or the service; it
is here so every number in the docs can be checked and rerun.

| Folder | Holds |
| --- | --- |
| `scripts/` | The experiments: benchmarks, ablations, scale runs, comparisons with other tools |
| `results/` | What they wrote, as JSON; the test suite checks every number quoted in the docs against these |
| `poc/` | Proofs of concept for the service: storage, real-time matching, memory |
| `patches/` | Our changes to the upstream research code the experiments clone (Ditto and others) |

What was found is in the [report](../docs/report/README.md), one chapter per question. How to
rerun each experiment, grouped by chapter, is in [reproducing.md](../docs/reproducing.md). Run the
scripts from the repository root, e.g. `python research/scripts/head_to_head.py`; datasets go in
the gitignored `data/`.
