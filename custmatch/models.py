"""The pairwise classifier. "matcher_model" in the config picks it; the random forest is the
default and the one every result in the docs used unless it says otherwise.

  forest            scikit-learn random forest, 300 trees, balanced class weights
  forest_catboost   opt-in (pip install custmatch[catboost]): the forest's and CatBoost's P(match)
                    averaged. +0.41 clustered F1 on average over six datasets and two label kinds,
                    three draws each, never more than 0.03 behind on a dataset's mean; +1.3 on the
                    dense historical_50k and +1.8 on a sparse file with random labels; about 30%
                    longer runs on files of up to 50,000 records (research/scripts/modal_models.py,
                    research/results/ensemble.json; docs/report/03-which-matcher.md "Which classifier")

hgb (histogram gradient boosting), CatBoost, TabPFN and "auto" (pick the best cross-fitted
model) were removed in 0.1.0: on six datasets, three label draws, none beat the forest by enough
to justify keeping it - hgb -2.34 on average, CatBoost -0.09, TabPFN +0.52 but a GPU, torch and
7 to 12 times the run time (research/scripts/modal_models.py, research/results/model_review.json; docs/guide.md
"Which classifier"). CatBoost came back averaged with the forest: bagging and boosting err
differently, and the pair beat either alone (rerun on the current code, CatBoost alone scored
+0.10 on average against the forest, the average +0.41).
"""
MODELS = ("forest", "forest_catboost")
NEEDS_CATBOOST = '"matcher_model": "forest_catboost" needs CatBoost'
REMOVED = ("hgb", "catboost", "tabpfn", "auto")


def factory(cfg, seed=0):
    """A zero-argument function returning a fresh, unfitted classifier."""
    name = cfg.get("matcher_model", "forest")
    if name == "forest":
        from sklearn.ensemble import RandomForestClassifier
        return lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                              class_weight="balanced_subsample")
    if name == "forest_catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ValueError(f"{NEEDS_CATBOOST}: pip install \"custmatch[catboost]\"") from None
        forest = factory({"matcher_model": "forest"}, seed)
        return lambda: Averaged(forest(), CatBoostClassifier(
            iterations=500, auto_class_weights="Balanced", random_seed=seed, verbose=False, thread_count=-1,
            allow_writing_files=False))
    if name in REMOVED:
        raise ValueError(f'matcher_model "{name}" was removed in 0.1.0 (see CHANGELOG.md); use one of {MODELS}')
    raise ValueError(f"matcher_model must be one of {MODELS}, not {name!r}")


class Averaged:
    """Several classifiers' P(match), averaged. A training fold holding one class fits nothing and
    predicts that class, as p_match does for any single-class model."""
    def __init__(self, *models):
        self.models = models

    def fit(self, X, y):
        import numpy as np
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            for m in self.models:
                m.fit(X, y)
        return self

    def predict_proba(self, X):
        import numpy as np
        if len(self.classes_) == 1:
            p = np.full(len(X), float(self.classes_[0] == 1))
        else:
            p = np.mean([p_match(m, X) for m in self.models], axis=0)   # each member by its fastest path
        return np.c_[1 - p, p]

    @property
    def feature_importances_(self):
        """Each member's importances scaled to sum to 1, averaged (CatBoost's sum to 100)."""
        import numpy as np
        return np.mean([m.feature_importances_ / max(m.feature_importances_.sum(), 1e-12) for m in self.models], axis=0)


def load_model(state):
    """A run's saved model (state/model.pkl). One trained with forest_catboost needs CatBoost to
    load: without it, say how to install it rather than fail on the import."""
    import pickle
    from pathlib import Path

    from .io import ConfigError
    try:
        with open(Path(state) / "model.pkl", "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] != "catboost":
            raise
        raise ConfigError(f"this run's model was trained with {NEEDS_CATBOOST}: "
                          'pip install "custmatch[catboost]"') from None


PREDICT_CHUNK = 500_000


SMALL_BATCH = 20_000
PREDICT_CHUNK_THREAD = 100_000   # rows per thread in p_match's forest path


def _leaf_p(model):
    """Each tree with its leaves' P(match): a node's class fractions, as predict_proba uses them."""
    out = []
    for t in model.estimators_:
        v = t.tree_.value[:, 0, :]
        out.append((t.tree_, v[:, 1] / v.sum(axis=1)))
    return out


def _forest_proba(model, X, leaves=None):
    """A random forest's P(match), tree by tree in a fixed order: each row's leaf looked up in
    every tree and the leaves' fractions averaged - what predict_proba computes, without its
    thread pool, which costs more than the work below about 20,000 rows (11.5 s of sleeping in a
    million-record run, from hundreds of small calls in cross-fitting and the threshold search).
    The service's real-time matcher does the same."""
    import numpy as np
    leaves = leaves or _leaf_p(model)
    X = np.ascontiguousarray(X, dtype=np.float32)
    total = np.zeros(len(X))
    for tree, p in leaves:
        total += p[tree.apply(X)]
    return total / len(leaves)


def _is_forest(model):
    from sklearn.ensemble import RandomForestClassifier
    return isinstance(model, RandomForestClassifier) and list(model.classes_) == [0, 1]


def p_match(model, X, rows=None):
    """P(class 1). A model whose training fold held one class has a single probability column.
    rows: P(class 1) of X[rows], gathered a chunk at a time rather than copied whole first.

    Predicted PREDICT_CHUNK rows at a time: a random forest's predict_proba has every worker
    thread build a rows x 2 float64 array per tree, which at 69 million candidate pairs and 16
    threads was gigabytes per thread. Chunking bounds it; the probabilities are the same."""
    import numpy as np
    n = len(X) if rows is None else len(rows)

    def take(a, b):
        return X[a:b] if rows is None else X[rows[a:b]]
    if not n:
        return np.zeros(0)
    if list(model.classes_) != [0, 1] and len(model.classes_) == 1:
        return np.full(n, float(list(model.classes_)[0] == 1))
    if _is_forest(model):
        if n <= SMALL_BATCH:
            return _forest_proba(model, take(0, n))
        # sklearn's predict_proba adds the trees' votes from several threads in whatever order
        # they finish, so a probability can differ in its last bit between runs - enough to
        # reorder tied pairs in active learning and change which pairs are asked. Threads over
        # rows instead, each adding its trees in order: the same answer every time.
        import os
        from concurrent.futures import ThreadPoolExecutor
        leaves = _leaf_p(model)
        workers = (os.cpu_count() or 1) if (model.n_jobs or 1) < 0 else (model.n_jobs or 1)
        with ThreadPoolExecutor(workers) as pool:
            parts = list(pool.map(lambda s: _forest_proba(model, take(s, s + PREDICT_CHUNK_THREAD), leaves),
                                  range(0, n, PREDICT_CHUNK_THREAD)))
        return np.concatenate(parts)
    out = np.empty(n)
    for s in range(0, n, PREDICT_CHUNK):
        proba = model.predict_proba(take(s, s + PREDICT_CHUNK))
        out[s:s + len(proba)] = proba[:, 1] if proba.shape[1] == 2 else float(list(model.classes_)[0] == 1)
    return out


def distinct_rows(X, check=500_000):
    """(rows, inverse): indices of X's distinct rows, bit for bit, and each row's position among
    them, so X[rows][inverse] is X. A model scores equal rows equally, and among candidate pairs
    about 44% of feature rows repeat another (NCVR, a million people), so each is scored once.
    Rows are grouped by a hash of their bits and each checked against its group's first row; one
    whose hash collides with a different row's is kept as a row of its own."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd
    w = np.ascontiguousarray(X).reshape(len(X), -1).view(np.uint32)
    h = np.zeros(len(w), np.uint64)
    starts = range(0, len(w), check)

    def hash_rows(s):                                    # slices on every core: numpy frees the GIL
        part = h[s:s + check]
        for k in range(w.shape[1]):
            part *= np.uint64(0x9E3779B97F4A7C15)        # wraps: uint64 arithmetic is modulo 2**64
            part += w[s:s + check, k]
    with ThreadPoolExecutor(os.cpu_count() or 1) as pool:
        list(pool.map(hash_rows, starts))
        inverse, keys = pd.factorize(h)
        rows = np.empty(len(keys), np.int64)
        rows[inverse[::-1]] = np.arange(len(w) - 1, -1, -1)  # each group's first row
        odd = list(pool.map(lambda s: s + np.flatnonzero((w[s:s + check] != w[rows[inverse[s:s + check]]]).any(axis=1)),
                            starts))
    odd = np.concatenate(odd) if odd else np.zeros(0, np.int64)
    if len(odd):
        inverse[odd] = len(rows) + np.arange(len(odd))
        rows = np.concatenate([rows, odd])
    return rows, inverse


def p_match_rows(models, X):
    """p_match(m, X) for each model, each distinct row of X scored once (distinct_rows)."""
    if len(X) <= SMALL_BATCH:
        return [p_match(m, X) for m in models]
    rows, inverse = distinct_rows(X)
    return [p_match(m, X, rows)[inverse] for m in models]      # no copy of the distinct rows


def score_pairs(models, df, pairs, profiles, fopts, transform=None, keep=None):
    """Each model's P(match) for every pair, from features made core.SCORE_CHUNK pairs at a time,
    so the features of every pair are never held at once (on all 5M NCVR records, 192M pairs, they
    were the largest thing in memory). transform(pair rows, X) adds columns (the match weight).
    The next block's features are made in a second thread while the models score this one:
    features run mostly on one core, the forests on all of them.
    keep: for each model, None (its probabilities) or a function of a block's probabilities giving
    what to keep instead (core.threshold_bands: a byte a pair, not eight)."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    from .core import SCORE_CHUNK, feature_chunks, join_pieces
    keep = keep or [None] * len(models)
    out = [None] * len(models)

    def gathered():
        # feature chunks gathered to SCORE_CHUNK rows, so the forest's threads are kept busy;
        # joined freeing each chunk as it is copied (core.join_pieces), not held twice
        start, parts = 0, []
        for s, X in feature_chunks(df, pairs, profiles, **fopts):
            if parts and sum(map(len, parts)) + len(X) > SCORE_CHUNK:
                yield start, join_pieces(parts, None)
                start, parts = s, []
            parts.append(X)
            del X
        yield start, join_pieces(parts, None)

    blocks = gathered()
    with ThreadPoolExecutor(1) as ahead:
        pending = ahead.submit(next, blocks, None)
        while (block := pending.result()) is not None:
            pending = ahead.submit(next, blocks, None)
            s, X = block
            if transform is not None:
                X = transform(pairs[s:s + len(X)], X)
            for m, prob in enumerate(p_match_rows(models, X)):
                if keep[m] is not None:
                    prob = keep[m](prob)
                if out[m] is None:
                    out[m] = np.empty(len(pairs), prob.dtype)
                out[m][s:s + len(X)] = prob
            del block, X
    return [np.empty(0) if o is None else o for o in out]
