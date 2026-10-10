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


def _forest_proba(model, X):
    """A random forest's P(match) for a small batch, tree by tree: predict_proba's thread pool
    costs more than the work below about 20,000 rows - it was 11.5 s of sleeping in a
    million-record run, from hundreds of small calls in cross-fitting and the threshold search.
    The same per-tree class fractions, averaged (the service's real-time matcher does the same)."""
    import numpy as np
    X = np.ascontiguousarray(X, dtype=np.float32)
    total = np.zeros(len(X))
    for tree in model.estimators_:
        v = tree.tree_.predict(X)
        v = v[:, 0, :] if v.ndim == 3 else v
        total += v[:, 1] / v.sum(axis=1)
    return total / len(model.estimators_)


def p_match(model, X):
    """P(class 1). A model whose training fold held one class has a single probability column.

    Predicted PREDICT_CHUNK rows at a time: a random forest's predict_proba has every worker
    thread build a rows x 2 float64 array per tree, which at 69 million candidate pairs and 16
    threads was gigabytes per thread. Chunking bounds it; the probabilities are the same."""
    import numpy as np
    if not len(X):
        return np.zeros(0)
    if list(model.classes_) != [0, 1] and len(model.classes_) == 1:
        return np.full(len(X), float(list(model.classes_)[0] == 1))
    if len(X) <= SMALL_BATCH and hasattr(model, "estimators_") and list(model.classes_) == [0, 1]:
        return _forest_proba(model, X)
    out = np.empty(len(X))
    for s in range(0, len(X), PREDICT_CHUNK):
        proba = model.predict_proba(X[s:s + PREDICT_CHUNK])
        out[s:s + len(proba)] = proba[:, 1] if proba.shape[1] == 2 else float(list(model.classes_)[0] == 1)
    return out
