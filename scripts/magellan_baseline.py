"""Magellan-style baseline (Mudgal et al., SIGMOD 2018, "Magellan" column).

Per-attribute similarity features (as py_entitymatching's auto feature generation does),
then the best of DT / RF / LR / NB / SVM selected on the validation split, F1 on test.
Data: DeepMatcher splits as serialised in the Ditto repo (COL <attr> VAL <value> ...).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler, Jaro
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external/ditto/data/er_magellan"

REPORTED = {  # Mudgal et al. 2018, Tables 3-5, Magellan column
    "Structured/Amazon-Google": 49.1, "Structured/Beer": 78.8, "Structured/DBLP-ACM": 98.4,
    "Structured/DBLP-GoogleScholar": 92.3, "Structured/Fodors-Zagats": 100.0,
    "Structured/iTunes-Amazon": 91.2, "Structured/Walmart-Amazon": 71.9,
    "Dirty/DBLP-ACM": 91.9, "Dirty/DBLP-GoogleScholar": 82.5, "Dirty/iTunes-Amazon": 46.8,
    "Dirty/Walmart-Amazon": 37.4, "Textual/Abt-Buy": 43.6,
}


def parse(rec):
    parts = re.split(r"COL (\S+) VAL ", " " + rec.strip())
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        a, b, y = line.rstrip("\n").split("\t")
        rows.append((parse(a), parse(b), int(y)))
    return rows


def qgrams(s, q=3):
    s = f"##{s}$$"
    return {s[i:i + q] for i in range(len(s) - q + 1)}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def monge_elkan(ta, tb):
    if not ta or not tb:
        return 0.0
    return float(np.mean([max(JaroWinkler.similarity(x, y) for y in tb) for x in ta]))


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group()) if m else None


def features(a, b, attrs):
    f = []
    for k in attrs:
        x, y = a.get(k, "").lower(), b.get(k, "").lower()
        missing = float(not x or not y)
        tx, ty = x.split(), y.split()
        f += [
            missing,
            float(x == y and x != ""),
            jac(set(tx), set(ty)),
            jac(qgrams(x), qgrams(y)),
            Levenshtein.normalized_similarity(x, y),
            Jaro.similarity(x, y),
            JaroWinkler.similarity(x, y),
            monge_elkan(tx, ty),
        ]
        nx, ny = num(x), num(y)
        if nx is not None and ny is not None:
            f += [abs(nx - ny), abs(nx - ny) / max(abs(nx), abs(ny), 1e-9)]
        else:
            f += [-1.0, -1.0]
    return f


def build(rows, attrs):
    X = np.array([features(a, b, attrs) for a, b, _ in rows], dtype=float)
    y = np.array([r[2] for r in rows])
    return X, y


MODELS = {
    "DT": lambda: DecisionTreeClassifier(random_state=0),
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4),
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "NB": lambda: GaussianNB(),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC()),
}


def run(task):
    d = DATA / task
    tr, va, te = (load(d / f"{s}.txt") for s in ("train", "valid", "test"))
    attrs = sorted({k for a, b, _ in tr for k in (*a, *b)})
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (build(r, attrs) for r in (tr, va, te))
    scores = {}
    for name, mk in MODELS.items():
        m = mk().fit(Xtr, ytr)
        scores[name] = (f1_score(yva, m.predict(Xva)), m)
    best = max(scores, key=lambda n: scores[n][0])
    # refit best on train+valid, as Magellan does after model selection
    m = MODELS[best]().fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    test_f1 = 100 * f1_score(yte, m.predict(Xte))
    rf_f1 = 100 * f1_score(yte, scores["RF"][1].predict(Xte))
    return {"task": task, "best_model": best, "test_f1": round(test_f1, 2),
            "rf_test_f1": round(rf_f1, 2), "reported_magellan": REPORTED.get(task),
            "valid_f1": {k: round(100 * v[0], 2) for k, v in scores.items()}}


if __name__ == "__main__":
    tasks = sys.argv[1:] or list(REPORTED)
    out = []
    for t in tasks:
        r = run(t)
        out.append(r)
        print(json.dumps(r), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results/magellan_baseline.json", "w"), indent=1)
