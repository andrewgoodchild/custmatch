"""PoC (adopted in core._standard_features): the 13 standard features without a Python loop per pair.

Result at 1,006,700 records, 2,437,950 candidate pairs, on a laptop: loop 7.7 s, vectorised 2.7 s
(2.8x), every value identical. The original loop is kept below as the baseline.

core._standard_features built each pair's 13 values in Python (a loop over pairs); everything else
in core.features was already vectorised with numpy and rapidfuzz.process.cpdist. This builds the
same 13 columns column by column, checks every value is identical to the loop's, and times both.

    python poc/vectorised_features.py [copies]     (the synthetic file copied, as custmatch_scale.py)
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def standard_features_loop(df, pairs, profiles, aware=True):
    """core._standard_features before: one Python loop iteration per pair."""
    fn, ln, mi = df.first_name.to_numpy(), df.last_name.to_numpy(), df.middle_initial.to_numpy()
    num, st, unit = df.addr_number.to_numpy(), df.addr_street.to_numpy(), df.addr_unit.to_numpy()
    em, ph = df.email.to_numpy(), df.phone.to_numpy()
    dob, ssn = df.date_of_birth.fillna("").to_numpy(), df.ssn.fillna("").to_numpy()
    X = np.zeros((len(pairs), 13), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        e_match = float(em[i] != "" and em[i] == em[j])
        p_match = float(ph[i] != "" and ph[i] == ph[j])
        if aware:
            e_w = profiles["email"].get(em[i], ("personal", 1.0))[1] if e_match else 0.0
            p_w = profiles["phone"].get(ph[i], ("personal", 1.0))[1] if p_match else 0.0
        else:
            e_w, p_w = e_match, p_match
        sim = JaroWinkler.similarity
        X[k] = [sim(fn[i], fn[j]), sim(ln[i], ln[j]), sim(fn[i] + ln[i], fn[j] + ln[j]),
                float(mi[i] != "" and mi[i] == mi[j]), float(num[i] != "" and num[i] == num[j]),
                sim(st[i], st[j]), float(unit[i] != "" and unit[i] == unit[j]),
                e_match, e_w, p_match, p_w,
                float(dob[i] != "" and dob[i] == dob[j]) if (dob[i] and dob[j]) else -1.0,
                float(ssn[i] != "" and ssn[i] == ssn[j]) if (ssn[i] and ssn[j]) else -1.0]
    return X


def standard_features_vectorised(df, pairs, profiles, aware=True):
    i, j = pairs[:, 0], pairs[:, 1]
    col = lambda c: df[c].fillna("").to_numpy(dtype=object)  # noqa: E731
    fn, ln, mi, num, st, unit, em, ph, dob, ssn = (col(c) for c in (
        "first_name", "last_name", "middle_initial", "addr_number", "addr_street", "addr_unit", "email", "phone",
        "date_of_birth", "ssn"))
    jw = lambda a, b: cpdist(list(a), list(b), scorer=JaroWinkler.similarity, workers=-1)  # noqa: E731
    agree = lambda v: (v[i] != "") & (v[i] == v[j])  # noqa: E731
    full = (df.first_name.fillna("") + df.last_name.fillna("")).to_numpy(dtype=object)
    e_match, p_match = agree(em), agree(ph)

    def weight(field, v, match):
        if not aware:
            return match.astype(float)
        prof = profiles[field]
        w = pd.Series({k: t[1] for k, t in prof.items()}, dtype=float)
        got = w.reindex(v[i]).to_numpy() if len(w) else np.full(len(i), np.nan)
        return np.where(match, np.nan_to_num(got, nan=1.0), 0.0)        # a value not profiled weighs 1.0

    both = lambda v: (v[i] != "") & (v[j] != "")  # noqa: E731
    X = np.column_stack([
        jw(fn[i], fn[j]), jw(ln[i], ln[j]), jw(full[i], full[j]),
        agree(mi), agree(num), jw(st[i], st[j]), agree(unit),
        e_match, weight("email", em, e_match), p_match, weight("phone", ph, p_match),
        np.where(both(dob), agree(dob), -1.0), np.where(both(ssn), agree(ssn), -1.0)])
    return X.astype(np.float32)


def main(copies=50):
    import custmatch_scale as CS
    from custmatch import io
    from custmatch.core import _standard_features, block, profile_contacts
    tmp = Path(tempfile.mkdtemp())
    df, _ = io.load(CS.scaled(copies, tmp))
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs, _ = block(df, prof)
    print(f"{len(df):,} records, {len(pairs):,} candidate pairs", flush=True)
    t = time.perf_counter()
    old = standard_features_loop(df, pairs, prof)
    t_old = time.perf_counter() - t
    t = time.perf_counter()
    new = _standard_features(df, pairs, prof)          # core's, as adopted
    t_new = time.perf_counter() - t
    same = np.array_equal(old, new)
    diff = float(np.abs(old - new).max()) if old.shape == new.shape else None
    print(f"loop {t_old:.1f} s, vectorised {t_new:.1f} s: {t_old / t_new:.1f}x; identical: {same} (max difference {diff})")
    for flag in (False,):
        o2 = standard_features_loop(df, pairs[:20000], prof, aware=flag)
        n2 = _standard_features(df, pairs[:20000], prof, aware=flag)
        print(f"aware={flag}: identical {np.array_equal(o2, n2)}")
    return same


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 50) else 1)
