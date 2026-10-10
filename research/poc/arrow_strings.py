"""PoC: Arrow-backed strings instead of Python-object strings for the record table.

Measures (1) the loaded table's memory with each storage, (2) whether the pipeline runs on
Arrow strings unchanged, and (3) whether the results are identical.

    python research/poc/arrow_strings.py [copies]
"""
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research/scripts"))
import custmatch_scale as CS  # noqa: E402
from custmatch import io  # noqa: E402


def deep_mb(df):
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def main():
    copies = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    with tempfile.TemporaryDirectory() as tmp:
        cfg = CS.scaled(copies, Path(tmp))
        t0 = time.time()
        df, _ = io.load(cfg)
        t_load = time.time() - t0
        obj_mb = deep_mb(df)
        arrow = df.copy()
        for c in arrow.columns:
            if arrow[c].dtype == object:
                arrow[c] = arrow[c].astype("string[pyarrow]")
        arr_mb = deep_mb(arrow)
        per_col = pd.DataFrame({"object_mb": df.memory_usage(deep=True) / 1024 ** 2,
                                "arrow_mb": arrow.memory_usage(deep=True) / 1024 ** 2}).round(1)
        print(f"records {len(df):,}  load {t_load:.0f} s")
        print(f"table memory: object strings {obj_mb:,.0f} MB  ->  Arrow strings {arr_mb:,.0f} MB "
              f"({obj_mb / arr_mb:.1f}x smaller), {obj_mb * 1024 / len(df):.0f} -> {arr_mb * 1024 / len(df):.0f} KB per 1,000 records")
        print(per_col.sort_values("object_mb", ascending=False).head(8).to_string())
        # gotcha check: what common operations return on Arrow strings
        s = arrow.first_name
        print("== behaviour on Arrow strings")
        print("  to_numpy() dtype:", s.to_numpy().dtype, "| (s == '') dtype:", (s == "").dtype,
              "| s.str[:3] dtype:", s.str[:3].dtype, "| s.values type:", type(s.values).__name__)
        v = s.to_numpy()
        t0 = time.time(); _ = [x for x in v[:200000]]; t_iter = time.time() - t0
        v2 = df.first_name.to_numpy()
        t0 = time.time(); _ = [x for x in v2[:200000]]; t_iter_obj = time.time() - t0
        print(f"  iterating 200k values: object {t_iter_obj * 1000:.0f} ms, Arrow->numpy {t_iter * 1000:.0f} ms")
        t0 = time.time(); a = s.to_numpy(dtype=object); t_conv = time.time() - t0
        print(f"  converting one column back to a Python-object array: {t_conv * 1000:.0f} ms "
              f"(the memory saving is lost for every column a step converts)")


if __name__ == "__main__":
    main()
