"""The service's smoke test (server/smoke_test.py) on SQLite: a batch run published, lookups equal
to the batch output, real-time matching, one customer for simultaneous arrivals, the write lease,
and a second run reconciling the API's records; and the real-time replay. Skipped where the server's dependencies are absent."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("fastapi", "sqlalchemy", "httpx")),
                    reason="pip install ./server (fastapi, sqlalchemy) to run the service tests")
def test_service_smoke_on_sqlite(monkeypatch, capsys):
    monkeypatch.delenv("CUSTMATCH_DB_URL", raising=False)
    sys.path.insert(0, str(ROOT / "server"))
    spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "server/smoke_test.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    assert smoke.main() == 0, capsys.readouterr().out[-2000:]


@pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("fastapi", "sqlalchemy")),
                    reason="pip install ./server (fastapi, sqlalchemy) to run the service tests")
def test_realtime_replay_keeps_up_with_the_batch_run():
    """server/replay_test.py: half the demo through the real-time matcher one record at a time.
    Real-time grouping must stay close to a batch run over the same records, and about as
    accurate (research/results/server_replay.json, five seeds: 86.78 agreement, 81.67 vs the batch run's
    81.91; merging every linked customer fell to 80.46)."""
    sys.path.insert(0, str(ROOT / "server"))
    spec = importlib.util.spec_from_file_location("replay_test", ROOT / "server/replay_test.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    # the mean of three seeds, as the published figures are: one seed's real-time F1 swings by
    # about 3 points either way with small changes to the first batch's model
    runs = [rp.replay(first=0.5, seed=seed) for seed in (0, 1, 2)]
    mean = {k: sum(r[k] for r in runs) / len(runs) for k in ("agreement_f1", "realtime_f1_vs_truth", "batch_f1_vs_truth")}
    assert mean["agreement_f1"] >= 85, runs
    assert mean["realtime_f1_vs_truth"] >= mean["batch_f1_vs_truth"] - 3, runs
