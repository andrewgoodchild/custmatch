"""Train Ditto (RoBERTa) on the person-record splits and score the test candidate graph.

Uses the authors' DittoDataset / DittoModel so the serialisation and model are theirs; adds
10% warmup, since the reproduction runs showed Ditto's zero-warmup recipe collapsing on some
datasets. Writes per-pair match probabilities for the clustering pass.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils import data
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/ditto"))
from ditto_light.dataset import DittoDataset  # noqa: E402
from ditto_light.ditto import DittoModel  # noqa: E402

DATA = ROOT / "data/person_em"


def probs(model, loader, device):
    out = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            out.append(model(x).softmax(dim=1)[:, 1].cpu().numpy())
            if device == "mps":
                torch.mps.empty_cache()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--data-dir", default=str(DATA), help="folder with train/valid/test.txt")
    ap.add_argument("--out", default="person", help="writes research/results/<out>_scores_ditto.npz "
                                                    "and research/results/<out>_ditto_train.json")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    tr = DittoDataset(str(data_dir / "train.txt"), max_len=args.max_len, lm="roberta")
    va = DittoDataset(str(data_dir / "valid.txt"), max_len=args.max_len, lm="roberta")
    te = DittoDataset(str(data_dir / "test.txt"), max_len=args.max_len, lm="roberta")
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,} device={device}", flush=True)

    tr_it = data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=tr.pad)
    va_it = data.DataLoader(va, batch_size=args.batch_size * 2, shuffle=False, collate_fn=va.pad)
    te_it = data.DataLoader(te, batch_size=args.batch_size * 2, shuffle=False, collate_fn=te.pad)

    model = DittoModel(device=device, lm="roberta").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = (len(tr) // args.batch_size) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_frac * steps), steps)
    crit = torch.nn.CrossEntropyLoss()

    y_va = np.array(va.labels)
    best = {"f1": -1.0}
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for i, (x, y) in enumerate(tr_it):
            opt.zero_grad()
            loss = crit(model(x), y.to(device))
            loss.backward()
            opt.step()
            sched.step()
            if i % 50 == 0:
                if device == "mps":
                    torch.mps.empty_cache()
                print(f"  epoch {ep} step {i}/{len(tr_it)} loss {loss.item():.4f}", flush=True)
        p_va = probs(model, va_it, device)
        th, f1 = max(((t, f1_score(y_va, p_va > t)) for t in np.arange(0.05, 0.96, 0.05)),
                     key=lambda z: z[1])
        print(f"epoch {ep}: valid_f1={100 * f1:.2f} th={th:.2f} ({time.time() - t0:.0f}s)",
              flush=True)
        if f1 > best["f1"]:
            best = {"f1": f1, "th": float(th), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    print(f"best epoch {best['epoch']} valid_f1={100 * best['f1']:.2f}", flush=True)
    model.load_state_dict(best["state"])
    t0 = time.time()
    p_te = probs(model, te_it, device)
    print(f"test inference on {len(p_te):,} pairs in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(ROOT / f"research/results/{args.out}_scores_ditto.npz", prob=p_te,
                        y=np.array(te.labels), threshold=best["th"])
    json.dump({"valid_f1": round(100 * best["f1"], 2), "threshold": best["th"],
               "epoch": best["epoch"], "epochs": args.epochs, "max_len": args.max_len,
               "batch_size": args.batch_size, "warmup_frac": args.warmup_frac},
              open(ROOT / f"research/results/{args.out}_ditto_train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
