"""Train baseline MobileNetV3-Small on the labeled seed split.

  python pipeline/train.py
Saves: checkpoints/baseline.pt
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import CsvDataset, TRAIN_TFMS, EVAL_TFMS, build_model, get_device

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def carve_dev(seed_csv, frac=0.10, rng=42):
    df = pd.read_csv(seed_csv)
    dev_idx = []
    for _, g in df.groupby("label"):
        dev_idx += list(g.sample(n=max(1, round(len(g) * frac)), random_state=rng).index)
    dev = df.loc[dev_idx]
    tr = df.drop(dev_idx)
    tr.to_csv(ROOT / "splits" / "seed_train.csv", index=False)
    dev.to_csv(ROOT / "splits" / "seed_dev.csv", index=False)


def run_epoch(model, loader, dev, opt=None):
    model.train(opt is not None)
    crit = nn.CrossEntropyLoss(reduction="none")
    loss_sum = correct = n = 0
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for x, y, w in loader:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            out = model(x)
            loss = (crit(out, y) * w).mean()
            if opt:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, correct / n


def main(args):
    dev = get_device()
    print(f"Device: {dev}")
    carve_dev(ROOT / "splits" / "seed.csv")

    tr = DataLoader(CsvDataset(ROOT / "splits" / "seed_train.csv", TRAIN_TFMS, ROOT),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dv = DataLoader(CsvDataset(ROOT / "splits" / "seed_dev.csv", EVAL_TFMS, ROOT),
                    batch_size=args.batch_size, num_workers=args.workers)

    model = build_model().to(dev)

    # Phase 1 — head only
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    for e in range(args.head_epochs):
        _, ta = run_epoch(model, tr, dev, opt)
        _, va = run_epoch(model, dv, dev)
        print(f"  [head {e+1}] train {ta:.3f} | dev {va:.3f}")

    # Phase 2 — full fine-tune with early stopping
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best, patience = 0.0, 0
    CKPT.mkdir(exist_ok=True)
    for e in range(args.ft_epochs):
        _, ta = run_epoch(model, tr, dev, opt)
        _, va = run_epoch(model, dv, dev)
        print(f"  [full {e+1}] train {ta:.3f} | dev {va:.3f}")
        if va > best:
            best, patience = va, 0
            torch.save(model.state_dict(), CKPT / "baseline.pt")
        else:
            patience += 1
            if patience >= 3:
                print("  Early stop."); break

    print(f"Saved checkpoints/baseline.pt  (best dev {best:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    main(ap.parse_args())
