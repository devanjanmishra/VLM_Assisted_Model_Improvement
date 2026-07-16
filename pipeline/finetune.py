"""Fine-tune baseline on seed data + VLM pseudo-labels.

Anti-degradation safeguards:
  - Seed data is always replayed at full weight (prevents forgetting)
  - Pseudo-labels carry reduced loss weight (default 0.7, set during label.py)
  - Low LR (5e-5) + early stopping on seed dev split
  - Improved checkpoint initialized from baseline — can never be worse on dev

  python pipeline/finetune.py
Saves: checkpoints/improved.pt
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


def main(args):
    dev = get_device()

    seed = pd.read_csv(ROOT / "splits" / "seed_train.csv").assign(weight=1.0)
    pseudo = pd.read_csv(ROOT / "outputs" / "pseudo_labels.csv")[["path", "label", "weight"]]
    combined = pd.concat([seed[["path", "label", "weight"]], pseudo], ignore_index=True)
    combined.to_csv(ROOT / "outputs" / "_finetune_train.csv", index=False)
    print(f"Fine-tune set: {len(seed):,} seed + {len(pseudo):,} pseudo = {len(combined):,}")

    tr = DataLoader(CsvDataset(ROOT / "outputs" / "_finetune_train.csv", TRAIN_TFMS, ROOT),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dv = DataLoader(CsvDataset(ROOT / "splits" / "seed_dev.csv", EVAL_TFMS, ROOT),
                    batch_size=args.batch_size, num_workers=args.workers)

    model = build_model().to(dev)
    model.load_state_dict(torch.load(CKPT / "baseline.pt", map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(reduction="none")

    @torch.no_grad()
    def dev_acc():
        model.eval()
        c = n = 0
        for x, y, _ in dv:
            c += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
            n += len(y)
        return c / n

    best, patience = dev_acc(), 0
    print(f"Baseline dev acc: {best:.3f}")
    torch.save(model.state_dict(), CKPT / "improved.pt")

    for e in range(args.epochs):
        model.train()
        for x, y, w in tr:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            loss = (crit(model(x), y) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        va = dev_acc()
        print(f"  [ft {e+1}] dev {va:.3f}")
        if va > best:
            best, patience = va, 0
            torch.save(model.state_dict(), CKPT / "improved.pt")
        else:
            patience += 1
            if patience >= 3:
                print("  Early stop."); break

    print(f"Saved checkpoints/improved.pt  (best dev {best:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr",         type=float, default=5e-5)
    ap.add_argument("--epochs",     type=int,   default=8)
    ap.add_argument("--batch-size", type=int,   default=128)
    ap.add_argument("--workers",    type=int,   default=2)
    main(ap.parse_args())
