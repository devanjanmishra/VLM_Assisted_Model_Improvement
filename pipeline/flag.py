"""Run the baseline over the unlabeled pool and flag predictions that are
low-confidence or have no clear winner between classes.

Flag if: max softmax prob < MIN_PROB  OR  (top1 − top2) margin < MIN_MARGIN

  python pipeline/flag.py
Writes: outputs/flagged.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import CsvDataset, EVAL_TFMS, NAMES, build_model, get_device

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def main(args):
    dev = get_device()
    model = build_model().to(dev)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "baseline.pt", map_location=dev))
    model.eval()

    pool = pd.read_csv(ROOT / "splits" / "pool.csv")
    # Temporarily add a dummy label column so CsvDataset can load
    pool_tmp = pool.copy()
    pool_tmp["label"] = 0
    pool_tmp.to_csv(ROOT / "splits" / "_pool_tmp.csv", index=False)

    loader = DataLoader(CsvDataset(ROOT / "splits" / "_pool_tmp.csv", EVAL_TFMS, ROOT),
                        batch_size=args.batch_size, num_workers=args.workers)

    probs_all = []
    for x, _, _ in loader:
        probs_all.append(torch.softmax(model(x.to(dev)), 1).cpu())
    probs = torch.cat(probs_all)

    top2 = probs.topk(2, dim=1)
    pool["top1"] = top2.indices[:, 0].numpy()
    pool["top2"] = top2.indices[:, 1].numpy()
    pool["top1_name"] = [NAMES[i] for i in pool["top1"]]
    pool["maxprob"] = top2.values[:, 0].numpy().round(4)
    pool["margin"] = (top2.values[:, 0] - top2.values[:, 1]).numpy().round(4)

    flagged = pool[(pool.maxprob < args.min_prob) | (pool.margin < args.min_margin)].copy()
    (ROOT / "outputs").mkdir(exist_ok=True)
    flagged.to_csv(ROOT / "outputs" / "flagged.csv", index=False)
    (ROOT / "splits" / "_pool_tmp.csv").unlink(missing_ok=True)

    confident = pool.drop(flagged.index)
    print(f"Pool: {len(pool):,}  |  flagged: {len(flagged):,} ({len(flagged)/len(pool):.1%})")
    if len(confident):
        print(f"Baseline acc on confident subset: {(confident.top1 == confident.true_label).mean():.3f}")
    print(f"Baseline acc on flagged subset:    {(flagged.top1 == flagged.true_label).mean():.3f}  ← VLM opportunity")
    print("→ outputs/flagged.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prob",   type=float, default=0.70)
    ap.add_argument("--min-margin", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int,   default=128)
    ap.add_argument("--workers",    type=int,   default=2)
    main(ap.parse_args())
