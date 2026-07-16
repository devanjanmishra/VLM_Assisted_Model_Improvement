"""Download GTSRB via torchvision and create seed / unlabeled-pool splits.

  python pipeline/prepare_data.py                   # full dataset (~280 MB)
  python pipeline/prepare_data.py --max-per-class 80  # quick smoke-test

Creates:
  data/gtsrb/<train|test>/<00..42>/*.jpg
  splits/seed.csv      path,label          — 5% of train, treated as "labeled"
  splits/pool.csv      path,true_label     — remaining 95%, unlabeled pool
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from torchvision.datasets import GTSRB
from tqdm import tqdm

from common import NUM_CLASSES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "gtsrb"


def materialize():
    for split in ["train", "test"]:
        out = DATA / split
        if out.exists():
            print(f"{out} already materialized — skipping.")
            continue
        ds = GTSRB(root=DATA, split=split, download=True)
        print(f"Saving {split} ({len(ds)} images)...")
        counts = [0] * NUM_CLASSES
        for img, label in tqdm(ds):
            d = out / f"{label:02d}"
            d.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(d / f"{counts[label]:05d}.jpg", quality=95)
            counts[label] += 1
    print("Done →", DATA)


def make_splits(seed_frac: float, max_per_class: int | None, rng_seed: int = 42):
    rng = random.Random(rng_seed)
    seed_rows, pool_rows = [], []
    for c in range(NUM_CLASSES):
        files = sorted((DATA / "train" / f"{c:02d}").glob("*.jpg"))
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        n_seed = max(2, int(len(files) * seed_frac))
        for f in files[:n_seed]:
            seed_rows.append({"path": str(f.relative_to(ROOT)), "label": c})
        for f in files[n_seed:]:
            pool_rows.append({"path": str(f.relative_to(ROOT)), "true_label": c})
    (ROOT / "splits").mkdir(exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(ROOT / "splits" / "seed.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(ROOT / "splits" / "pool.csv", index=False)
    print(f"seed.csv: {len(seed_rows):,} | pool.csv: {len(pool_rows):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-frac", type=float, default=0.05)
    ap.add_argument("--max-per-class", type=int, default=None)
    args = ap.parse_args()
    materialize()
    make_splits(args.seed_frac, args.max_per_class)
