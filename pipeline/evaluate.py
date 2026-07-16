"""Compare baseline vs improved on the held-out GTSRB test set.

Per-class regression gate: if any class drops more than --regress-tol
the improved model is NOT promoted and the script exits with code 1
(so CI fails and the bad checkpoint is never shipped).

  python pipeline/evaluate.py
Exit 0 — improved promoted to checkpoints/promoted.pt
Exit 1 — regression detected, baseline retained
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import CsvDataset, EVAL_TFMS, NUM_CLASSES, build_model, get_device, evaluate

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def build_test_loader(batch_size, workers):
    rows = []
    test_dir = ROOT / "data" / "gtsrb" / "test"
    for c in range(NUM_CLASSES):
        for f in sorted((test_dir / f"{c:02d}").glob("*.jpg")):
            rows.append({"path": str(f.relative_to(ROOT)), "label": c})
    csv = ROOT / "outputs" / "_test.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return DataLoader(CsvDataset(csv, EVAL_TFMS, ROOT),
                      batch_size=batch_size, num_workers=workers)


def load_model(name: str, dev):
    m = build_model().to(dev)
    m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=dev))
    return m


def main(args):
    dev = get_device()
    loader = build_test_loader(args.batch_size, args.workers)

    results = {}
    for name in ["baseline", "improved"]:
        overall, per_class = evaluate(load_model(name, dev), loader, dev)
        results[name] = (overall, per_class)
        print(f"{name:9s}  test acc: {overall:.4f}")

    b_acc, b_pc = results["baseline"]
    i_acc, i_pc = results["improved"]
    delta = i_acc - b_acc
    print(f"\nOverall Δ: {delta:+.4f}\n")
    print(f"{'Class':<44} {'baseline':>9} {'improved':>9} {'Δ':>7}")
    print("─" * 73)

    regressions = []
    for cls in b_pc:
        d = i_pc[cls] - b_pc[cls]
        flag = "  ← REGRESSION" if d < -args.regress_tol else ""
        print(f"{cls:<44} {b_pc[cls]:>9.3f} {i_pc[cls]:>9.3f} {d:>+7.3f}{flag}")
        if d < -args.regress_tol:
            regressions.append(cls)

    # Write JSON report (useful as a CI artefact)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": {"baseline": round(b_acc, 4), "improved": round(i_acc, 4), "delta": round(delta, 4)},
        "regressions": regressions,
        "per_class": {cls: {"baseline": round(b_pc[cls], 4), "improved": round(i_pc[cls], 4),
                             "delta": round(i_pc[cls] - b_pc[cls], 4)} for cls in b_pc},
    }
    out = ROOT / "outputs" / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport → outputs/eval_report.json")

    if regressions:
        print(f"\n✗  {len(regressions)} class(es) regressed beyond {args.regress_tol:.0%}.")
        print("   Improve VLM label quality (raise --min-conf) or lower finetune LR.")
        print("   Baseline retained.")
        sys.exit(1)
    else:
        import shutil
        shutil.copy(CKPT / "improved.pt", CKPT / "promoted.pt")
        print(f"\n✓  No regression detected. Promoted → checkpoints/promoted.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regress-tol", type=float, default=0.01)
    ap.add_argument("--batch-size",  type=int,   default=128)
    ap.add_argument("--workers",     type=int,   default=2)
    main(ap.parse_args())
