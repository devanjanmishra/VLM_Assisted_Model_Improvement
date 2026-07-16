"""Run the full SignSight-GT pipeline in one command.

  python pipeline/run_pipeline.py
  python pipeline/run_pipeline.py --skip-train --backend anthropic --limit-vlm 300
  python pipeline/run_pipeline.py --workers 0   # Windows DataLoader fix

Exit 0 — improved model promoted
Exit 1 — regression gate failed; baseline retained
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def step(label: str, cmd: list) -> None:
    print(f"\n{'─'*60}\n  {label}\n{'─'*60}")
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    print(f"  ({time.perf_counter()-t0:.1f}s)")
    if r.returncode not in (0, 1):          # 1 is a valid regression-gate signal
        sys.exit(r.returncode)
    if r.returncode == 1:
        print("\n  Regression gate triggered — pipeline halted.")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data",  action="store_true", help="Data already prepared")
    ap.add_argument("--skip-train", action="store_true", help="Baseline checkpoint exists")
    ap.add_argument("--skip-label", action="store_true", help="pseudo_labels.csv exists")
    ap.add_argument("--backend",    choices=["ollama","anthropic"], default="ollama")
    ap.add_argument("--vlm-model",  default=None)
    ap.add_argument("--limit-vlm",  type=int, default=None)
    ap.add_argument("--workers",    type=int, default=2)
    args = ap.parse_args()

    if not args.skip_data:
        step("1 / 5  Prepare data",
             [PY, "pipeline/prepare_data.py"])

    if not args.skip_train:
        step("2 / 5  Train baseline",
             [PY, "pipeline/train.py", "--workers", args.workers])

    step("3 / 5  Flag uncertain predictions",
         [PY, "pipeline/flag.py", "--workers", args.workers])

    if not args.skip_label:
        cmd = [PY, "pipeline/label.py", "--backend", args.backend]
        if args.vlm_model:  cmd += ["--model", args.vlm_model]
        if args.limit_vlm:  cmd += ["--limit", args.limit_vlm]
        step("4 / 5  VLM ground-truth generation", cmd)

    step("5 / 5  Fine-tune + regression-gated evaluation",
         [PY, "pipeline/finetune.py", "--workers", args.workers])

    step("      Evaluate",
         [PY, "pipeline/evaluate.py", "--workers", args.workers])

    print("\n  Pipeline complete. Check outputs/eval_report.json")


if __name__ == "__main__":
    main()
