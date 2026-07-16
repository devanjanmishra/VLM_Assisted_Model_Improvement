"""VLM-powered automated ground-truth generation for flagged images.

Sends each flagged image to a vision-language model which classifies it
against the 43 GTSRB classes. Accepted labels become pseudo ground truth
used to retrain the model. This is the core of the CI/CD loop.

Backends:
  ollama     — local inference, e.g. `ollama pull qwen2.5vl:7b`
  anthropic  — set ANTHROPIC_API_KEY env var

  python pipeline/label.py                              # ollama default
  python pipeline/label.py --backend anthropic          # Anthropic API
  python pipeline/label.py --limit 300                  # cap calls for cost control

Writes: outputs/pseudo_labels.csv  (resumable — reruns skip done paths)
"""
import argparse
import base64
import difflib
import io
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from common import NAMES, NUM_CLASSES

ROOT = Path(__file__).resolve().parent.parent

# Upscale to this before sending — GTSRB crops can be as small as 15px
VLM_SIZE = 336

PROMPT = (
    "This is a cropped photo of a German traffic sign. "
    "It may be small, blurry, or partially occluded.\n"
    "Classify it as exactly one of these 43 classes:\n"
    + "\n".join(f"{i}: {n}" for i, n in enumerate(NAMES))
    + "\n\nReply ONLY with valid JSON, no extra text:\n"
    '{"class_id": <int 0-42>, "class_name": "<name>", "confidence": <0.0-1.0>}'
)


def encode(path: Path) -> tuple[str, str]:
    img = Image.open(path).convert("RGB").resize((VLM_SIZE, VLM_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


_NORM_MAP = {_norm(n): i for i, n in enumerate(NAMES)}


def parse(text: str) -> tuple[int | None, float]:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None, 0.0
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None, 0.0
    try:
        conf = float(obj.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    # numeric class_id is primary signal
    cid = obj.get("class_id")
    if isinstance(cid, (int, float)) and not isinstance(cid, bool) and 0 <= int(cid) < NUM_CLASSES:
        return int(cid), conf
    # fallback: fuzzy-match class_name string
    name = _norm(str(obj.get("class_name", "")))
    if name in _NORM_MAP:
        return _NORM_MAP[name], conf
    close = difflib.get_close_matches(name, _NORM_MAP.keys(), n=1, cutoff=0.85)
    return (_NORM_MAP[close[0]], conf) if close else (None, conf)


def call_ollama(path: Path, model: str, host: str) -> str:
    b64, _ = encode(path)
    r = requests.post(f"{host}/api/chat", json={
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
        "options": {"temperature": 0},
    }, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


def call_anthropic(path: Path, model: str) -> str:
    b64, mt = encode(path)
    r = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }, json={
        "model": model, "max_tokens": 200,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}},
            {"type": "text", "text": PROMPT},
        ]}],
    }, timeout=60)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def main(args):
    flagged = pd.read_csv(ROOT / "outputs" / "flagged.csv")
    if args.limit:
        flagged = flagged.head(args.limit)

    out = ROOT / "outputs" / "pseudo_labels.csv"
    rows, done = [], set()
    if out.exists():
        prev = pd.read_csv(out)
        rows, done = prev.to_dict("records"), set(prev["path"])
        print(f"Resuming: {len(done)} already labeled")

    accepted = rejected = errors = 0
    for _, row in tqdm(flagged.iterrows(), total=len(flagged), desc="VLM labeling"):
        if row["path"] in done:
            continue
        try:
            text = (call_ollama(ROOT / row["path"], args.model, args.ollama_host)
                    if args.backend == "ollama"
                    else call_anthropic(ROOT / row["path"], args.model))
        except Exception as e:
            errors += 1
            tqdm.write(f"Error: {row['path']}: {e}")
            continue

        idx, conf = parse(text)
        if idx is not None and conf >= args.min_conf:
            rows.append({"path": row["path"], "label": idx,
                         "vlm_name": NAMES[idx], "vlm_conf": conf,
                         "weight": args.pseudo_weight})
            accepted += 1
        else:
            rejected += 1

        if (accepted + rejected) % 50 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nAccepted: {accepted}  |  Rejected: {rejected}  |  Errors: {errors}")

    # Audit label quality against hidden ground truth (never used in training)
    if rows:
        df = pd.DataFrame(rows).merge(pd.read_csv(ROOT / "splits" / "pool.csv"),
                                      on="path", how="left")
        acc = (df["label"] == df["true_label"]).mean()
        print(f"VLM label accuracy vs ground truth: {acc:.3f}")
        if acc < 0.85:
            print("  ⚠  Below 0.85 — consider raising --min-conf or using a stronger model.")
    print("→ outputs/pseudo_labels.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",      choices=["ollama", "anthropic"], default="ollama")
    ap.add_argument("--model",        default=None)
    ap.add_argument("--ollama-host",  default="http://localhost:11434")
    ap.add_argument("--min-conf",     type=float, default=0.70)
    ap.add_argument("--pseudo-weight",type=float, default=0.70)
    ap.add_argument("--limit",        type=int,   default=None)
    args = ap.parse_args()
    if args.model is None:
        args.model = "qwen2.5vl:7b" if args.backend == "ollama" else "claude-haiku-4-5"
    main(args)
