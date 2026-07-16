# VLM_Assisted_Model_Improvement (SignSight-GT)

**Automated ground-truth generation and CI/CD-style model lifecycle management for traffic sign classifiers.**

A VLM (vision-language model) continuously labels the images a lightweight CNN is uncertain about, turning those labels into new training data. Each cycle ends with a regression-gated evaluation: if any class accuracy drops the improved model is rejected and the pipeline exits non-zero — so it can block a CI merge or a deployment.

Built on **GTSRB** (43 German traffic sign classes), a real ADAS perception task where fine-grained confusion (speed limit digits, triangular warning family) and hard imaging conditions (motion blur, sub-30px crops, low light) make this problem genuinely non-trivial.

---

## How it works

```
Labeled seed (5% of train)
        │
        ▼
  ┌─────────────┐
  │  Train      │  MobileNetV3-Small on labeled seed
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Flag       │  Detect low-confidence / no-clear-winner predictions
  └──────┬──────┘  on the unlabeled pool
         │
         ▼
  ┌─────────────┐
  │  Label      │  VLM (Qwen2.5-VL or Claude) classifies flagged images
  └──────┬──────┘  → automated ground truth
         │
         ▼
  ┌─────────────┐
  │  Fine-tune  │  Retrain on seed + pseudo-labels (weighted loss replay)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Evaluate   │  Per-class regression gate on held-out test set
  └──────┬──────┘
         │
    ┌────┴────┐
    ✓ promote  ✗ reject
    promoted.pt  (baseline retained, CI fails)
```

---

## Quickstart

```bash
# 1. Install
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# GPU (CUDA 12.x):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Data  (~280 MB, one-time download via torchvision)
python pipeline/prepare_data.py

# 3. Full pipeline  (Ollama backend)
python pipeline/run_pipeline.py --backend ollama

# 3b. Full pipeline  (Anthropic API)
ANTHROPIC_API_KEY=sk-... python pipeline/run_pipeline.py --backend anthropic
```

**Windows note:** if DataLoader hangs, add `--workers 0` to any command.

**Quick sanity check** (80 imgs/class, no real VLM call):
```bash
python pipeline/prepare_data.py --max-per-class 80
python pipeline/train.py --head-epochs 1 --ft-epochs 1 --workers 0
python pipeline/flag.py --workers 0
python tests/inject_pseudo_labels.py     # simulates VLM output
python pipeline/finetune.py --epochs 1 --workers 0
python pipeline/evaluate.py --workers 0
```

---

## Pipeline scripts

| Script | What it does |
|---|---|
| `pipeline/prepare_data.py` | Download GTSRB, materialize to JPEGs, create seed / pool splits |
| `pipeline/train.py` | Train MobileNetV3-Small on the labeled seed (frozen backbone → full fine-tune) |
| `pipeline/flag.py` | Detect low-confidence predictions on the unlabeled pool |
| `pipeline/label.py` | VLM classifies flagged images; audits label quality vs hidden ground truth |
| `pipeline/finetune.py` | Retrain on seed + pseudo-labels with weighted loss and early stopping |
| `pipeline/evaluate.py` | Per-class comparison; promotes `improved.pt` → `promoted.pt` only if no regression |
| `pipeline/run_pipeline.py` | Single command to run the whole cycle |

---

## VLM backends

```bash
# Ollama (local, free)
ollama pull qwen2.5vl:7b
python pipeline/label.py --backend ollama --model qwen2.5vl:7b

# Anthropic API
python pipeline/label.py --backend anthropic --model claude-haiku-4-5 --limit 300
```

`--limit N` caps VLM calls for cost control. The labeler is resumable — interrupted runs pick up where they left off.

---

## Anti-degradation design

| Guard | Where |
|---|---|
| VLM label only accepted if confidence ≥ 0.70 | `label.py --min-conf` |
| Pseudo-labels down-weighted in loss (default 0.7) | `label.py --pseudo-weight` |
| Seed data always replayed at full weight | `finetune.py` |
| Fine-tune LR conservative (5e-5) + early stopping | `finetune.py` |
| Improved checkpoint initialized from baseline | `finetune.py` |
| Per-class regression gate on held-out test set | `evaluate.py --regress-tol` |
| Label quality audited vs hidden ground truth | `label.py` (step 4 output) |

---

## CI/CD

GitHub Actions runs three jobs on every push:

1. **test** — ruff lint + pytest (parser unit tests, no GPU, ~30s)
2. **smoke** — mini dataset, simulated VLM labels, full pipeline on CPU (~5 min)
3. **full** — real VLM, manual trigger via `workflow_dispatch`; uploads `promoted.pt` as artefact

The smoke job uses `tests/inject_pseudo_labels.py` to simulate 90%-accurate VLM output so the pipeline can be tested without API keys in CI.

---

## What to expect

| Stage | Typical result |
|---|---|
| Baseline (5% labels) | 85–92% test accuracy |
| Flagged by uncertainty detector | 10–25% of pool |
| VLM label accuracy (audit, step 4) | 0.87–0.93 with `qwen2.5vl:7b` |
| After fine-tune | +2–4 pts overall, concentrated on speed-limit and warning classes |
| Regression check | 0–2 classes flagged on a good run; 0 means safe to promote |

If step 4 prints VLM accuracy < 0.85, raise `--min-conf` to 0.80 before fine-tuning.

---

## Dataset

[GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html) — German Traffic Sign Recognition Benchmark.  
~39k train / ~12.6k test, 43 classes, downloaded automatically via `torchvision.datasets.GTSRB`.

---

## Stack

- **Model:** MobileNetV3-Small (2.5M params, ImageNet-pretrained) — light enough for edge deployment, real enough to have a meaningful accuracy ceiling on GTSRB
- **VLM:** Qwen2.5-VL 7B (Ollama) or Claude Haiku (Anthropic API)
- **Framework:** PyTorch + torchvision
- **CI:** GitHub Actions
