"""CI helper: inject simulated pseudo-labels so the pipeline can run end-to-end
without making real VLM API calls. Simulates 90% label accuracy with noise."""
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from common import NAMES, NUM_CLASSES

random.seed(42)
flagged = pd.read_csv(ROOT / "outputs" / "flagged.csv")
rows = []
for _, r in flagged.iterrows():
    label = int(r["true_label"]) if random.random() < 0.90 else random.randrange(NUM_CLASSES)
    rows.append({"path": r["path"], "label": label,
                 "vlm_name": NAMES[label], "vlm_conf": 0.85, "weight": 0.70})

(ROOT / "outputs").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(ROOT / "outputs" / "pseudo_labels.csv", index=False)
print(f"Injected {len(rows)} pseudo-labels (simulated, 90% accuracy)")
