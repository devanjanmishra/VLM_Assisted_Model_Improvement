"""Shared model, dataset, and evaluation utilities."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# GTSRB: 43 German traffic sign classes (index == class id)
NAMES = [
    "Speed limit 20",      "Speed limit 30",      "Speed limit 50",
    "Speed limit 60",      "Speed limit 70",      "Speed limit 80",
    "End speed limit 80",  "Speed limit 100",     "Speed limit 120",
    "No passing",          "No passing >3.5t",    "Right-of-way next",
    "Priority road",       "Yield",               "Stop",
    "No vehicles",         "No vehicles >3.5t",   "No entry",
    "General caution",     "Curve left",          "Curve right",
    "Double curve",        "Bumpy road",          "Slippery road",
    "Narrows right",       "Road work",           "Traffic signals",
    "Pedestrians",         "Children crossing",   "Bicycles crossing",
    "Ice/snow",            "Wild animals",        "End restrictions",
    "Turn right ahead",    "Turn left ahead",     "Ahead only",
    "Straight or right",   "Straight or left",    "Keep right",
    "Keep left",           "Roundabout",          "End no passing",
    "End no passing >3.5t",
]
NUM_CLASSES = len(NAMES)
IMG_SIZE = 64

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TFMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CsvDataset(Dataset):
    def __init__(self, csv_path, tfms, root="."):
        self.df = pd.read_csv(csv_path)
        self.tfms = tfms
        self.root = Path(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
        label = int(row["label"]) if "label" in row else -1
        weight = float(row.get("weight", 1.0))
        return self.tfms(img), label, weight


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = torch.zeros(NUM_CLASSES)
    total = torch.zeros(NUM_CLASSES)
    for x, y, _ in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(1)
        for c in range(NUM_CLASSES):
            mask = y == c
            total[c] += mask.sum().item()
            correct[c] += (pred[mask] == c).sum().item()
    per_class = {NAMES[c]: (correct[c] / max(total[c], 1)).item() for c in range(NUM_CLASSES)}
    return (correct.sum() / total.sum()).item(), per_class
