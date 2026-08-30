"""Evaluate the saved checkpoint on the held-out validation split.

Rebuilds the exact 80/20 stratified split from src/data.py (RANDOM_SEED=42),
so the images scored here are the ones train.py never saw — provided the
checkpoint was trained with the committed code and seed. Writes artifacts to
evaluation/ instead of showing interactive plots.

Run:  python -m src.eval_holdout
"""

import json
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.config import BASE_DIR, DEVICE, MODEL_DIR
from src.data import get_dataloaders
from src.model import build_model

OUT_DIR = BASE_DIR / "evaluation"
OUT_DIR.mkdir(exist_ok=True)


def main():
    _, val_loader, class_to_idx = get_dataloaders()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    checkpoint = torch.load(MODEL_DIR / "crop_disease_model.pth",
                            map_location=DEVICE)
    if checkpoint.get("class_to_idx") != class_to_idx:
        raise SystemExit("Checkpoint class_to_idx does not match the dataset "
                         "on disk — split reconstruction would be invalid.")

    model = build_model(len(class_to_idx))
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    report_txt = classification_report(y_true, y_pred,
                                       target_names=class_names, digits=4)
    metrics = {
        "date": date.today().isoformat(),
        "n_eval_images": len(y_true),
        "n_classes": len(class_names),
        "split": "20% stratified hold-out, random_state=42 (src/data.py)",
        "device": DEVICE,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "per_class": classification_report(y_true, y_pred,
                                           target_names=class_names,
                                           output_dict=True),
    }

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT_DIR / "classification_report.txt").write_text(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix — held-out split (n={len(y_true)})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    print(f"n={len(y_true)}  accuracy={acc:.4f}  "
          f"f1_macro={metrics['f1_macro']}  f1_weighted={metrics['f1_weighted']}")
    print(report_txt)


if __name__ == "__main__":
    main()
