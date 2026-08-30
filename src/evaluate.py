import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from src.data import get_dataloaders
from src.model import build_model
from src.config import DEVICE, MODEL_DIR


def evaluate():
    # Load validation data
    _, val_loader, class_to_idx = get_dataloaders()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Load trained model
    checkpoint = torch.load(
        MODEL_DIR / "crop_disease_model.pth",
        map_location=DEVICE
    )

    model = build_model(len(class_to_idx))
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Metrics
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[idx_to_class[i] for i in range(len(idx_to_class))]
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        xticklabels=idx_to_class.values(),
        yticklabels=idx_to_class.values(),
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()
