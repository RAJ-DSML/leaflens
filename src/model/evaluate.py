import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data.dataset import PlantVillageDataset, get_val_transforms, DATA_DIR
from src.model.clip_model import CLIPClassifier

def evaluate():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load("models/best_model.pt", map_location=device)
    config = checkpoint["config"]
    classes = checkpoint["classes"]

    model = CLIPClassifier(
        num_classes=len(classes),
        model_name=config["model_name"],
        pretrained=config["pretrained"],
        dropout=config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Test data
    test_ds = PlantVillageDataset(DATA_DIR, split="test", transform=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=150)
    plt.show()
    print("\nConfusion matrix saved to logs/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()