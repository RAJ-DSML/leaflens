import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time

# from src.data.dataset import PlantVillageDataset, get_train_transforms, get_val_transforms, DATA_DIR
# from src.model.clip_model import CLIPClassifier

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import PlantVillageDataset, get_train_transforms, get_val_transforms, DATA_DIR
from src.model.clip_model import CLIPClassifier

# --- Config ---
# --- First Run ---
# CONFIG = {
#     "model_name"   : "ViT-B-32",
#     "pretrained"   : "openai",
#     "num_classes"  : 15,
#     "batch_size"   : 64,
#     "num_epochs"   : 10,
#     "lr"           : 3e-4,
#     "dropout"      : 0.3,
#     "weight_decay" : 1e-4,
#     "num_workers"  : 2,
#     "unfreeze_epoch": 5,   # unfreeze last 2 CLIP blocks at this epoch
#     "unfreeze_lr"  : 1e-5, # lower LR for unfrozen backbone
# }

# --- Run 2 — Push toward 90%+ ---
CONFIG = {
    "model_name"    : "ViT-B-32",
    "pretrained"    : "openai",
    "num_classes"   : 15,
    "batch_size"    : 64,
    "num_epochs"    : 20,
    "lr"            : 3e-4,
    "dropout"       : 0.2,
    "weight_decay"  : 1e-4,
    "num_workers"   : 2,
    "unfreeze_epoch": 3,    # unfreeze earlier
    "unfreeze_lr"   : 3e-5, # slightly higher than before
}

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0) # accumulate total loss
        correct += (outputs.argmax(1) == labels).sum().item() # accumulate correct predictions
        total += images.size(0) # accumulate total samples

    return total_loss / total, correct / total


@torch.no_grad() # no need to track gradients during evaluation
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set and return average loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train():
    """Main training loop."""
    device = get_device()
    print(f"Device: {device}")

    # Data
    train_ds = PlantVillageDataset(DATA_DIR, split="train", transform=get_train_transforms())
    val_ds   = PlantVillageDataset(DATA_DIR, split="val",   transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], pin_memory=False)

    # Model
    model = CLIPClassifier(
        num_classes=CONFIG["num_classes"],
        model_name=CONFIG["model_name"],
        pretrained=CONFIG["pretrained"],
        dropout=CONFIG["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )

    # MLflow
    mlflow.set_experiment("leaflens-clip") # <-- Change this to your desired experiment name

    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        best_val_acc = 0.0
        save_path = Path("models/best_model.pt")
        save_path.parent.mkdir(exist_ok=True)

        for epoch in range(1, CONFIG["num_epochs"] + 1):
            start = time.time()

            # Unfreeze backbone partway through
            if epoch == CONFIG["unfreeze_epoch"]:
                model.unfreeze_last_n_blocks(n=2)
                for pg in optimizer.param_groups:
                    pg["lr"] = CONFIG["unfreeze_lr"]
                print(f"\nEpoch {epoch}: Unfreezing backbone, LR → {CONFIG['unfreeze_lr']}")

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - start

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc" : train_acc,
                "val_loss"  : val_loss,
                "val_acc"   : val_acc,
            }, step=epoch)

            print(f"Epoch {epoch:02d}/{CONFIG['num_epochs']} | "
                  f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
                  f"{elapsed:.1f}s")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch"     : epoch,
                    "model_state": model.state_dict(),
                    "val_acc"   : val_acc,
                    "config"    : CONFIG,
                    "classes"   : train_ds.classes,
                }, save_path)
                print(f"  ✅ Best model saved (val_acc: {val_acc:.4f})")

        print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(str(save_path))


if __name__ == "__main__":
    train()