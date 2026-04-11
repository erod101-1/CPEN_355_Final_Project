import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from visualize import show_sample_predictions


class ResNet50Model(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        # Replace the final fully-connected layer to match num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_resnet50(
    model: ResNet50Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> ResNet50Model:
    """Train the ResNet50 model and return the trained model.

    Use val_loader each epoch to monitor overfitting and save the best checkpoint.
    Do NOT use test_loader here — it is reserved for final evaluation only.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", unit="batch")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  -> New best ResNet50 model (val_loss={val_loss:.4f})")

    model.load_state_dict(best_weights)
    return model


def test_resnet50(
    model: ResNet50Model,
    test_loader: DataLoader,
    device: str = "cpu",
    class_names: list = None,
    show_grid: bool = True,
) -> dict:
    """Evaluate ResNet50 on the test set. Returns a dict with accuracy, loss,
    per-class precision/recall/F1, and saves a confusion matrix plot.

    Call this once after training is complete. Use val_loader during training instead.
    """
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    inference_start = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing ResNet50", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    inference_time = time.time() - inference_start

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    target_names = class_names if class_names else [str(i) for i in range(len(set(all_labels)))]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    print(f"\n--- ResNet50 Classification Report ---")
    print(f"Inference time: {inference_time:.2f}s total | {inference_time / total * 1000:.3f}ms per image")
    print(report)

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("ResNet50 Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/resnet50_confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved to models/resnet50_confusion_matrix.png")

    # Sample image grid
    if show_grid:
        show_sample_predictions(model, test_loader, target_names, device=device, model_name="ResNet50")

    return {
        "accuracy": correct / total,
        "loss": test_loss / total,
        "inference_time_s": round(inference_time, 4),
        "inference_ms_per_image": round(inference_time / total * 1000, 4),
        "classification_report": report_dict,
    }
