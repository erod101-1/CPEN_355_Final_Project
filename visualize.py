import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# ImageNet normalization stats (must match prepare_dataset.py)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denormalize(tensor):
    """Convert a normalized image tensor (C, H, W) back to a displayable numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0, 1)


def show_sample_predictions(model: nn.Module, test_loader: DataLoader,
                             class_names: list, device: str = "cpu",
                             num_samples: int = 16, model_name: str = "Model"):
    """Display a grid of test images with predicted and true labels.

    Titles are green when the prediction is correct, red when wrong.

    Args:
        model:       Trained model (ResNet50Model, CNNModel, or DNNModel).
        test_loader: DataLoader for the test set.
        class_names: Ordered list of class name strings (from ImageFolder.classes).
        device:      Device the model is on.
        num_samples: Number of images to display (default 16, shown in a 4×4 grid).
        model_name:  Name shown in the plot title (e.g. "CNN", "DNN", "ResNet50").
    """
    model.eval()

    # Randomly sample indices from the full test set
    dataset = test_loader.dataset
    indices = random.sample(range(len(dataset)), num_samples)
    subset_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=num_samples,
        shuffle=False,
        num_workers=0,
    )

    images_all, labels_all = next(iter(subset_loader))
    with torch.no_grad():
        outputs = model(images_all.to(device))
    _, preds_all = torch.max(outputs.cpu(), 1)

    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(_denormalize(images_all[i]))
        ax.axis("off")

        pred_name  = class_names[preds_all[i]]
        true_name  = class_names[labels_all[i]]
        correct    = preds_all[i] == labels_all[i]
        color      = "green" if correct else "red"

        ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}", color=color, fontsize=8)

    # Hide any unused subplot slots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{model_name} — Sample Predictions (green = correct, red = wrong)", fontsize=12)
    plt.tight_layout()
    plt.show()
