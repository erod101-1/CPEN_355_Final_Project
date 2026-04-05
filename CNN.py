import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CNNModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # TODO: define convolutional layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass
        raise NotImplementedError


def train_cnn(
    model: CNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> CNNModel:
    """Train the CNN model and return the trained model.

    Use val_loader each epoch to monitor overfitting and save the best checkpoint.
    Do NOT use test_loader here — it is reserved for final evaluation only.
    """
    # TODO: implement training loop
    raise NotImplementedError


def test_cnn(
    model: CNNModel,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """Evaluate the CNN on the test set. Returns a dict with accuracy and loss.

    Call this once after training is complete. Use val_loader during training instead.
    """
    # TODO: implement evaluation
    raise NotImplementedError
