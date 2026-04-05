import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DNNModel(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 224 * 224 * 3):
        super().__init__()
        # TODO: define fully-connected layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: flatten input and implement forward pass
        raise NotImplementedError


def train_dnn(
    model: DNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> DNNModel:
    """Train the DNN model and return the trained model.

    Use val_loader each epoch to monitor overfitting and save the best checkpoint.
    Do NOT use test_loader here — it is reserved for final evaluation only.
    """
    # TODO: implement training loop
    raise NotImplementedError


def test_dnn(
    model: DNNModel,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """Evaluate the DNN on the test set. Returns a dict with accuracy and loss.

    Call this once after training is complete. Use val_loader during training instead.
    """
    # TODO: implement evaluation
    raise NotImplementedError
